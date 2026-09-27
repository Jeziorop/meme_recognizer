#include "CudaInference.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>

namespace meme {

namespace {

// Exact GELU matching PyTorch nn.GELU() (erf formulation)
__device__ __forceinline__ float deviceGelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

// Single-block high-speed cooperative kernel for MemePoseClassifier [54 -> 128 -> Res1(128) -> Res2(128) -> 8]
// Runs entirely in fast shared memory on Quadro P620 (sm_61) with zero intermediate host-device transfers!
__global__ void memePoseClassifierKernel(
    const float* __restrict__ d_in,      // [54]
    const float* __restrict__ d_weights, // Packed parameters
    float* __restrict__ d_out            // [8]
) {
    __shared__ float s_in[54];
    __shared__ float s_x[128];
    __shared__ float s_h1[128];
    __shared__ float s_h2[128];
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    const int tid = threadIdx.x; // 0..127

    if (tid < 54) {
        s_in[tid] = d_in[tid];
    }
    __syncthreads();

    // Parameter offsets inside d_weights
    // 1. stem_fc: W[128*54], b[128]
    // 2. stem_ln: W[128], b[128]
    const float* p = d_weights;
    const float* stem_fc_w = p; p += 128 * 54;
    const float* stem_fc_b = p; p += 128;
    const float* stem_ln_w = p; p += 128;
    const float* stem_ln_b = p; p += 128;

    auto applyLayerNorm128 = [&](float* buf, const float* gamma, const float* beta) {
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int i = 0; i < 128; ++i) sum += buf[i];
            s_mean = sum / 128.0f;
            float sq = 0.0f;
            for (int i = 0; i < 128; ++i) {
                float d = buf[i] - s_mean;
                sq += d * d;
            }
            s_inv_std = rsqrtf((sq / 128.0f) + 1e-5f);
        }
        __syncthreads();
        if (tid < 128) {
            buf[tid] = ((buf[tid] - s_mean) * s_inv_std) * gamma[tid] + beta[tid];
        }
        __syncthreads();
    };

    // Stem: Linear(54 -> 128) + LayerNorm + GELU
    if (tid < 128) {
        float acc = stem_fc_b[tid];
        const float* row = stem_fc_w + tid * 54;
        for (int j = 0; j < 54; ++j) {
            acc += row[j] * s_in[j];
        }
        s_x[tid] = acc;
    }
    applyLayerNorm128(s_x, stem_ln_w, stem_ln_b);
    if (tid < 128) {
        s_x[tid] = deviceGelu(s_x[tid]);
    }
    __syncthreads();

    // Two Residual Blocks (res1 and res2)
    for (int r = 0; r < 2; ++r) {
        const float* fc1_w = p; p += 128 * 128;
        const float* fc1_b = p; p += 128;
        const float* ln1_w = p; p += 128;
        const float* ln1_b = p; p += 128;
        const float* fc2_w = p; p += 128 * 128;
        const float* fc2_b = p; p += 128;
        const float* ln2_w = p; p += 128;
        const float* ln2_b = p; p += 128;

        if (tid < 128) {
            float acc = fc1_b[tid];
            const float* row = fc1_w + tid * 128;
            for (int j = 0; j < 128; ++j) {
                acc += row[j] * s_x[j];
            }
            s_h1[tid] = acc;
        }
        applyLayerNorm128(s_h1, ln1_w, ln1_b);
        if (tid < 128) {
            s_h1[tid] = deviceGelu(s_h1[tid]);
        }
        __syncthreads();

        if (tid < 128) {
            float acc = fc2_b[tid];
            const float* row = fc2_w + tid * 128;
            for (int j = 0; j < 128; ++j) {
                acc += row[j] * s_h1[j];
            }
            s_h2[tid] = acc;
        }
        applyLayerNorm128(s_h2, ln2_w, ln2_b);
        if (tid < 128) {
            s_x[tid] = deviceGelu(s_x[tid] + s_h2[tid]);
        }
        __syncthreads();
    }

    // Head: Linear(128 -> 8)
    const float* head_w = p; p += 8 * 128;
    const float* head_b = p;
    if (tid < 8) {
        float acc = head_b[tid];
        const float* row = head_w + tid * 128;
        for (int j = 0; j < 128; ++j) {
            acc += row[j] * s_x[j];
        }
        d_out[tid] = acc;
    }
}

} // namespace

CudaInferenceEngine::~CudaInferenceEngine() {
    if (d_weights_ != nullptr) {
        cudaFree(d_weights_);
        d_weights_ = nullptr;
    }
    if (d_workspace_ != nullptr) {
        cudaFree(d_workspace_);
        d_workspace_ = nullptr;
    }
}

bool CudaInferenceEngine::loadWeights(const std::filesystem::path& bin_path) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
        return false;
    }
    cudaSetDevice(0);

    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    std::uint32_t header[5]{};
    ifs.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!ifs || header[0] != 0x4D454D45u || header[2] != 54u || header[3] != 128u || header[4] != 8u) {
        return false;
    }

    ifs.seekg(0, std::ios::end);
    const std::size_t file_bytes = static_cast<std::size_t>(ifs.tellg()) - sizeof(header);
    total_floats_ = file_bytes / sizeof(float);
    std::vector<float> host_weights(total_floats_);
    ifs.seekg(sizeof(header), std::ios::beg);
    ifs.read(reinterpret_cast<char*>(host_weights.data()), static_cast<std::streamsize>(file_bytes));
    if (!ifs) {
        return false;
    }

    if (d_weights_ != nullptr) cudaFree(d_weights_);
    if (d_workspace_ != nullptr) cudaFree(d_workspace_);

    if (cudaMalloc(&d_weights_, file_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(d_weights_, host_weights.data(), file_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_weights_);
        d_weights_ = nullptr;
        return false;
    }

    if (cudaMalloc(&d_workspace_, (kFeatureDim + kNumMemeClasses) * sizeof(float)) != cudaSuccess) {
        cudaFree(d_weights_);
        d_weights_ = nullptr;
        return false;
    }

    ready_ = true;
    return true;
}

bool CudaInferenceEngine::forward(
    const FeatureVector& input,
    std::array<float, kNumMemeClasses>& out_logits
) const {
    if (!ready_ || d_weights_ == nullptr || d_workspace_ == nullptr) {
        return false;
    }

    float* d_in = d_workspace_;
    float* d_out = d_workspace_ + kFeatureDim;

    if (cudaMemcpy(d_in, input.data(), kFeatureDim * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    memePoseClassifierKernel<<<1, 128>>>(d_in, d_weights_, d_out);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    if (cudaMemcpy(out_logits.data(), d_out, kNumMemeClasses * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    return true;
}

} // namespace meme
