#pragma once

#include "Types.hpp"
#include <filesystem>
#include <string>

namespace meme {

class CudaInferenceEngine {
public:
    CudaInferenceEngine() = default;
    ~CudaInferenceEngine();

    CudaInferenceEngine(const CudaInferenceEngine&) = delete;
    CudaInferenceEngine& operator=(const CudaInferenceEngine&) = delete;

    bool loadWeights(const std::filesystem::path& bin_path);
    [[nodiscard]] bool isReady() const noexcept { return ready_; }
    bool forward(const FeatureVector& input, std::array<float, kNumMemeClasses>& out_logits) const;

private:
    bool ready_{false};
    float* d_weights_{nullptr};
    float* d_workspace_{nullptr};
    std::size_t total_floats_{0};
};

} // namespace meme
