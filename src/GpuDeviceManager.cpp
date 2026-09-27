#include "GpuDeviceManager.hpp"

#include <opencv2/core/ocl.hpp>
#include <sstream>

#if defined(MEME_ENABLE_CUDA) && MEME_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace meme {

GpuStatus GpuDeviceManager::initialize() {
    GpuStatus status{};

#if defined(MEME_ENABLE_CUDA) && MEME_ENABLE_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            cudaSetDevice(0);
            status.cuda_available = true;
            status.cuda_device_name = prop.name;
            status.compute_major = prop.major;
            status.compute_minor = prop.minor;
        }
    }
#endif

    // Disable OpenCV's built-in OpenCL runtime to avoid ocl4dnn compiler errors on NVIDIA drivers
    // (-cl-no-subgroup-ifp unrecognized argument)
    cv::ocl::setUseOpenCL(false);

    if (status.cuda_available) {
        status.active_summary = "Hardware Acceleration: GPU";
    } else {
        status.active_summary = "Hardware Acceleration: CPU Fallback";
    }
    return status;
}

void GpuDeviceManager::configureDnnNetForGpu(cv::dnn::Net& net) {
    const auto cuda_targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_CUDA);
    if (!cuda_targets.empty()) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        return;
    }

    if (cv::ocl::useOpenCL()) {
        const auto ocl_targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV);
        for (const auto t : ocl_targets) {
            if (t == cv::dnn::DNN_TARGET_OPENCL) {
                net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                return;
            }
        }
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

} // namespace meme
