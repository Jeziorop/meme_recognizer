#pragma once

#include <opencv2/dnn.hpp>
#include <string>

namespace meme {

struct GpuStatus {
    bool cuda_available{false};
    bool opencl_available{false};
    std::string cuda_device_name{"None"};
    std::string opencl_device_name{"None"};
    int compute_major{0};
    int compute_minor{0};
    std::string active_summary{"CPU"};
};

class GpuDeviceManager {
public:
    static GpuStatus initialize();
    static void configureDnnNetForGpu(cv::dnn::Net& net);
};

} // namespace meme
