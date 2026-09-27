#pragma once

#include "Types.hpp"
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace meme {

class PoseDetector {
public:
    PoseDetector() = default;

    bool loadModels(const std::filesystem::path& onnx_path);

    [[nodiscard]] bool isLoaded() const noexcept { return onnx_loaded_; }

    PoseSkeleton detect(const cv::Mat& bgr_frame);

    static void drawSkeletonOverlay(
        cv::Mat& bgr_frame,
        const PoseSkeleton& kp,
        const std::string& gesture_title,
        float confidence,
        const std::string& gpu_summary
    );

private:
    cv::dnn::Net pose_net_;
    bool onnx_loaded_{false};
    bool has_prev_kp_{false};
    PoseSkeleton prev_kp_{};
};

} // namespace meme
