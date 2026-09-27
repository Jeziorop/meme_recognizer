#pragma once

#include "CudaInference.hpp"
#include "Types.hpp"
#include <filesystem>
#include <opencv2/dnn.hpp>
#include <vector>

namespace meme {

class MemeClassifier {
public:
    MemeClassifier() = default;

    bool loadManifest(const std::filesystem::path& manifest_json_path);
    bool loadModels(
        const std::filesystem::path& onnx_path,
        const std::filesystem::path& cuda_bin_path = {}
    );

    [[nodiscard]] bool isReady() const noexcept { return onnx_loaded_ || cuda_engine_.isReady(); }
    [[nodiscard]] const std::vector<MemeClassInfo>& classes() const noexcept { return classes_; }

    PredictionResult classify(const FeatureVector& features, float ema_alpha = 0.40f);
    void resetSmoothing() noexcept;

private:
    std::vector<MemeClassInfo> classes_;
    cv::dnn::Net dnn_net_;
    CudaInferenceEngine cuda_engine_;
    bool onnx_loaded_{false};
    bool has_ema_{false};
    std::array<float, kNumMemeClasses> ema_probs_{};
};

} // namespace meme
