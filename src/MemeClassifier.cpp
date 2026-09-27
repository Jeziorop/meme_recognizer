#include "MemeClassifier.hpp"
#include "GpuDeviceManager.hpp"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace meme {

bool MemeClassifier::loadManifest(const std::filesystem::path& manifest_json_path) {
    QFile file(QString::fromStdString(manifest_json_path.string()));
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) {
        return false;
    }

    const QJsonArray arr = doc.object().value("classes").toArray();
    classes_.clear();
    const auto base_dir = manifest_json_path.parent_path();

    for (const QJsonValue& val : arr) {
        const QJsonObject obj = val.toObject();
        MemeClassInfo info{};
        info.index = obj.value("index").toInt();
        info.id = obj.value("id").toString().toStdString();
        info.title = obj.value("title").toString().toStdString();
        info.subtitle = obj.value("subtitle").toString().toStdString();
        info.hint = obj.value("hint").toString().toStdString();
        info.accent_hex = obj.value("accent_hex").toString().toStdString();
        info.image_path = (base_dir / obj.value("image").toString().toStdString()).string();
        classes_.push_back(std::move(info));
    }

    return classes_.size() == kNumMemeClasses;
}

bool MemeClassifier::loadModels(
    const std::filesystem::path& onnx_path,
    const std::filesystem::path& cuda_bin_path
) {
    if (!cuda_bin_path.empty() && std::filesystem::exists(cuda_bin_path)) {
        cuda_engine_.loadWeights(cuda_bin_path);
    }

    if (std::filesystem::exists(onnx_path)) {
        try {
            dnn_net_ = cv::dnn::readNetFromONNX(onnx_path.string());
            GpuDeviceManager::configureDnnNetForGpu(dnn_net_);
            onnx_loaded_ = !dnn_net_.empty();
        } catch (const cv::Exception&) {
            onnx_loaded_ = false;
        }
    }

    return isReady();
}

void MemeClassifier::resetSmoothing() noexcept {
    has_ema_ = false;
    ema_probs_.fill(0.0f);
}

PredictionResult MemeClassifier::classify(const FeatureVector& features, float ema_alpha) {
    PredictionResult result{};
    std::array<float, kNumMemeClasses> logits{};
    bool inferred = false;

    if (cuda_engine_.isReady()) {
        inferred = cuda_engine_.forward(features, logits);
        if (inferred) {
            result.backend_used = "GPU Acceleration (CUDA)";
        }
    }

    if (!inferred && onnx_loaded_) {
        cv::Mat input_blob(1, static_cast<int>(kFeatureDim), CV_32F);
        std::copy(features.begin(), features.end(), input_blob.ptr<float>());
        dnn_net_.setInput(input_blob);
        cv::Mat out = dnn_net_.forward();
        if (out.total() >= kNumMemeClasses) {
            const float* ptr = out.ptr<float>();
            for (std::size_t i = 0; i < kNumMemeClasses; ++i) {
                logits[i] = ptr[i];
            }
            inferred = true;
            result.backend_used = "OpenCV DNN (CPU Fallback)";
        }
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    std::array<float, kNumMemeClasses> probs{};
    for (std::size_t i = 0; i < kNumMemeClasses; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    const float inv_sum = 1.0f / std::max(sum_exp, 1e-8f);
    for (float& p : probs) {
        p *= inv_sum;
    }

    const float alpha = std::clamp(ema_alpha, 0.05f, 1.0f);
    if (!has_ema_ || alpha >= 0.99f) {
        ema_probs_ = probs;
        has_ema_ = true;
    } else {
        float norm = 0.0f;
        for (std::size_t i = 0; i < kNumMemeClasses; ++i) {
            ema_probs_[i] = alpha * probs[i] + (1.0f - alpha) * ema_probs_[i];
            norm += ema_probs_[i];
        }
        for (float& p : ema_probs_) {
            p /= std::max(norm, 1e-8f);
        }
    }

    result.probabilities = ema_probs_;
    const auto max_it = std::max_element(ema_probs_.begin(), ema_probs_.end());
    result.class_index = static_cast<int>(std::distance(ema_probs_.begin(), max_it));
    result.confidence = *max_it;
    return result;
}

} // namespace meme
