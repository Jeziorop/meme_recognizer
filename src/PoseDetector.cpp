#include "PoseDetector.hpp"
#include "GpuDeviceManager.hpp"
#include "PoseFeatureExtractor.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace meme {

bool PoseDetector::loadModels(const std::filesystem::path& onnx_path) {
    if (std::filesystem::exists(onnx_path)) {
        try {
            pose_net_ = cv::dnn::readNetFromONNX(onnx_path.string());
            GpuDeviceManager::configureDnnNetForGpu(pose_net_);
            onnx_loaded_ = !pose_net_.empty();
        } catch (const cv::Exception&) {
            onnx_loaded_ = false;
        }
    }
    return onnx_loaded_;
}

PoseSkeleton PoseDetector::detect(const cv::Mat& bgr_frame) {
    PoseSkeleton kp = PoseFeatureExtractor::getCanonicalPose(static_cast<int>(kNumMemeClasses) - 1, false);
    if (bgr_frame.empty() || !onnx_loaded_) {
        return kp;
    }

    const float orig_w = static_cast<float>(bgr_frame.cols);
    const float orig_h = static_cast<float>(bgr_frame.rows);

    // Prepare letterbox 384x384 input for YOLOv8n-pose
    constexpr int net_size = 384;
    const float scale = std::min(static_cast<float>(net_size) / orig_w, static_cast<float>(net_size) / orig_h);
    const int new_w = static_cast<int>(std::round(orig_w * scale));
    const int new_h = static_cast<int>(std::round(orig_h * scale));
    const int pad_x = (net_size - new_w) / 2;
    const int pad_y = (net_size - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr_frame, resized, cv::Size(new_w, new_h));
    cv::Mat input_img(net_size, net_size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(input_img(cv::Rect(pad_x, pad_y, new_w, new_h)));

    cv::Mat blob = cv::dnn::blobFromImage(
        input_img,
        1.0 / 255.0,
        cv::Size(net_size, net_size),
        cv::Scalar(0, 0, 0),
        true,   // swapRB: BGR -> RGB
        false,
        CV_32F
    );

    pose_net_.setInput(blob);
    cv::Mat out = pose_net_.forward(); // Shape: [1, 56, 3024]

    // Parse YOLOv8-pose output: 4 box coords + 1 person conf + 17 * 3 keypoints = 56 rows
    // Rows:
    // 0..3: cx, cy, w, h
    // 4: person confidence
    // 5..55: keypoint (x, y, conf) for 17 keypoints
    const int channels = out.size[1]; // 56
    const int num_anchors = out.size[2]; // 3024

    if (channels >= 56 && num_anchors > 0) {
        float max_person_conf = 0.25f;
        int best_anchor = -1;

        const float* out_ptr = out.ptr<float>();
        for (int a = 0; a < num_anchors; ++a) {
            const float conf = out_ptr[4 * num_anchors + a];
            if (conf > max_person_conf) {
                max_person_conf = conf;
                best_anchor = a;
            }
        }

        if (best_anchor >= 0) {
            const float inv_scale = 1.0f / scale;
            for (std::size_t i = 0; i < kNumKeypoints; ++i) {
                const float kx = out_ptr[(5 + 3 * i + 0) * num_anchors + best_anchor];
                const float ky = out_ptr[(5 + 3 * i + 1) * num_anchors + best_anchor];
                const float kconf = out_ptr[(5 + 3 * i + 2) * num_anchors + best_anchor];

                // Unletterbox back to original image space
                const float real_x = (kx - pad_x) * inv_scale;
                const float real_y = (ky - pad_y) * inv_scale;

                kp[i].x = std::clamp(real_x, 0.0f, orig_w);
                kp[i].y = std::clamp(real_y, 0.0f, orig_h);
                kp[i].confidence = std::clamp(kconf, 0.0f, 1.0f);
            }
        }
    }

    // Temporal smoothing for rock-solid tracking
    if (has_prev_kp_) {
        constexpr float kAlpha = 0.55f;
        for (std::size_t i = 0; i < kNumKeypoints; ++i) {
            if (kp[i].confidence >= 0.30f) {
                kp[i].x = kAlpha * kp[i].x + (1.0f - kAlpha) * prev_kp_[i].x;
                kp[i].y = kAlpha * kp[i].y + (1.0f - kAlpha) * prev_kp_[i].y;
            } else {
                kp[i] = prev_kp_[i];
            }
        }
    }
    prev_kp_ = kp;
    has_prev_kp_ = true;
    return kp;
}

void PoseDetector::drawSkeletonOverlay(
    cv::Mat& bgr_frame,
    const PoseSkeleton& kp,
    const std::string& gesture_title,
    float confidence,
    const std::string& gpu_summary
) {
    if (bgr_frame.empty()) return;

    // Connect upper-body bones (head, shoulders, elbows, wrists)
    constexpr std::array<std::pair<std::size_t, std::size_t>, 9> kUpperBones{{
        {0, 1}, {0, 2}, {1, 3}, {2, 4},
        {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}
    }};

    for (const auto& [a, b] : kUpperBones) {
        if (kp[a].confidence >= 0.35f && kp[b].confidence >= 0.35f) {
            const cv::Point pa(static_cast<int>(kp[a].x), static_cast<int>(kp[a].y));
            const cv::Point pb(static_cast<int>(kp[b].x), static_cast<int>(kp[b].y));
            cv::line(bgr_frame, pa, pb, cv::Scalar(20, 20, 25), 7, cv::LINE_AA);
            cv::line(bgr_frame, pa, pb, cv::Scalar(0, 235, 175), 3, cv::LINE_AA);
        }
    }

    // Draw only upper-body keypoints (0..10)
    for (std::size_t i = 0; i <= 10; ++i) {
        if (kp[i].confidence >= 0.35f) {
            const cv::Point pt(static_cast<int>(kp[i].x), static_cast<int>(kp[i].y));
            const int radius = (i == 9 || i == 10) ? 9 : 6;
            const cv::Scalar color = (i == 9 || i == 10) ? cv::Scalar(40, 180, 255) : cv::Scalar(255, 215, 60);
            cv::circle(bgr_frame, pt, radius + 2, cv::Scalar(15, 15, 20), -1, cv::LINE_AA);
            cv::circle(bgr_frame, pt, radius, color, -1, cv::LINE_AA);
        }
    }

    // Top HUD banner
    cv::Mat roi = bgr_frame(cv::Rect(0, 0, bgr_frame.cols, std::min(68, bgr_frame.rows)));
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(12, 14, 20));
    cv::addWeighted(dark, 0.78, roi, 0.22, 0.0, roi);

    const std::string hud_line1 = "DETECTED: " + gesture_title + " (" +
                                  std::to_string(static_cast<int>(std::round(confidence * 100.0f))) + "%)";
    cv::putText(bgr_frame, hud_line1, cv::Point(150, 28), cv::FONT_HERSHEY_DUPLEX, 0.64, cv::Scalar(40, 225, 255), 2, cv::LINE_AA);
    cv::putText(bgr_frame, gpu_summary, cv::Point(150, 54), cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(180, 235, 190), 1, cv::LINE_AA);
}

} // namespace meme
