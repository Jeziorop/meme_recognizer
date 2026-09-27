#pragma once

#include "GpuDeviceManager.hpp"
#include "MemeClassifier.hpp"
#include "PoseDetector.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

namespace meme {

struct InputSourceEntry {
    std::string label;
    int mode; // >= 0: V4L2 device index, -1: Auto-cycle demo, <= -2: Pose preset
};

class MainWindow {
public:
    explicit MainWindow(const std::filesystem::path& project_root);
    ~MainWindow();

    void setSimulatedPoseMode(int class_index);
    void processSingleFrame();

    void initGlResources();
    void releaseGlResources();
    void initFonts();

    void renderImGui(int viewport_width, int viewport_height);
    static void applyImGuiDarkTheme();
    cv::Mat renderCompositeBgr() const;

    int width() const { return 1480; }
    int height() const { return 920; }

private:
    bool configureAndOpenCamera(int device_index);
    void populateCameraSources();
    void onSourceChanged(int combo_index);
    cv::Mat renderSyntheticWebcamFrame(const PoseSkeleton& kp, int class_index) const;
    void uploadMatToGlTexture(const cv::Mat& rgba, unsigned int& tex_id);

    std::filesystem::path root_dir_;
    GpuStatus gpu_status_;
    PoseDetector pose_detector_;
    MemeClassifier classifier_;
    cv::VideoCapture cap_;

    std::vector<InputSourceEntry> source_entries_;
    int selected_source_idx_{0};
    int active_source_mode_{-1};
    int demo_tick_{0};

    bool mirror_camera_{true};
    bool show_skeleton_hud_{true};

    cv::Mat current_webcam_bgr_;
    cv::Mat current_webcam_rgba_;
    PredictionResult latest_result_{};
    float latest_inference_ms_{0.0f};
    int active_class_idx_{7};
    std::string hint_banner_text_{"Show a gesture (e.g. raise both hands for 'Absolute Cinema')"};

    std::vector<cv::Mat> meme_images_bgr_;
    std::vector<cv::Mat> meme_images_rgba_;

    bool gl_initialized_{false};
    unsigned int webcam_texture_{0};
    std::vector<unsigned int> meme_textures_;

    void* font_regular_{nullptr};
    void* font_bold_{nullptr};
    void* font_heading_{nullptr};

    std::chrono::steady_clock::time_point last_frame_time_{};
    float current_fps_{0.0f};
    bool has_fps_sample_{false};
};

} // namespace meme
