#include "MainWindow.hpp"
#include "PoseFeatureExtractor.hpp"

#if __has_include("imgui.h")
#include "imgui.h"
#else
#include "../third_party/imgui/imgui.h"
#endif

#include <GL/gl.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

namespace meme {

namespace {

ImVec4 hexToImVec4(const std::string& hex, float alpha = 1.0f) {
    if (hex.size() == 7 && hex[0] == '#') {
        unsigned int r = 245, g = 158, b = 11;
        if (std::sscanf(hex.c_str() + 1, "%02x%02x%02x", &r, &g, &b) == 3) {
            return ImVec4(
                static_cast<float>(r) / 255.0f,
                static_cast<float>(g) / 255.0f,
                static_cast<float>(b) / 255.0f,
                alpha
            );
        }
    }
    return ImVec4(0.96f, 0.62f, 0.04f, alpha);
}

} // namespace

MainWindow::MainWindow(const std::filesystem::path& project_root)
    : root_dir_(project_root) {
    gpu_status_ = GpuDeviceManager::initialize();

    pose_detector_.loadModels(root_dir_ / "models" / "yolov8n-pose.onnx");
    classifier_.loadManifest(root_dir_ / "assets" / "memes" / "manifest.json");
    classifier_.loadModels(
        root_dir_ / "models" / "meme_gesture_classifier.onnx",
        root_dir_ / "models" / "meme_gesture_classifier.bin"
    );

    for (const auto& info : classifier_.classes()) {
        cv::Mat bgr = cv::imread(info.image_path, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            bgr = cv::Mat(360, 480, CV_8UC3, cv::Scalar(30, 41, 59));
        }
        cv::Mat rgba;
        cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
        meme_images_bgr_.push_back(std::move(bgr));
        meme_images_rgba_.push_back(std::move(rgba));
    }

    populateCameraSources();
}

MainWindow::~MainWindow() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    releaseGlResources();
}

void MainWindow::initGlResources() {
    if (gl_initialized_) {
        return;
    }
    gl_initialized_ = true;

    meme_textures_.resize(meme_images_rgba_.size(), 0);
    for (size_t i = 0; i < meme_images_rgba_.size(); ++i) {
        uploadMatToGlTexture(meme_images_rgba_[i], meme_textures_[i]);
    }

    if (!current_webcam_rgba_.empty()) {
        uploadMatToGlTexture(current_webcam_rgba_, webcam_texture_);
    }
}

void MainWindow::releaseGlResources() {
    if (!gl_initialized_) {
        return;
    }
    if (webcam_texture_ != 0) {
        glDeleteTextures(1, &webcam_texture_);
        webcam_texture_ = 0;
    }
    for (unsigned int& tex : meme_textures_) {
        if (tex != 0) {
            glDeleteTextures(1, &tex);
            tex = 0;
        }
    }
    gl_initialized_ = false;
}

void MainWindow::uploadMatToGlTexture(const cv::Mat& rgba, unsigned int& tex_id) {
    if (rgba.empty()) {
        return;
    }
    if (tex_id == 0) {
        glGenTextures(1, &tex_id);
        glBindTexture(GL_TEXTURE_2D, tex_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            rgba.cols,
            rgba.rows,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            rgba.data
        );
    } else {
        glBindTexture(GL_TEXTURE_2D, tex_id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            rgba.cols,
            rgba.rows,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            rgba.data
        );
    }
}

bool MainWindow::configureAndOpenCamera(int device_index) {
    if (cap_.isOpened()) {
        cap_.release();
    }

    if (!cap_.open(device_index, cv::CAP_V4L2) && !cap_.open(device_index)) {
        return false;
    }

    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap_.set(cv::CAP_PROP_FPS, 30);
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);

    cv::Mat dummy;
    for (int i = 0; i < 5; ++i) {
        cap_.read(dummy);
    }

    return true;
}

void MainWindow::populateCameraSources() {
    source_entries_.clear();

    struct CameraDev {
        int index;
        std::string name;
    };
    std::vector<CameraDev> detected_cams;

    for (int i = 0; i < 16; ++i) {
        const std::string dev_path = "/dev/video" + std::to_string(i);
        int fd = ::open(dev_path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) {
            continue;
        }

        v4l2_capability cap{};
        if (::ioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            const uint32_t caps = cap.device_caps ? cap.device_caps : cap.capabilities;
            if (caps & V4L2_CAP_VIDEO_CAPTURE) {
                std::string card_name = reinterpret_cast<const char*>(cap.card);
                detected_cams.push_back({i, card_name});
            }
        }
        ::close(fd);
    }

    for (size_t idx = 0; idx < detected_cams.size(); ++idx) {
        const auto& cam = detected_cams[idx];
        char buf[256];
        std::snprintf(
            buf,
            sizeof(buf),
            "Camera %zu: %s (/dev/video%d)",
            idx + 1,
            cam.name.c_str(),
            cam.index
        );
        source_entries_.push_back({buf, cam.index});
    }

    source_entries_.push_back({"Auto-Cycle All 8 Meme Poses (Demo)", -1});
    for (const auto& cls : classifier_.classes()) {
        source_entries_.push_back({"Pose Preset: " + cls.title, -2 - cls.index});
    }

    int opened_combo_idx = -1;
    for (size_t idx = 0; idx < detected_cams.size(); ++idx) {
        if (configureAndOpenCamera(detected_cams[idx].index)) {
            active_source_mode_ = detected_cams[idx].index;
            opened_combo_idx = static_cast<int>(idx);
            break;
        }
    }

    if (opened_combo_idx >= 0) {
        selected_source_idx_ = opened_combo_idx;
    } else {
        active_source_mode_ = -1;
        for (size_t i = 0; i < source_entries_.size(); ++i) {
            if (source_entries_[i].mode == -1) {
                selected_source_idx_ = static_cast<int>(i);
                break;
            }
        }
    }
}

void MainWindow::setSimulatedPoseMode(int class_index) {
    if (cap_.isOpened()) {
        cap_.release();
    }
    classifier_.resetSmoothing();
    const int target_mode = -2 - std::clamp(class_index, 0, static_cast<int>(kNumMemeClasses) - 1);
    active_source_mode_ = target_mode;
    for (size_t i = 0; i < source_entries_.size(); ++i) {
        if (source_entries_[i].mode == target_mode) {
            selected_source_idx_ = static_cast<int>(i);
            break;
        }
    }
}

void MainWindow::onSourceChanged(int combo_index) {
    if (combo_index < 0 || combo_index >= static_cast<int>(source_entries_.size())) {
        return;
    }
    selected_source_idx_ = combo_index;
    const int mode = source_entries_[combo_index].mode;
    active_source_mode_ = mode;
    classifier_.resetSmoothing();

    if (mode >= 0) {
        if (!configureAndOpenCamera(mode)) {
            active_source_mode_ = -1;
        }
    } else {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }
}

cv::Mat MainWindow::renderSyntheticWebcamFrame(const PoseSkeleton& kp, int class_index) const {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(22, 26, 36));

    for (int x = 0; x < 640; x += 64) {
        cv::line(frame, cv::Point(x, 0), cv::Point(x, 480), cv::Scalar(30, 36, 48), 1);
    }
    for (int y = 0; y < 480; y += 64) {
        cv::line(frame, cv::Point(0, y), cv::Point(640, y), cv::Scalar(30, 36, 48), 1);
    }

    const cv::Point l_sh(static_cast<int>(kp[5].x), static_cast<int>(kp[5].y));
    const cv::Point r_sh(static_cast<int>(kp[6].x), static_cast<int>(kp[6].y));
    const cv::Point l_hip(static_cast<int>(kp[11].x), static_cast<int>(kp[11].y));
    const cv::Point r_hip(static_cast<int>(kp[12].x), static_cast<int>(kp[12].y));
    std::vector<cv::Point> torso{l_sh, r_sh, r_hip, l_hip};
    cv::fillConvexPoly(frame, torso, cv::Scalar(55, 68, 92), cv::LINE_AA);

    const cv::Point nose(static_cast<int>(kp[0].x), static_cast<int>(kp[0].y));
    cv::circle(frame, nose, 36, cv::Scalar(160, 195, 230), -1, cv::LINE_AA);

    constexpr std::array<std::tuple<int, int, int>, 2> kArms{{{5, 7, 9}, {6, 8, 10}}};
    for (const auto& [sh, el, wr] : kArms) {
        const cv::Point p_sh(static_cast<int>(kp[sh].x), static_cast<int>(kp[sh].y));
        const cv::Point p_el(static_cast<int>(kp[el].x), static_cast<int>(kp[el].y));
        const cv::Point p_wr(static_cast<int>(kp[wr].x), static_cast<int>(kp[wr].y));
        cv::line(frame, p_sh, p_el, cv::Scalar(82, 100, 132), 16, cv::LINE_AA);
        cv::line(frame, p_el, p_wr, cv::Scalar(110, 135, 175), 14, cv::LINE_AA);
        cv::circle(frame, p_wr, 14, cv::Scalar(170, 205, 240), -1, cv::LINE_AA);
    }

    cv::putText(
        frame,
        "SIMULATED WEBCAM POSE #" + std::to_string(class_index + 1),
        cv::Point(18, 462),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(148, 163, 184),
        1,
        cv::LINE_AA
    );
    return frame;
}

void MainWindow::processSingleFrame() {
    const auto now = std::chrono::steady_clock::now();
    if (has_fps_sample_) {
        const float dt = std::chrono::duration<float>(now - last_frame_time_).count();
        if (dt > 0.0001f) {
            const float inst_fps = 1.0f / dt;
            current_fps_ = (current_fps_ <= 0.1f) ? inst_fps : (0.85f * current_fps_ + 0.15f * inst_fps);
        }
    } else {
        has_fps_sample_ = true;
        current_fps_ = 30.0f;
    }
    last_frame_time_ = now;

    cv::Mat frame;
    PoseSkeleton kp{};

    const auto infer_start = std::chrono::steady_clock::now();
    if (active_source_mode_ >= 0 && cap_.isOpened() && cap_.read(frame) && !frame.empty()) {
        if (mirror_camera_) {
            cv::flip(frame, frame, 1);
        }
        cv::resize(frame, frame, cv::Size(640, 480));
        kp = pose_detector_.detect(frame);
    } else {
        int target_class = 0;
        if (active_source_mode_ == -1) {
            target_class = (demo_tick_ / 45) % static_cast<int>(kNumMemeClasses);
            ++demo_tick_;
        } else if (active_source_mode_ <= -2) {
            target_class = std::clamp(-2 - active_source_mode_, 0, static_cast<int>(kNumMemeClasses) - 1);
        }
        kp = PoseFeatureExtractor::getCanonicalPose(target_class);
        frame = renderSyntheticWebcamFrame(kp, target_class);
    }

    const FeatureVector features = PoseFeatureExtractor::extract(kp);
    latest_result_ = classifier_.classify(features);
    latest_inference_ms_ = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - infer_start
    ).count();

    const auto& classes = classifier_.classes();
    active_class_idx_ = std::clamp(latest_result_.class_index, 0, static_cast<int>(classes.size()) - 1);
    const auto& active_info = classes[active_class_idx_];

    if (show_skeleton_hud_) {
        PoseDetector::drawSkeletonOverlay(
            frame,
            kp,
            active_info.title,
            latest_result_.confidence,
            latest_result_.backend_used
        );
    }

    char fps_buf[32];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", current_fps_);
    cv::rectangle(frame, cv::Point(12, 12), cv::Point(140, 48), cv::Scalar(15, 23, 42), -1, cv::LINE_AA);
    cv::rectangle(frame, cv::Point(12, 12), cv::Point(140, 48), cv::Scalar(16, 185, 129), 2, cv::LINE_AA);
    cv::putText(
        frame,
        fps_buf,
        cv::Point(22, 36),
        cv::FONT_HERSHEY_DUPLEX,
        0.62,
        cv::Scalar(110, 231, 183),
        1,
        cv::LINE_AA
    );

    hint_banner_text_ = "How to trigger '" + active_info.title + "': " + active_info.hint;
    current_webcam_bgr_ = frame;
    cv::cvtColor(current_webcam_bgr_, current_webcam_rgba_, cv::COLOR_BGR2RGBA);

    if (gl_initialized_) {
        uploadMatToGlTexture(current_webcam_rgba_, webcam_texture_);
    }
}

void MainWindow::initFonts() {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const std::vector<std::filesystem::path> reg_candidates = {
        root_dir_ / "assets" / "fonts" / "LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    };
    const std::vector<std::filesystem::path> bold_candidates = {
        root_dir_ / "assets" / "fonts" / "LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    };

    std::string reg_path;
    for (const auto& p : reg_candidates) {
        if (std::filesystem::exists(p)) {
            reg_path = p.string();
            break;
        }
    }
    std::string bold_path;
    for (const auto& p : bold_candidates) {
        if (std::filesystem::exists(p)) {
            bold_path = p.string();
            break;
        }
    }

    ImFontConfig cfg_reg{};
    cfg_reg.OversampleH = 3;
    cfg_reg.OversampleV = 2;
    cfg_reg.PixelSnapH = true;

    ImFontConfig cfg_bold{};
    cfg_bold.OversampleH = 3;
    cfg_bold.OversampleV = 2;
    cfg_bold.PixelSnapH = true;

    ImFontConfig cfg_head{};
    cfg_head.OversampleH = 3;
    cfg_head.OversampleV = 2;
    cfg_head.PixelSnapH = true;

    if (!reg_path.empty()) {
        font_regular_ = io.Fonts->AddFontFromFileTTF(reg_path.c_str(), 17.0f, &cfg_reg);
    }
    if (!bold_path.empty()) {
        font_bold_ = io.Fonts->AddFontFromFileTTF(bold_path.c_str(), 18.0f, &cfg_bold);
        font_heading_ = io.Fonts->AddFontFromFileTTF(bold_path.c_str(), 23.0f, &cfg_head);
    }

    if (font_regular_ == nullptr) {
        font_regular_ = io.Fonts->AddFontDefault();
    }
    if (font_bold_ == nullptr) {
        font_bold_ = font_regular_;
    }
    if (font_heading_ == nullptr) {
        font_heading_ = font_bold_;
    }

    io.FontDefault = static_cast<ImFont*>(font_regular_);
}

void MainWindow::applyImGuiDarkTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = ImVec2(18.0f, 16.0f);
    style.FramePadding = ImVec2(10.0f, 7.0f);
    style.ItemSpacing = ImVec2(12.0f, 8.0f);
    style.CellPadding = ImVec2(8.0f, 3.5f);
    style.WindowRounding = 0.0f;
    style.ChildRounding = 12.0f;
    style.FrameRounding = 8.0f;
    style.PopupRounding = 8.0f;
    style.GrabRounding = 6.0f;
    style.ChildBorderSize = 1.5f;
    style.FrameBorderSize = 1.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.043f, 0.059f, 0.098f, 1.00f);       // #0B0F19
    colors[ImGuiCol_ChildBg] = ImVec4(0.067f, 0.094f, 0.153f, 1.00f);        // #111827
    colors[ImGuiCol_PopupBg] = ImVec4(0.118f, 0.161f, 0.231f, 0.98f);        // #1E293B
    colors[ImGuiCol_Border] = ImVec4(0.149f, 0.200f, 0.275f, 1.00f);         // #263346
    colors[ImGuiCol_FrameBg] = ImVec4(0.118f, 0.161f, 0.231f, 1.00f);        // #1E293B
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.165f, 0.224f, 0.314f, 1.00f); // #2A3950
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.231f, 0.510f, 0.965f, 0.45f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.063f, 0.725f, 0.506f, 1.00f);      // #10B981
    colors[ImGuiCol_Button] = ImVec4(0.118f, 0.161f, 0.231f, 1.00f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.231f, 0.510f, 0.965f, 1.00f);
    colors[ImGuiCol_Header] = ImVec4(0.231f, 0.510f, 0.965f, 0.45f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.231f, 0.510f, 0.965f, 0.80f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.231f, 0.510f, 0.965f, 1.00f);
    colors[ImGuiCol_Text] = ImVec4(0.973f, 0.980f, 0.988f, 1.00f);           // #F8FAFC
    colors[ImGuiCol_TextDisabled] = ImVec4(0.580f, 0.639f, 0.722f, 1.00f);   // #94A3B8
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.961f, 0.620f, 0.043f, 1.00f);  // #F59E0B
}

void MainWindow::renderImGui(int viewport_width, int viewport_height) {
    if (!gl_initialized_) {
        initGlResources();
    }

    const float scale = std::clamp(
        std::min(static_cast<float>(viewport_width) / 1480.0f, static_cast<float>(viewport_height) / 920.0f),
        0.85f,
        2.25f
    );
    ImGui::GetIO().FontGlobalScale = scale;

    auto* f_bold = static_cast<ImFont*>(font_bold_);
    auto* f_head = static_cast<ImFont*>(font_heading_);

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(static_cast<float>(viewport_width), static_cast<float>(viewport_height)));

    constexpr ImGuiWindowFlags kRootFlags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::Begin("MemeRecognizerMain", nullptr, kRootFlags);

    // ==================== TOP NAVIGATION BAR ====================
    ImGui::BeginChild("TopBarCard", ImVec2(0.0f, 66.0f * scale), true, ImGuiWindowFlags_NoScrollbar);
    {
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 3.0f * scale);

        if (f_head != nullptr) ImGui::PushFont(f_head);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(0.97f, 0.98f, 1.00f, 1.0f), "MEME GESTURE RECOGNIZER");
        if (f_head != nullptr) ImGui::PopFont();

        ImGui::SameLine(0.0f, 14.0f * scale);

        // GPU Status Pill Badge
        const std::string gpu_badge_text = gpu_status_.cuda_available
            ? ("CUDA GPU: " + gpu_status_.cuda_device_name)
            : gpu_status_.active_summary;
        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.024f, 0.306f, 0.231f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.024f, 0.306f, 0.231f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.024f, 0.306f, 0.231f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.431f, 0.906f, 0.718f, 1.0f));
        ImGui::Button(gpu_badge_text.c_str(), ImVec2(0.0f, 34.0f * scale));
        ImGui::PopStyleColor(4);
        if (f_bold != nullptr) ImGui::PopFont();

        ImGui::SameLine(0.0f, 18.0f * scale);

        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(ImVec4(0.886f, 0.910f, 0.941f, 1.0f), "Input Source:");
        if (f_bold != nullptr) ImGui::PopFont();

        ImGui::SameLine(0.0f, 8.0f * scale);
        ImGui::SetNextItemWidth(310.0f * scale);
        const char* current_label = source_entries_.empty()
            ? "None"
            : source_entries_[selected_source_idx_].label.c_str();
        if (ImGui::BeginCombo("##InputSourceCombo", current_label)) {
            for (int i = 0; i < static_cast<int>(source_entries_.size()); ++i) {
                const bool is_selected = (selected_source_idx_ == i);
                if (ImGui::Selectable(source_entries_[i].label.c_str(), is_selected)) {
                    if (i != selected_source_idx_) {
                        onSourceChanged(i);
                    }
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::SameLine(0.0f, 16.0f * scale);
        ImGui::Checkbox("Mirror Camera", &mirror_camera_);
        ImGui::SameLine(0.0f, 16.0f * scale);
        ImGui::Checkbox("Show Skeleton HUD", &show_skeleton_hud_);
    }
    ImGui::EndChild();

    // ==================== MAIN TWO-COLUMN CONTENT ====================
    const float content_height = ImGui::GetContentRegionAvail().y;
    const float total_width = ImGui::GetContentRegionAvail().x;
    const float left_width = total_width * 0.53f;

    // -------------------- LEFT CARD: LIVE WEBCAM --------------------
    ImGui::BeginChild("LeftWebcamCard", ImVec2(left_width, content_height), true, ImGuiWindowFlags_NoScrollbar);
    {
        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::TextColored(ImVec4(0.580f, 0.639f, 0.722f, 1.0f), "LIVE WEBCAM & UPPER-BODY KEYPOINTS");
        if (f_bold != nullptr) ImGui::PopFont();
        ImGui::Spacing();

        const float banner_h = 60.0f * scale;
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = std::max(120.0f, ImGui::GetContentRegionAvail().y - banner_h - 12.0f * scale);
        float disp_w = avail_w;
        float disp_h = disp_w * (480.0f / 640.0f);
        if (disp_h > avail_h) {
            disp_h = avail_h;
            disp_w = disp_h * (640.0f / 480.0f);
        }

        const float offset_x = (avail_w - disp_w) * 0.5f;
        if (offset_x > 0.0f) {
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
        }

        if (webcam_texture_ != 0) {
            ImGui::Image(
                static_cast<ImTextureID>(static_cast<uintptr_t>(webcam_texture_)),
                ImVec2(disp_w, disp_h)
            );
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.118f, 0.161f, 0.231f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.961f, 0.620f, 0.043f, 0.65f));
        ImGui::BeginChild("HintBanner", ImVec2(0.0f, banner_h), true, ImGuiWindowFlags_NoScrollbar);
        ImGui::PushTextWrapPos(0.0f);
        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::TextColored(ImVec4(0.992f, 0.902f, 0.541f, 1.0f), "%s", hint_banner_text_.c_str());
        if (f_bold != nullptr) ImGui::PopFont();
        ImGui::PopTextWrapPos();
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // -------------------- RIGHT CARD: DETECTED MEME & PROBABILITIES --------------------
    ImGui::BeginChild("RightMemeCard", ImVec2(0.0f, content_height), true, ImGuiWindowFlags_NoScrollbar);
    {
        const auto& classes = classifier_.classes();
        const auto& active_info = classes[active_class_idx_];

        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::TextColored(ImVec4(0.580f, 0.639f, 0.722f, 1.0f), "DETECTED MEME REACTION");
        if (f_bold != nullptr) ImGui::PopFont();

        const ImVec4 accent_color = hexToImVec4(active_info.accent_hex);
        if (f_head != nullptr) ImGui::PushFont(f_head);
        ImGui::TextColored(accent_color, "%s", active_info.title.c_str());
        if (f_head != nullptr) ImGui::PopFont();

        char subtitle_buf[256];
        std::snprintf(
            subtitle_buf,
            sizeof(subtitle_buf),
            "%s  |  %s (%.1f ms)  |  %.1f%% Confidence",
            active_info.subtitle.c_str(),
            latest_result_.backend_used.c_str(),
            latest_inference_ms_,
            latest_result_.confidence * 100.0f
        );
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(ImVec4(0.68f, 0.74f, 0.82f, 1.0f), "%s", subtitle_buf);
        ImGui::PopTextWrapPos();
        ImGui::Spacing();

        // Reserve space so all 9 rows of CLASS PROBABILITY ACTIVATIONS fit without scrolling
        const float prob_section_h = 370.0f * scale;
        const float avail_w = ImGui::GetContentRegionAvail().x;
        const float avail_h = std::max(140.0f * scale, ImGui::GetContentRegionAvail().y - prob_section_h);
        if (active_class_idx_ < static_cast<int>(meme_textures_.size()) &&
            meme_textures_[active_class_idx_] != 0) {
            const cv::Mat& m_rgba = meme_images_rgba_[active_class_idx_];
            const float aspect = static_cast<float>(m_rgba.cols) / std::max(1.0f, static_cast<float>(m_rgba.rows));
            float mw = avail_w;
            float mh = mw / aspect;
            if (mh > avail_h) {
                mh = avail_h;
                mw = mh * aspect;
            }
            const float off_x = (avail_w - mw) * 0.5f;
            if (off_x > 0.0f) {
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off_x);
            }
            ImGui::Image(
                static_cast<ImTextureID>(static_cast<uintptr_t>(meme_textures_[active_class_idx_])),
                ImVec2(mw, mh)
            );
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (f_bold != nullptr) ImGui::PushFont(f_bold);
        ImGui::TextColored(ImVec4(0.580f, 0.639f, 0.722f, 1.0f), "CLASS PROBABILITY ACTIVATIONS");
        if (f_bold != nullptr) ImGui::PopFont();

        if (ImGui::BeginTable("ProbTable", 3, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("ClassName", ImGuiTableColumnFlags_WidthFixed, 235.0f * scale);
            ImGui::TableSetupColumn("ConfidenceBar", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("PctValue", ImGuiTableColumnFlags_WidthFixed, 72.0f * scale);

            for (size_t i = 0; i < kNumMemeClasses && i < classes.size(); ++i) {
                ImGui::TableNextRow();
                const bool is_top = (static_cast<int>(i) == active_class_idx_);
                const ImVec4 cls_accent = hexToImVec4(classes[i].accent_hex);

                // Column 0: Gesture Class Name
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                if (is_top && f_bold != nullptr) ImGui::PushFont(f_bold);
                ImGui::TextColored(
                    is_top ? cls_accent : ImVec4(0.85f, 0.88f, 0.93f, 1.0f),
                    "%s",
                    classes[i].title.c_str()
                );
                if (is_top && f_bold != nullptr) ImGui::PopFont();

                // Column 1: Smooth Progress Bar (without overlapping text)
                ImGui::TableSetColumnIndex(1);
                const float prob = std::clamp(latest_result_.probabilities[i], 0.0f, 1.0f);
                const ImVec4 bar_color = is_top
                    ? cls_accent
                    : ImVec4(0.231f, 0.510f, 0.965f, 0.85f);
                ImGui::PushStyleColor(ImGuiCol_PlotHistogram, bar_color);
                ImGui::ProgressBar(prob, ImVec2(-1.0f, 19.0f * scale), "");
                ImGui::PopStyleColor();

                // Column 2: Dedicated Percentage Readout
                ImGui::TableSetColumnIndex(2);
                ImGui::AlignTextToFramePadding();
                if (f_bold != nullptr) ImGui::PushFont(f_bold);
                ImGui::TextColored(
                    is_top ? cls_accent : ImVec4(0.68f, 0.74f, 0.82f, 1.0f),
                    "%5.1f%%",
                    prob * 100.0f
                );
                if (f_bold != nullptr) ImGui::PopFont();
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();

    ImGui::End();
}

cv::Mat MainWindow::renderCompositeBgr() const {
    cv::Mat canvas(height(), width(), CV_8UC3, cv::Scalar(25, 15, 11));
    if (!current_webcam_bgr_.empty()) {
        cv::Mat resized_cam;
        cv::resize(current_webcam_bgr_, resized_cam, cv::Size(720, 540));
        resized_cam.copyTo(canvas(cv::Rect(30, 110, 720, 540)));
    }
    if (active_class_idx_ >= 0 && active_class_idx_ < static_cast<int>(meme_images_bgr_.size())) {
        cv::Mat resized_meme;
        cv::resize(meme_images_bgr_[active_class_idx_], resized_meme, cv::Size(560, 420));
        resized_meme.copyTo(canvas(cv::Rect(800, 150, 560, 420)));
    }
    const auto& classes = classifier_.classes();
    if (active_class_idx_ >= 0 && active_class_idx_ < static_cast<int>(classes.size())) {
        cv::putText(
            canvas,
            classes[active_class_idx_].title,
            cv::Point(800, 120),
            cv::FONT_HERSHEY_DUPLEX,
            1.1,
            cv::Scalar(11, 158, 245),
            2,
            cv::LINE_AA
        );
    }
    return canvas;
}

} // namespace meme
