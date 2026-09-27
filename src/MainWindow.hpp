#pragma once

#include "GpuDeviceManager.hpp"
#include "MemeClassifier.hpp"
#include "PoseDetector.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QProgressBar>
#include <QTimer>
#include <array>
#include <filesystem>
#include <opencv2/videoio.hpp>
#include <vector>

namespace meme {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(const std::filesystem::path& project_root, QWidget* parent = nullptr);
    ~MainWindow() override;

    void setSimulatedPoseMode(int class_index);
    void processSingleFrame();

private slots:
    void onTimerTick();
    void onSourceChanged(int combo_index);

private:
    void setupUi();
    void applyDarkTheme();
    cv::Mat renderSyntheticWebcamFrame(const PoseSkeleton& kp, int class_index) const;
    static QImage cvMatToQImage(const cv::Mat& bgr);

    std::filesystem::path root_dir_;
    GpuStatus gpu_status_;
    PoseDetector pose_detector_;
    MemeClassifier classifier_;
    cv::VideoCapture cap_;

    QTimer* frame_timer_{nullptr};
    QComboBox* source_combo_{nullptr};
    QCheckBox* mirror_check_{nullptr};
    QCheckBox* skeleton_check_{nullptr};
    QLabel* gpu_badge_label_{nullptr};
    QLabel* webcam_view_label_{nullptr};
    QLabel* hint_banner_label_{nullptr};
    QLabel* meme_image_label_{nullptr};
    QLabel* meme_title_label_{nullptr};
    QLabel* meme_subtitle_label_{nullptr};
    std::array<QProgressBar*, kNumMemeClasses> prob_bars_{};
    std::vector<QPixmap> meme_pixmaps_;

    int active_source_mode_{-1};
    int demo_tick_{0};
};

} // namespace meme
