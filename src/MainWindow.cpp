#include "MainWindow.hpp"
#include "PoseFeatureExtractor.hpp"

#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <tuple>

namespace meme {

MainWindow::MainWindow(const std::filesystem::path& project_root, QWidget* parent)
    : QMainWindow(parent), root_dir_(project_root) {
    gpu_status_ = GpuDeviceManager::initialize();

    pose_detector_.loadModels(
        root_dir_ / "models" / "yolov8n-pose.onnx"
    );
    classifier_.loadManifest(root_dir_ / "assets" / "memes" / "manifest.json");
    classifier_.loadModels(
        root_dir_ / "models" / "meme_gesture_classifier.onnx",
        root_dir_ / "models" / "meme_gesture_classifier.bin"
    );

    for (const auto& info : classifier_.classes()) {
        QPixmap pix(QString::fromStdString(info.image_path));
        meme_pixmaps_.push_back(pix);
    }

    setupUi();
    applyDarkTheme();

    if (cap_.open(0, cv::CAP_V4L2) || cap_.open(0)) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        active_source_mode_ = 0;
        source_combo_->setCurrentIndex(0);
    } else {
        active_source_mode_ = -1;
        source_combo_->setCurrentIndex(2);
    }

    frame_timer_ = new QTimer(this);
    connect(frame_timer_, &QTimer::timeout, this, &MainWindow::onTimerTick);
    frame_timer_->start(33);
}

MainWindow::~MainWindow() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

void MainWindow::setSimulatedPoseMode(int class_index) {
    if (cap_.isOpened()) {
        cap_.release();
    }
    classifier_.resetSmoothing();
    active_source_mode_ = -2 - std::clamp(class_index, 0, static_cast<int>(kNumMemeClasses) - 1);
    source_combo_->blockSignals(true);
    source_combo_->setCurrentIndex(3 + std::clamp(class_index, 0, static_cast<int>(kNumMemeClasses) - 1));
    source_combo_->blockSignals(false);
}

void MainWindow::setupUi() {
    setWindowTitle("Meme Gesture Recognizer — C++20 | OpenCV 4 DNN | CUDA GPU | Qt 6");
    resize(1480, 920);
    setMinimumSize(1280, 800);

    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* root_layout = new QVBoxLayout(central);
    root_layout->setContentsMargins(20, 16, 20, 18);
    root_layout->setSpacing(16);

    auto* top_bar = new QFrame(this);
    top_bar->setObjectName("CardFrame");
    auto* top_layout = new QHBoxLayout(top_bar);
    top_layout->setContentsMargins(18, 12, 18, 12);
    top_layout->setSpacing(16);

    auto* app_title = new QLabel("REAL-TIME MEME GESTURE RECOGNIZER", this);
    app_title->setStyleSheet("font-size: 20px; font-weight: 800; color: #F8FAFC; letter-spacing: 1px;");
    top_layout->addWidget(app_title);

    gpu_badge_label_ = new QLabel(QString::fromStdString(gpu_status_.active_summary), this);
    gpu_badge_label_->setStyleSheet(
        "background-color: #064E3B; color: #6EE7B7; font-weight: 700; font-size: 13px; "
        "padding: 6px 14px; border-radius: 8px; border: 1px solid #10B981;"
    );
    top_layout->addWidget(gpu_badge_label_);
    top_layout->addStretch();

    auto* src_label = new QLabel("Input Source:", this);
    src_label->setStyleSheet("color: #E2E8F0; font-size: 14px; font-weight: 700;");
    top_layout->addWidget(src_label);

    source_combo_ = new QComboBox(this);
    source_combo_->addItem("Webcam /dev/video0 (Live)", 0);
    source_combo_->addItem("Webcam /dev/video2 (Live)", 2);
    source_combo_->addItem("Auto-Cycle All 8 Meme Poses (Demo)", -1);
    for (const auto& cls : classifier_.classes()) {
        source_combo_->addItem(
            QString::fromStdString("Pose Preset: " + cls.title),
            -2 - cls.index
        );
    }
    source_combo_->setMinimumHeight(38);
    connect(source_combo_, &QComboBox::currentIndexChanged, this, &MainWindow::onSourceChanged);
    top_layout->addWidget(source_combo_);

    mirror_check_ = new QCheckBox("Mirror Camera", this);
    mirror_check_->setChecked(true);
    mirror_check_->setCursor(Qt::PointingHandCursor);
    mirror_check_->setMinimumHeight(38);
    top_layout->addWidget(mirror_check_);

    skeleton_check_ = new QCheckBox("Show Skeleton HUD", this);
    skeleton_check_->setChecked(true);
    skeleton_check_->setCursor(Qt::PointingHandCursor);
    skeleton_check_->setMinimumHeight(38);
    top_layout->addWidget(skeleton_check_);

    root_layout->addWidget(top_bar);

    auto* content_layout = new QHBoxLayout();
    content_layout->setSpacing(18);

    auto* left_card = new QFrame(this);
    left_card->setObjectName("CardFrame");
    auto* left_layout = new QVBoxLayout(left_card);
    left_layout->setContentsMargins(18, 18, 18, 18);
    left_layout->setSpacing(12);

    auto* cam_header = new QLabel("LIVE WEBCAM & UPPER-BODY KEYPOINTS", this);
    cam_header->setStyleSheet("font-size: 15px; font-weight: 800; color: #94A3B8; letter-spacing: 0.8px;");
    left_layout->addWidget(cam_header);

    webcam_view_label_ = new QLabel(this);
    webcam_view_label_->setMinimumSize(720, 520);
    webcam_view_label_->setAlignment(Qt::AlignCenter);
    webcam_view_label_->setStyleSheet("background-color: #090D16; border: 2px solid #1E293B; border-radius: 12px;");
    left_layout->addWidget(webcam_view_label_, 1);

    hint_banner_label_ = new QLabel("Show a gesture (e.g. raise both hands for 'Absolute Cinema')", this);
    hint_banner_label_->setWordWrap(true);
    hint_banner_label_->setStyleSheet(
        "background-color: #1E293B; color: #FDE68A; font-size: 14px; font-weight: 600; "
        "padding: 12px 16px; border-radius: 10px; border-left: 4px solid #F59E0B;"
    );
    left_layout->addWidget(hint_banner_label_);
    content_layout->addWidget(left_card, 5);

    auto* right_card = new QFrame(this);
    right_card->setObjectName("CardFrame");
    auto* right_layout = new QVBoxLayout(right_card);
    right_layout->setContentsMargins(18, 18, 18, 18);
    right_layout->setSpacing(12);

    auto* meme_header = new QLabel("MATCHED MEME & NEURAL ACTIVATIONS", this);
    meme_header->setStyleSheet("font-size: 15px; font-weight: 800; color: #94A3B8; letter-spacing: 0.8px;");
    right_layout->addWidget(meme_header);

    meme_image_label_ = new QLabel(this);
    meme_image_label_->setMinimumSize(440, 420);
    meme_image_label_->setAlignment(Qt::AlignCenter);
    meme_image_label_->setStyleSheet("background-color: #090D16; border: 2px solid #334155; border-radius: 12px;");
    right_layout->addWidget(meme_image_label_, 1);

    meme_title_label_ = new QLabel("ABSOLUTE CINEMA", this);
    meme_title_label_->setAlignment(Qt::AlignCenter);
    meme_title_label_->setStyleSheet("font-size: 26px; font-weight: 900; color: #F59E0B; margin-top: 4px;");
    right_layout->addWidget(meme_title_label_);

    meme_subtitle_label_ = new QLabel("Martin Scorsese — Both hands raised open at head height", this);
    meme_subtitle_label_->setAlignment(Qt::AlignCenter);
    meme_subtitle_label_->setStyleSheet("font-size: 14px; color: #94A3B8; margin-bottom: 4px;");
    right_layout->addWidget(meme_subtitle_label_);

    auto* grid = new QGridLayout();
    grid->setVerticalSpacing(8);
    grid->setHorizontalSpacing(14);
    const auto& cls_list = classifier_.classes();
    for (std::size_t i = 0; i < kNumMemeClasses; ++i) {
        const std::string name = (i < cls_list.size()) ? cls_list[i].title : ("Class " + std::to_string(i));
        auto* lbl = new QLabel(QString::fromStdString(name), this);
        lbl->setStyleSheet("color: #E2E8F0; font-size: 13px; font-weight: 700;");

        auto* bar = new QProgressBar(this);
        bar->setRange(0, 1000);
        bar->setValue(0);
        bar->setFormat("%p%");
        bar->setFixedHeight(20);
        prob_bars_[i] = bar;

        grid->addWidget(lbl, static_cast<int>(i), 0);
        grid->addWidget(bar, static_cast<int>(i), 1);
    }
    right_layout->addLayout(grid);

    content_layout->addWidget(right_card, 4);
    root_layout->addLayout(content_layout, 1);
}

void MainWindow::applyDarkTheme() {
    setStyleSheet(R"(
        QMainWindow {
            background-color: #0B0F19;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        QFrame#CardFrame {
            background-color: #111827;
            border: 1px solid #1F2937;
            border-radius: 14px;
        }
        QComboBox {
            background-color: #1E293B;
            color: #F8FAFC;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 14px;
            font-weight: 600;
        }
        QComboBox:hover {
            border: 1px solid #475569;
            background-color: #27354A;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 28px;
            border-left: 1px solid #334155;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #1E293B;
            color: #F8FAFC;
            selection-background-color: #3B82F6;
            selection-color: #FFFFFF;
            border: 1px solid #475569;
            font-size: 13px;
            padding: 4px;
        }
        QCheckBox {
            color: #E2E8F0;
            font-size: 14px;
            font-weight: 600;
            spacing: 10px;
            padding: 4px 8px;
        }
        QCheckBox::indicator {
            width: 22px;
            height: 22px;
            border-radius: 6px;
            border: 2px solid #475569;
            background-color: #1E293B;
        }
        QCheckBox::indicator:hover {
            border: 2px solid #60A5FA;
            background-color: #27354A;
        }
        QCheckBox::indicator:checked {
            border: 2px solid #2563EB;
            background-color: #3B82F6;
        }
        QProgressBar {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 6px;
            text-align: right;
            color: #F8FAFC;
            font-size: 12px;
            font-weight: 700;
            padding-right: 6px;
        }
        QProgressBar::chunk {
            background-color: #F59E0B;
            border-radius: 5px;
        }
    )");
}

void MainWindow::onSourceChanged(int combo_index) {
    const int mode = source_combo_->itemData(combo_index).toInt();
    active_source_mode_ = mode;
    classifier_.resetSmoothing();

    if (cap_.isOpened()) {
        cap_.release();
    }
    if (mode >= 0) {
        if (!cap_.open(mode, cv::CAP_V4L2) && !cap_.open(mode)) {
            active_source_mode_ = -1;
        } else {
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
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

void MainWindow::onTimerTick() {
    processSingleFrame();
}

void MainWindow::processSingleFrame() {
    cv::Mat frame;
    PoseSkeleton kp{};

    if (active_source_mode_ >= 0 && cap_.isOpened() && cap_.read(frame) && !frame.empty()) {
        if (mirror_check_->isChecked()) {
            cv::flip(frame, frame, 1);
        }
        cv::resize(frame, frame, cv::Size(640, 480));
        kp = pose_detector_.detect(frame);
    } else {
        int sim_class = 0;
        if (active_source_mode_ == -1) {
            sim_class = (demo_tick_ / 75) % static_cast<int>(kNumMemeClasses);
        } else if (active_source_mode_ <= -2) {
            sim_class = std::clamp(-active_source_mode_ - 2, 0, static_cast<int>(kNumMemeClasses) - 1);
        }
        kp = PoseFeatureExtractor::getCanonicalPose(sim_class, false);
        const float wave = std::sin(static_cast<float>(demo_tick_) * 0.12f) * 4.5f;
        kp[9].y += wave;
        kp[10].y -= wave * 0.8f;
        frame = renderSyntheticWebcamFrame(kp, sim_class);
        ++demo_tick_;
    }

    const FeatureVector feat = PoseFeatureExtractor::extract(kp);
    const PredictionResult pred = classifier_.classify(feat, 0.45f);

    const auto& classes = classifier_.classes();
    const int cls_idx = std::clamp(pred.class_index, 0, static_cast<int>(classes.size()) - 1);
    const std::string title = classes.empty() ? "ABSOLUTE CINEMA" : classes[cls_idx].title;
    const std::string subtitle = classes.empty() ? "" : classes[cls_idx].subtitle;
    const std::string hint = classes.empty() ? "" : classes[cls_idx].hint;

    if (skeleton_check_->isChecked()) {
        PoseDetector::drawSkeletonOverlay(frame, kp, title, pred.confidence, pred.backend_used);
    }

    webcam_view_label_->setPixmap(
        QPixmap::fromImage(cvMatToQImage(frame))
            .scaled(webcam_view_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation)
    );

    if (cls_idx >= 0 && static_cast<std::size_t>(cls_idx) < meme_pixmaps_.size()) {
        meme_image_label_->setPixmap(
            meme_pixmaps_[cls_idx].scaled(
                meme_image_label_->size(),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation
            )
        );
    }

    meme_title_label_->setText(
        QString::fromStdString(
            title + " (" + std::to_string(static_cast<int>(std::round(pred.confidence * 100.0f))) + "%)"
        )
    );
    meme_subtitle_label_->setText(QString::fromStdString(subtitle));
    hint_banner_label_->setText(QString::fromStdString("Gesture Guide: " + hint));

    for (std::size_t i = 0; i < kNumMemeClasses; ++i) {
        if (prob_bars_[i] != nullptr) {
            prob_bars_[i]->setValue(static_cast<int>(std::round(pred.probabilities[i] * 1000.0f)));
        }
    }
}

QImage MainWindow::cvMatToQImage(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return QImage(
               rgb.data,
               rgb.cols,
               rgb.rows,
               static_cast<int>(rgb.step),
               QImage::Format_RGB888
    )
        .copy();
}

} // namespace meme
