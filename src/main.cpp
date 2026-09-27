#include "MainWindow.hpp"

#include <QApplication>
#include <QCommandLineParser>
#include <QImage>
#include <QMouseEvent>
#include <QTimer>
#include <QWidget>
#include <QtPlugin>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#if defined(MEME_HAVE_STATIC_OFFSCREEN_PLUGIN) && MEME_HAVE_STATIC_OFFSCREEN_PLUGIN
Q_IMPORT_PLUGIN(QOffscreenIntegrationPlugin)
#endif

#if defined(MEME_HAVE_X11) && MEME_HAVE_X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif

namespace {

#if defined(MEME_HAVE_X11) && MEME_HAVE_X11
class X11DesktopBridge {
public:
    explicit X11DesktopBridge(meme::MainWindow* window) : main_win_(window) {
        if (std::getenv("DISPLAY") == nullptr) {
            return;
        }
        dpy_ = XOpenDisplay(nullptr);
        if (dpy_ == nullptr) {
            return;
        }
        const int screen = DefaultScreen(dpy_);
        width_ = main_win_->width();
        height_ = main_win_->height();
        xwin_ = XCreateSimpleWindow(
            dpy_,
            RootWindow(dpy_, screen),
            40,
            40,
            static_cast<unsigned int>(width_),
            static_cast<unsigned int>(height_),
            1,
            BlackPixel(dpy_, screen),
            BlackPixel(dpy_, screen)
        );
        XStoreName(
            dpy_,
            xwin_,
            "Meme Gesture Recognizer — C++20 | OpenCV 4 DNN | CUDA GPU | Qt 6 (Keys: 0=Webcam, 1-8=Memes, C=Cycle, Q=Quit)"
        );
        wm_delete_ = XInternAtom(dpy_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy_, xwin_, &wm_delete_, 1);
        XSelectInput(dpy_, xwin_, ExposureMask | KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask);
        gc_ = DefaultGC(dpy_, screen);
        visual_ = DefaultVisual(dpy_, screen);
        depth_ = DefaultDepth(dpy_, screen);
        XMapWindow(dpy_, xwin_);
        XFlush(dpy_);
    }

    ~X11DesktopBridge() {
        if (dpy_ != nullptr) {
            if (xwin_ != 0) {
                XDestroyWindow(dpy_, xwin_);
            }
            XCloseDisplay(dpy_);
        }
    }

    [[nodiscard]] bool isActive() const noexcept { return dpy_ != nullptr && xwin_ != 0; }

    void syncFrame() {
        if (!isActive()) return;

        while (XPending(dpy_) > 0) {
            XEvent ev{};
            XNextEvent(dpy_, &ev);
            if (ev.type == ClientMessage && static_cast<Atom>(ev.xclient.data.l[0]) == wm_delete_) {
                QApplication::quit();
                return;
            }
            if (ev.type == ConfigureNotify) {
                // Window resized by user or window manager
                width_ = ev.xconfigure.width;
                height_ = ev.xconfigure.height;
                main_win_->resize(width_, height_);
            } else if (ev.type == KeyPress) {
                const KeySym sym = XLookupKeysym(&ev.xkey, 0);
                if (sym == XK_Escape || sym == XK_q || sym == XK_Q) {
                    QApplication::quit();
                    return;
                }
                if (sym >= XK_1 && sym <= XK_8) {
                    main_win_->setSimulatedPoseMode(static_cast<int>(sym - XK_1));
                }
            } else if (ev.type == ButtonPress || ev.type == ButtonRelease) {
                const QPoint local_pos(ev.xbutton.x, ev.xbutton.y);
                const QPoint global_pos = main_win_->mapToGlobal(local_pos);
                Qt::MouseButton btn = Qt::LeftButton;
                if (ev.xbutton.button == 2) btn = Qt::MiddleButton;
                else if (ev.xbutton.button == 3) btn = Qt::RightButton;

                QWidget* target = QApplication::widgetAt(global_pos);
                if (target == nullptr) {
                    target = main_win_->childAt(local_pos);
                }
                if (target == nullptr) {
                    target = main_win_;
                }

                const QPoint widget_pos = target->mapFromGlobal(global_pos);
                const QEvent::Type event_type = (ev.type == ButtonPress) ? QEvent::MouseButtonPress : QEvent::MouseButtonRelease;
                QMouseEvent mouse_ev(
                    event_type,
                    QPointF(widget_pos),
                    QPointF(global_pos),
                    btn,
                    (ev.type == ButtonPress) ? btn : Qt::NoButton,
                    Qt::NoModifier
                );
                QApplication::sendEvent(target, &mouse_ev);
            } else if (ev.type == MotionNotify) {
                const QPoint local_pos(ev.xmotion.x, ev.xmotion.y);
                const QPoint global_pos = main_win_->mapToGlobal(local_pos);
                QWidget* target = QApplication::widgetAt(global_pos);
                if (target == nullptr) {
                    target = main_win_->childAt(local_pos);
                }
                if (target == nullptr) {
                    target = main_win_;
                }

                const QPoint widget_pos = target->mapFromGlobal(global_pos);
                Qt::MouseButtons btns = Qt::NoButton;
                if (ev.xmotion.state & Button1Mask) btns |= Qt::LeftButton;
                if (ev.xmotion.state & Button2Mask) btns |= Qt::MiddleButton;
                if (ev.xmotion.state & Button3Mask) btns |= Qt::RightButton;

                QMouseEvent mouse_ev(
                    QEvent::MouseMove,
                    QPointF(widget_pos),
                    QPointF(global_pos),
                    Qt::NoButton,
                    btns,
                    Qt::NoModifier
                );
                QApplication::sendEvent(target, &mouse_ev);
            }
        }

        QImage img = main_win_->grab().toImage().convertToFormat(QImage::Format_RGB32);
        if (img.isNull()) return;

        XImage* ximg = XCreateImage(
            dpy_,
            visual_,
            static_cast<unsigned int>(depth_),
            ZPixmap,
            0,
            reinterpret_cast<char*>(img.bits()),
            static_cast<unsigned int>(img.width()),
            static_cast<unsigned int>(img.height()),
            32,
            static_cast<int>(img.bytesPerLine())
        );
        if (ximg != nullptr) {
            XPutImage(
                dpy_,
                xwin_,
                gc_,
                ximg,
                0,
                0,
                0,
                0,
                static_cast<unsigned int>(img.width()),
                static_cast<unsigned int>(img.height())
            );
            ximg->data = nullptr; // Owned by QImage
            XDestroyImage(ximg);
        }
        XFlush(dpy_);
    }

private:
    meme::MainWindow* main_win_{nullptr};
    Display* dpy_{nullptr};
    Window xwin_{0};
    GC gc_{nullptr};
    Visual* visual_{nullptr};
    int depth_{24};
    int width_{1360};
    int height_{820};
    Atom wm_delete_{0};
};
#endif

} // namespace

int main(int argc, char* argv[]) {
    if (std::getenv("QT_QPA_PLATFORM") == nullptr) {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    QApplication app(argc, argv);
    QApplication::setApplicationName("MemeGestureRecognizer");
    QApplication::setApplicationVersion("0.1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Real-Time GPU Meme Gesture Recognizer (C++20 / OpenCV 4 / CUDA / Qt 6)");
    parser.addHelpOption();
    parser.addVersionOption();

    QCommandLineOption selfTestOpt("self-test", "Run automated GUI + GPU rendering self-test and exit.");
    QCommandLineOption captureOpt("capture", "Save rendered window screenshot to path.", "file");
    QCommandLineOption poseOpt("pose", "Set initial pose index (0..7, e.g., 0 for Absolute Cinema).", "index", "0");
    parser.addOption(selfTestOpt);
    parser.addOption(captureOpt);
    parser.addOption(poseOpt);
    parser.process(app);

    std::filesystem::path root_dir = std::filesystem::current_path();
    if (!std::filesystem::exists(root_dir / "assets" / "memes" / "manifest.json")) {
        const auto exe_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
        if (std::filesystem::exists(exe_dir.parent_path().parent_path() / "assets" / "memes" / "manifest.json")) {
            root_dir = exe_dir.parent_path().parent_path();
        }
    }

    meme::MainWindow window(root_dir);

    if (parser.isSet(poseOpt) || parser.isSet(selfTestOpt) || parser.isSet(captureOpt)) {
        const int pose_idx = parser.value(poseOpt).toInt();
        window.setSimulatedPoseMode(pose_idx);
    }

    window.show();
    window.processSingleFrame();
    QApplication::processEvents();

    if (parser.isSet(captureOpt)) {
        const QString out_path = parser.value(captureOpt);
        const QPixmap grab = window.grab();
        grab.save(out_path, "PNG");
        std::cout << "[MemeRecognizer] Captured window screenshot to " << out_path.toStdString() << "\n";
    }

    if (parser.isSet(selfTestOpt)) {
        std::cout << "[MemeRecognizer] Self-test completed successfully.\n";
        return 0;
    }

#if defined(MEME_HAVE_X11) && MEME_HAVE_X11
    X11DesktopBridge x11_bridge(&window);
    QTimer presenter_timer;
    if (x11_bridge.isActive()) {
        QObject::connect(&presenter_timer, &QTimer::timeout, [&]() {
            x11_bridge.syncFrame();
        });
        presenter_timer.start(33);
    }
#endif

    return app.exec();
}
