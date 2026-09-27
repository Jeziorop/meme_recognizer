#include "MainWindow.hpp"

#if __has_include("imgui.h")
#include "imgui.h"
#include "backends/imgui_impl_opengl2.h"
#include "backends/imgui_impl_opengl3.h"
#else
#include "../third_party/imgui/imgui.h"
#include "../third_party/imgui/backends/imgui_impl_opengl2.h"
#include "../third_party/imgui/backends/imgui_impl_opengl3.h"
#endif

#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>

namespace {

struct CliOptions {
    bool self_test{false};
    bool has_pose{false};
    int pose_index{0};
    std::string capture_path{};
};

CliOptions parseCli(int argc, char* argv[]) {
    CliOptions opts{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--self-test") {
            opts.self_test = true;
        } else if (arg == "--capture" && i + 1 < argc) {
            opts.capture_path = argv[++i];
        } else if (arg == "--pose" && i + 1 < argc) {
            opts.has_pose = true;
            opts.pose_index = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: meme_recognizer [--self-test] [--capture <file.png>] [--pose <0..7>]\n";
            std::exit(0);
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "MemeGestureRecognizer 0.2.0 (Dear ImGui + OpenGL + CUDA)\n";
            std::exit(0);
        }
    }
    return opts;
}

class GlxImGuiWindow {
public:
    GlxImGuiWindow(int width, int height, const char* title, meme::MainWindow* main_win)
        : width_(width), height_(height) {
        if (std::getenv("DISPLAY") == nullptr) {
            return;
        }
        dpy_ = XOpenDisplay(nullptr);
        if (dpy_ == nullptr) {
            return;
        }

        int attribs[] = {
            GLX_RGBA,
            GLX_DEPTH_SIZE, 24,
            GLX_DOUBLEBUFFER,
            None
        };
        vi_ = glXChooseVisual(dpy_, 0, attribs);
        if (vi_ == nullptr) {
            XCloseDisplay(dpy_);
            dpy_ = nullptr;
            return;
        }

        Window root = DefaultRootWindow(dpy_);
        cmap_ = XCreateColormap(dpy_, root, vi_->visual, AllocNone);

        XSetWindowAttributes swa{};
        swa.colormap = cmap_;
        swa.event_mask =
            ExposureMask | KeyPressMask | KeyReleaseMask |
            ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
            StructureNotifyMask;

        xwin_ = XCreateWindow(
            dpy_,
            root,
            40,
            40,
            static_cast<unsigned int>(width_),
            static_cast<unsigned int>(height_),
            0,
            vi_->depth,
            InputOutput,
            vi_->visual,
            CWColormap | CWEventMask,
            &swa
        );

        wm_delete_ = XInternAtom(dpy_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(dpy_, xwin_, &wm_delete_, 1);
        XStoreName(dpy_, xwin_, title);

        Atom net_wm_name = XInternAtom(dpy_, "_NET_WM_NAME", False);
        Atom utf8_string = XInternAtom(dpy_, "UTF8_STRING", False);
        if (net_wm_name != None && utf8_string != None) {
            XChangeProperty(
                dpy_,
                xwin_,
                net_wm_name,
                utf8_string,
                8,
                PropModeReplace,
                reinterpret_cast<const unsigned char*>(title),
                static_cast<int>(std::strlen(title))
            );
        }

        XMapRaised(dpy_, xwin_);
        XFlush(dpy_);

        const auto map_wait_start = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - map_wait_start
               ).count() < 120) {
            XEvent ev{};
            if (XCheckTypedWindowEvent(dpy_, xwin_, MapNotify, &ev)) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        glc_ = glXCreateContext(dpy_, vi_, nullptr, GL_TRUE);
        if (glc_ == nullptr) {
            XDestroyWindow(dpy_, xwin_);
            XFreeColormap(dpy_, cmap_);
            XFree(vi_);
            XCloseDisplay(dpy_);
            dpy_ = nullptr;
            return;
        }

        glXMakeCurrent(dpy_, xwin_, glc_);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.DisplaySize = ImVec2(static_cast<float>(width_), static_cast<float>(height_));

        if (main_win != nullptr) {
            main_win->initFonts();
        }
        meme::MainWindow::applyImGuiDarkTheme();

        if (ImGui_ImplOpenGL3_Init("#version 130")) {
            use_gl3_ = true;
        } else {
            ImGui_ImplOpenGL2_Init();
            use_gl3_ = false;
        }

        last_frame_tp_ = std::chrono::steady_clock::now();
        active_ = true;
    }

    ~GlxImGuiWindow() {
        if (!active_) {
            return;
        }
        if (use_gl3_) {
            ImGui_ImplOpenGL3_Shutdown();
        } else {
            ImGui_ImplOpenGL2_Shutdown();
        }
        ImGui::DestroyContext();

        glXMakeCurrent(dpy_, None, nullptr);
        glXDestroyContext(dpy_, glc_);
        XDestroyWindow(dpy_, xwin_);
        XFreeColormap(dpy_, cmap_);
        XFree(vi_);
        XCloseDisplay(dpy_);
    }

    bool isActive() const { return active_; }
    bool shouldClose() const { return should_close_; }
    int width() const { return width_; }
    int height() const { return height_; }

    void pollEvents() {
        if (!active_) {
            return;
        }
        ImGuiIO& io = ImGui::GetIO();
        while (XPending(dpy_) > 0) {
            XEvent ev{};
            XNextEvent(dpy_, &ev);
            if (ev.type == ClientMessage) {
                if (static_cast<Atom>(ev.xclient.data.l[0]) == wm_delete_) {
                    should_close_ = true;
                }
            } else if (ev.type == ConfigureNotify) {
                if (ev.xconfigure.width > 0 && ev.xconfigure.height > 0) {
                    width_ = ev.xconfigure.width;
                    height_ = ev.xconfigure.height;
                }
            } else if (ev.type == MotionNotify) {
                io.AddMousePosEvent(
                    static_cast<float>(ev.xmotion.x),
                    static_cast<float>(ev.xmotion.y)
                );
            } else if (ev.type == ButtonPress || ev.type == ButtonRelease) {
                const bool is_down = (ev.type == ButtonPress);
                if (ev.xbutton.button == Button1) {
                    io.AddMouseButtonEvent(0, is_down);
                } else if (ev.xbutton.button == Button3) {
                    io.AddMouseButtonEvent(1, is_down);
                } else if (ev.xbutton.button == Button2) {
                    io.AddMouseButtonEvent(2, is_down);
                } else if (ev.xbutton.button == Button4 && is_down) {
                    io.AddMouseWheelEvent(0.0f, 1.0f);
                } else if (ev.xbutton.button == Button5 && is_down) {
                    io.AddMouseWheelEvent(0.0f, -1.0f);
                }
            } else if (ev.type == KeyPress) {
                KeySym ks = XLookupKeysym(&ev.xkey, 0);
                if (ks == XK_Escape || ks == XK_q || ks == XK_Q) {
                    should_close_ = true;
                }
            }
        }
    }

    void beginFrame() {
        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last_frame_tp_).count();
        if (dt <= 0.0f || dt > 0.5f) {
            dt = 1.0f / 30.0f;
        }
        last_frame_tp_ = now;

        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(static_cast<float>(width_), static_cast<float>(height_));
        io.DeltaTime = dt;

        if (use_gl3_) {
            ImGui_ImplOpenGL3_NewFrame();
        } else {
            ImGui_ImplOpenGL2_NewFrame();
        }
        ImGui::NewFrame();
    }

    void renderDrawData() {
        ImGui::Render();
        glViewport(0, 0, width_, height_);
        glClearColor(0.043f, 0.059f, 0.098f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (use_gl3_) {
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        } else {
            ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        }
        glFinish();
    }

    void swapBuffers() {
        glXSwapBuffers(dpy_, xwin_);
    }

    cv::Mat captureFramebufferBgr() const {
        cv::Mat rgba(height_, width_, CV_8UC4);
        glReadBuffer(GL_BACK);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
        cv::Mat flipped;
        cv::flip(rgba, flipped, 0);
        cv::Mat bgr;
        cv::cvtColor(flipped, bgr, cv::COLOR_RGBA2BGR);
        return bgr;
    }

private:
    int width_{1480};
    int height_{920};
    Display* dpy_{nullptr};
    XVisualInfo* vi_{nullptr};
    Colormap cmap_{0};
    Window xwin_{0};
    GLXContext glc_{nullptr};
    Atom wm_delete_{0};
    bool use_gl3_{true};
    bool active_{false};
    bool should_close_{false};
    std::chrono::steady_clock::time_point last_frame_tp_{};
};

} // namespace

int main(int argc, char* argv[]) {
    const CliOptions opts = parseCli(argc, argv);

    std::filesystem::path root_dir = std::filesystem::current_path();
    if (!std::filesystem::exists(root_dir / "assets" / "memes" / "manifest.json")) {
        const auto exe_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
        if (std::filesystem::exists(exe_dir.parent_path().parent_path() / "assets" / "memes" / "manifest.json")) {
            root_dir = exe_dir.parent_path().parent_path();
        }
    }

    meme::MainWindow window(root_dir);

    if (opts.has_pose || opts.self_test || !opts.capture_path.empty()) {
        window.setSimulatedPoseMode(opts.pose_index);
    }

    GlxImGuiWindow gl_win(
        window.width(),
        window.height(),
        "Meme Gesture Recognizer - C++20 | OpenCV 4 DNN | CUDA GPU | Dear ImGui + OpenGL",
        &window
    );

    if (opts.self_test || !opts.capture_path.empty()) {
        window.processSingleFrame();
        if (gl_win.isActive()) {
            // Render two warmup frames so Dear ImGui layout/table columns settle
            for (int f = 0; f < 2; ++f) {
                gl_win.pollEvents();
                gl_win.beginFrame();
                window.renderImGui(gl_win.width(), gl_win.height());
                gl_win.renderDrawData();
                if (f == 0) {
                    gl_win.swapBuffers();
                }
            }
        }
        if (!opts.capture_path.empty()) {
            const cv::Mat shot = gl_win.isActive()
                ? gl_win.captureFramebufferBgr()
                : window.renderCompositeBgr();
            cv::imwrite(opts.capture_path, shot);
            std::cout << "[MemeRecognizer] Captured window screenshot to " << opts.capture_path << "\n";
        }
        if (gl_win.isActive()) {
            gl_win.swapBuffers();
        }
        if (opts.self_test) {
            std::cout << "[MemeRecognizer] Self-test completed successfully.\n";
            return 0;
        }
    }

    if (!gl_win.isActive()) {
        std::cerr << "[MemeRecognizer] Error: Could not open X11/GLX display window (check DISPLAY environment variable).\n";
        return 1;
    }

    while (!gl_win.shouldClose()) {
        const auto frame_start = std::chrono::steady_clock::now();

        gl_win.pollEvents();
        window.processSingleFrame();

        gl_win.beginFrame();
        window.renderImGui(gl_win.width(), gl_win.height());
        gl_win.renderDrawData();
        gl_win.swapBuffers();

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - frame_start
        ).count();
        if (elapsed_ms < 16) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16 - elapsed_ms));
        }
    }

    return 0;
}
