#include "gui_app.h"
#include "pipeline.h"
#include "logger.h"

#include <algorithm>
#include <chrono>
#include <fstream>

// Filesystem: Jetson uses experimental, Desktop may use std
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#else
  #include <experimental/filesystem>
  namespace fs = std::experimental::filesystem;
#endif

namespace fcw {
namespace gui {

// ============================================================================
// Constructor & Mouse Callback
// ============================================================================
GuiApp::GuiApp() {}

void GuiApp::onMouse(int event, int x, int y, int flags, void* userdata) {
    auto* app = static_cast<GuiApp*>(userdata);
    app->mx_ = x;
    app->my_ = y;
    if (event == cv::EVENT_LBUTTONDOWN) {
        app->clicked_ = true;
    }
    if (event == cv::EVENT_MOUSEWHEEL) {
        int delta = cv::getMouseWheelDelta(flags);
        app->scrollDelta_ += (delta > 0) ? -1 : 1;
    }
}

// ============================================================================
// Drawing Helpers
// ============================================================================
void GuiApp::textCentered(cv::Mat& img, const std::string& text, int y,
                           double scale, cv::Scalar color, int thickness, int font) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, font, scale, thickness, &baseline);
    cv::putText(img, text, cv::Point((img.cols - ts.width) / 2, y),
                font, scale, color, thickness, cv::LINE_AA);
}

void GuiApp::textOnRect(cv::Mat& img, const std::string& text, const cv::Rect& rect,
                         int yOffset, double scale, cv::Scalar color, int thickness, int font) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(text, font, scale, thickness, &baseline);
    cv::putText(img, text,
                cv::Point(rect.x + (rect.width - ts.width) / 2, rect.y + yOffset),
                font, scale, color, thickness, cv::LINE_AA);
}

void GuiApp::drawCameraIcon(cv::Mat& img, cv::Point c, int sz, cv::Scalar color) {
    int hw = sz / 2, hh = sz / 3;
    // Camera body
    cv::rectangle(img, cv::Rect(c.x - hw, c.y - hh, sz, sz * 2 / 3), color, 3, cv::LINE_AA);
    // Lens
    cv::circle(img, c, sz / 4, color, 3, cv::LINE_AA);
    // Flash bump
    cv::rectangle(img, cv::Rect(c.x - hw / 2, c.y - hh - sz / 6, hw, sz / 6), color, -1, cv::LINE_AA);
}

void GuiApp::drawPlayIcon(cv::Mat& img, cv::Point c, int sz, cv::Scalar color) {
    std::vector<cv::Point> tri = {
        cv::Point(c.x - sz / 3, c.y - sz / 2),
        cv::Point(c.x - sz / 3, c.y + sz / 2),
        cv::Point(c.x + sz / 2, c.y)
    };
    cv::fillConvexPoly(img, tri, color, cv::LINE_AA);
}

std::vector<std::string> GuiApp::scanVideoFiles() {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(videoDir_)) {
            if (!fs::is_regular_file(entry)) continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".avi" || ext == ".mp4" || ext == ".mkv" || ext == ".mov") {
                files.push_back(entry.path().string());
            }
        }
    } catch (...) { /* directory not found */ }
    std::sort(files.begin(), files.end());
    return files;
}

// ============================================================================
// SPLASH SCREEN  (3 seconds, fade-in logo + loading bar)
// ============================================================================
void GuiApp::showSplash() {
    cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);

    auto t0 = std::chrono::steady_clock::now();
    const int durationMs = 3000;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        int elapsed = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count();
        if (elapsed >= durationMs) break;

        float progress = (float)elapsed / durationMs;
        float alpha = std::min(1.0f, elapsed / 800.0f);   // fade in first 0.8s

        cv::Mat canvas(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

        // Background image with fade-in
        if (!bgImage_.empty()) {
            cv::addWeighted(bgImage_, alpha, canvas, 1.0 - alpha, 0, canvas);
        }

        auto fade = [&](cv::Scalar c) -> cv::Scalar {
            return cv::Scalar(c[0] * alpha, c[1] * alpha, c[2] * alpha);
        };

        // Title text at the top
        textCentered(canvas, "FCW SYSTEM", 60, 1.8,
                     fade(accent_), 4, cv::FONT_HERSHEY_TRIPLEX);

        // Subtitle at the bottom
        textCentered(canvas, "FORWARD COLLISION WARNING SYSTEM",
                     height_ - 160, 0.7, fade(textWhite_), 2);

        // Tagline above loading bar
        textCentered(canvas, "Vision-Based FCW  |  YOLOv8 + TTC  |  Jetson Nano",
                     height_ - 115, 0.5, fade(textGray_), 1);

        // Loading bar at bottom
        int barW = 400, barH = 6;
        int barX = (width_ - barW) / 2, barY = height_ - 80;
        cv::rectangle(canvas, cv::Rect(barX, barY, barW, barH),
                      cv::Scalar(40, 40, 40), -1);
        cv::rectangle(canvas, cv::Rect(barX, barY, (int)(barW * progress), barH),
                      accent_, -1);

        textCentered(canvas, "Initializing...", barY + 30, 0.4, fade(textGray_), 1);

        cv::imshow(windowName_, canvas);
        if (cv::waitKey(16) == 27) return;   // ESC to skip
    }
}

// ============================================================================
// MAIN MENU  (Camera  |  Video Test)
// ============================================================================
int GuiApp::showMainMenu() {
    cv::setMouseCallback(windowName_, onMouse, this);

    const int btnW = 350, btnH = 380;
    const int gap = 80;
    int startX = (width_ - (btnW * 2 + gap)) / 2;
    int startY = (height_ - btnH) / 2 + 30;

    cv::Rect camRect(startX, startY, btnW, btnH);
    cv::Rect vidRect(startX + btnW + gap, startY, btnW, btnH);

    while (true) {
        clicked_ = false;

        // Background: logo image with dark overlay
        cv::Mat canvas;
        if (!bgImage_.empty()) {
            canvas = bgImage_.clone();
            cv::Mat darkOverlay(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::addWeighted(canvas, 0.3, darkOverlay, 0.7, 0, canvas);
        } else {
            canvas = cv::Mat(height_, width_, CV_8UC3, bgDark_);
        }

        // ---- Title ----
        textCentered(canvas, "FCW SYSTEM", 70, 1.5, accent_, 3, cv::FONT_HERSHEY_TRIPLEX);
        textCentered(canvas, "Forward Collision Warning", 110, 0.7, textGray_, 1);
        cv::line(canvas, cv::Point(width_ / 2 - 300, 130),
                 cv::Point(width_ / 2 + 300, 130), cv::Scalar(50, 50, 50), 1, cv::LINE_AA);

        bool camHov = camRect.contains(cv::Point(mx_, my_));
        bool vidHov = vidRect.contains(cv::Point(mx_, my_));

        // ---- Camera Card ----
        {
            cv::Scalar bg = camHov ? cv::Scalar(50, 45, 35) : bgCard_;
            cv::Scalar border = camHov ? accent_ : cv::Scalar(60, 60, 60);
            cv::rectangle(canvas, camRect, bg, -1, cv::LINE_AA);
            cv::rectangle(canvas, camRect, border, 2, cv::LINE_AA);

            drawCameraIcon(canvas, cv::Point(camRect.x + btnW / 2, startY + 120), 60, accent_);
            textOnRect(canvas, "LIVE CAMERA", camRect, 220, 0.9, textWhite_, 2);
            textOnRect(canvas, "Real-time monitoring", camRect, 260, 0.5, textGray_);
            textOnRect(canvas, "with FCW processing", camRect, 290, 0.5, textGray_);
        }

        // ---- Video Card ----
        {
            cv::Scalar bg = vidHov ? cv::Scalar(35, 50, 35) : bgCard_;
            cv::Scalar border = vidHov ? accentGreen_ : cv::Scalar(60, 60, 60);
            cv::rectangle(canvas, vidRect, bg, -1, cv::LINE_AA);
            cv::rectangle(canvas, vidRect, border, 2, cv::LINE_AA);

            drawPlayIcon(canvas, cv::Point(vidRect.x + btnW / 2, startY + 120), 50, accentGreen_);
            textOnRect(canvas, "VIDEO TEST", vidRect, 220, 0.9, textWhite_, 2);
            textOnRect(canvas, "Test with recorded", vidRect, 260, 0.5, textGray_);
            textOnRect(canvas, "KITTI dataset videos", vidRect, 290, 0.5, textGray_);
        }

        // ---- Footer ----
        textCentered(canvas, "Select an option to begin  |  ESC to quit",
                     height_ - 30, 0.45, textGray_, 1);

        cv::imshow(windowName_, canvas);
        int key = cv::waitKey(30);
        if (key == 27) return -1;

        if (clicked_) {
            if (camHov) return 0;
            if (vidHov) return 1;
        }
    }
}

// ============================================================================
// VIDEO SELECTION  (scrollable file list)
// ============================================================================
std::string GuiApp::showVideoSelect() {
    cv::setMouseCallback(windowName_, onMouse, this);

    auto videos = scanVideoFiles();
    int scrollOff = 0;
    const int itemH = 55;
    const int headerH = 140;
    const int footerH = 80;
    int visibleItems = (height_ - headerH - footerH) / itemH;
    int maxScroll = std::max(0, (int)videos.size() - visibleItems);

    cv::Rect backBtn(40, height_ - 65, 150, 45);

    while (true) {
        // Handle mouse-wheel scroll
        if (scrollDelta_ != 0) {
            scrollOff = std::max(0, std::min(maxScroll, scrollOff + scrollDelta_));
            scrollDelta_ = 0;
        }

        clicked_ = false;

        // Background: logo image with dark overlay
        cv::Mat canvas;
        if (!bgImage_.empty()) {
            canvas = bgImage_.clone();
            cv::Mat darkOverlay(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
            cv::addWeighted(canvas, 0.2, darkOverlay, 0.8, 0, canvas);
        } else {
            canvas = cv::Mat(height_, width_, CV_8UC3, bgDark_);
        }

        // ---- Header ----
        textCentered(canvas, "SELECT TEST VIDEO", 60, 1.2, accent_, 2, cv::FONT_HERSHEY_DUPLEX);
        std::string info = std::to_string(videos.size()) + " videos found";
        textCentered(canvas, info, 95, 0.5, textGray_, 1);
        cv::line(canvas, cv::Point(40, 115), cv::Point(width_ - 40, 115),
                 cv::Scalar(50, 50, 50), 1);

        // ---- Video list ----
        int hoverIdx = -1;
        for (int i = scrollOff;
             i < std::min((int)videos.size(), scrollOff + visibleItems); ++i) {

            int y = headerH + (i - scrollOff) * itemH;
            cv::Rect row(60, y, width_ - 120, itemH - 5);

            bool hov = row.contains(cv::Point(mx_, my_));
            if (hov) hoverIdx = i;

            cv::Scalar bg = hov ? hoverColor_ : cv::Scalar(35, 32, 28);
            cv::rectangle(canvas, row, bg, -1, cv::LINE_AA);
            if (hov) cv::rectangle(canvas, row, accent_, 1, cv::LINE_AA);

            // Index
            std::string idx = std::to_string(i + 1) + ".";
            cv::putText(canvas, idx, cv::Point(row.x + 15, y + 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, textGray_, 1, cv::LINE_AA);

            // Filename
            std::string name = fs::path(videos[i]).filename().string();
            cv::putText(canvas, name, cv::Point(row.x + 60, y + 35),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, textWhite_, 1, cv::LINE_AA);

            // Play icon on hover
            if (hov)
                drawPlayIcon(canvas, cv::Point(row.x + row.width - 30, y + 25), 15, accent_);
        }

        // ---- Scroll bar ----
        if ((int)videos.size() > visibleItems) {
            int barArea = height_ - headerH - footerH;
            int thumbH = std::max(30, barArea * visibleItems / (int)videos.size());
            int thumbY = headerH + (barArea - thumbH) * scrollOff / std::max(1, maxScroll);
            cv::rectangle(canvas, cv::Rect(width_ - 20, headerH, 8, barArea),
                          cv::Scalar(30, 30, 30), -1);
            cv::rectangle(canvas, cv::Rect(width_ - 20, thumbY, 8, thumbH),
                          cv::Scalar(80, 80, 80), -1);
        }

        // ---- Empty state ----
        if (videos.empty()) {
            textCentered(canvas, "No video files found in:", height_ / 2 - 20, 0.7, dangerRed_, 2);
            textCentered(canvas, videoDir_, height_ / 2 + 20, 0.6, textGray_, 1);
            textCentered(canvas, "Place .avi / .mp4 files in that folder and restart",
                         height_ / 2 + 55, 0.5, textGray_, 1);
        }

        // ---- Back button ----
        {
            bool hov = backBtn.contains(cv::Point(mx_, my_));
            cv::Scalar bg = hov ? cv::Scalar(60, 40, 40) : cv::Scalar(50, 35, 35);
            cv::rectangle(canvas, backBtn, bg, -1, cv::LINE_AA);
            cv::rectangle(canvas, backBtn, hov ? dangerRed_ : cv::Scalar(80, 60, 60), 2, cv::LINE_AA);
            textOnRect(canvas, "< BACK", backBtn, 30, 0.6, textWhite_, 2);
        }

        // ---- Footer ----
        textCentered(canvas, "Click a video to start  |  Scroll / Arrow keys  |  ESC = back",
                     height_ - 20, 0.4, textGray_, 1);

        cv::imshow(windowName_, canvas);
        int key = cv::waitKey(30) & 0xFFFF;

        if (key == 27) return "";                               // ESC = back
        if (key == 0x2600 || key == 65362 || key == 'w')        // Up arrow / w
            scrollOff = std::max(0, scrollOff - 1);
        if (key == 0x2800 || key == 65364 || key == 's')        // Down arrow / s
            scrollOff = std::min(maxScroll, scrollOff + 1);
        if (key == 0x2100)                                      // Page Up
            scrollOff = std::max(0, scrollOff - visibleItems);
        if (key == 0x2200)                                      // Page Down
            scrollOff = std::min(maxScroll, scrollOff + visibleItems);

        if (clicked_) {
            if (backBtn.contains(cv::Point(mx_, my_))) return "";
            if (hoverIdx >= 0) return videos[hoverIdx];
        }
    }
}

// ============================================================================
// LAUNCH PIPELINE
// ============================================================================
void GuiApp::launchPipeline(const std::string& inputType, const std::string& source) {
    cv::destroyWindow(windowName_);

    // Find best model
    std::string modelPath;
    const char* candidates[] = {
        "./models/yolov8s.engine", "./models/yolov8n.engine",
        "./models/yolov8s.onnx",  "./models/yolov8n.onnx"
    };
    for (auto& p : candidates) {
        std::ifstream f(p);
        if (f.good()) { modelPath = p; break; }
    }
    if (modelPath.empty()) modelPath = "./models/yolov8s.onnx";

    fcw::Pipeline pipeline;

    // Try loading YAML configs (gives full functionality: danger zone, thresholds, etc.)
    std::string sysConfig  = configDir_ + "/system_config.yaml";
    std::string camConfig  = configDir_ + "/camera_config.yaml";
    std::string warnConfig = configDir_ + "/warning_config.yaml";

    if (fs::exists(sysConfig)) {
        pipeline.loadConfig(sysConfig, camConfig, warnConfig);
    }

    // Always override input, model, and KITTI root from GUI selections
    pipeline.overrideInput(inputType, source);
    pipeline.overrideModel(modelPath, "./models/labels.txt");
    pipeline.overrideKittiRoot(kittiRoot_);

    if (inputType == "camera") {
#ifdef PLATFORM_JETSON
        pipeline.overrideCameraType("csi");
#else
        pipeline.overrideCameraType("usb");
#endif
    }

    if (!pipeline.initFromLoadedConfig()) {
        LOG_ERROR("GUI", "Pipeline init failed for: " + source);
        cv::destroyAllWindows();
        return;
    }

    pipeline.run();
    cv::destroyAllWindows();
}

// ============================================================================
// MAIN RUN LOOP
// ============================================================================
int GuiApp::run() {
    // Load logo/background image once
    {
        cv::Mat tmp = cv::imread(logoPath_, cv::IMREAD_COLOR);
        if (!tmp.empty()) {
            cv::resize(tmp, bgImage_, cv::Size(width_, height_), 0, 0, cv::INTER_AREA);
        }
    }

    showSplash();

    while (true) {
        // Re-create window for menu
        cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(windowName_, onMouse, this);

        int choice = showMainMenu();

        if (choice == -1) break;   // Quit

        if (choice == 0) {
            // Camera mode
            launchPipeline("camera", "0");
        }
        else if (choice == 1) {
            // Video selection loop
            while (true) {
                // Re-create window for video list
                cv::namedWindow(windowName_, cv::WINDOW_AUTOSIZE);
                cv::setMouseCallback(windowName_, onMouse, this);

                std::string videoPath = showVideoSelect();
                if (videoPath.empty()) break;   // Back to main menu

                launchPipeline("video", videoPath);
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}

} // namespace gui
} // namespace fcw
