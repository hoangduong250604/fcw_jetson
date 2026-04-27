#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fcw {
namespace gui {

class GuiApp {
public:
    GuiApp();
    ~GuiApp() = default;

    void setVideoDir(const std::string& dir) { videoDir_ = dir; }
    void setKittiRoot(const std::string& dir) { kittiRoot_ = dir; }
    void setConfigDir(const std::string& dir) { configDir_ = dir; }
    int run();

private:
    // Screens
    void showSplash();
    int  showMainMenu();             // 0 = camera, 1 = video, -1 = quit
    std::string showVideoSelect();   // returns video path or "" (back)
    void launchPipeline(const std::string& inputType, const std::string& source);

    // Drawing helpers
    void textCentered(cv::Mat& img, const std::string& text, int y,
                      double scale, cv::Scalar color, int thickness = 1,
                      int font = cv::FONT_HERSHEY_SIMPLEX);
    void textOnRect(cv::Mat& img, const std::string& text, const cv::Rect& rect,
                    int yOffset, double scale, cv::Scalar color, int thickness = 1,
                    int font = cv::FONT_HERSHEY_SIMPLEX);
    void drawCameraIcon(cv::Mat& img, cv::Point center, int size, cv::Scalar color);
    void drawPlayIcon(cv::Mat& img, cv::Point center, int size, cv::Scalar color);

    // Utilities
    std::vector<std::string> scanVideoFiles();

    // Mouse callback
    static void onMouse(int event, int x, int y, int flags, void* userdata);

    // Window
    std::string windowName_ = "FCW System";
    int width_  = 1280;
    int height_ = 720;

    // Mouse state
    int  mx_ = -1, my_ = -1;
    bool clicked_ = false;
    int  scrollDelta_ = 0;

    // Config
    std::string videoDir_ = "./video_data";
    std::string kittiRoot_ = "../KITTI";
    std::string configDir_ = "./config";
    std::string logoPath_  = "./logo/logo1.png";

    // Cached background
    cv::Mat bgImage_;

    // Theme colors (BGR)
    cv::Scalar bgDark_      {20, 20, 30};
    cv::Scalar bgCard_      {40, 35, 30};
    cv::Scalar accent_      {220, 160, 50};    // Light blue
    cv::Scalar accentGreen_ {100, 200, 80};    // Green
    cv::Scalar textWhite_   {255, 255, 255};
    cv::Scalar textGray_    {180, 180, 180};
    cv::Scalar hoverColor_  {60, 55, 45};
    cv::Scalar dangerRed_   {60, 60, 230};     // Red
};

} // namespace gui
} // namespace fcw
