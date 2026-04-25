#pragma once
// ==============================================================================
// Camera Module - Capture from CSI/USB camera or video file
// ==============================================================================

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace fcw {

struct CameraConfig {
    // Intrinsic parameters
    float fx = 721.5377f;
    float fy = 721.5377f;
    float cx = 609.5593f;
    float cy = 172.854f;

    // Image size
    int imageWidth = 1242;
    int imageHeight = 375;

    // Camera mounting
    float mountHeight = 1.65f;    // meters from ground
    float pitchAngle = 0.0f;     // degrees

    // Capture settings
    int captureWidth = 1280;
    int captureHeight = 720;
    int fps = 30;
};

class Camera {
public:
    Camera();
    ~Camera();

    /**
     * Initialize camera from a video file path.
     */
    bool openVideo(const std::string& videoPath);

    /**
     * Initialize CSI camera on Jetson Nano using GStreamer pipeline.
     */
    bool openCSI(int sensorId = 0, int captureWidth = 1280,
                 int captureHeight = 720, int fps = 30, int flipMethod = 0);

    /**
     * Initialize USB camera.
     */
    bool openUSB(int deviceId = 0, int width = 1280, int height = 720);

    /**
     * Read next frame.
     * @return true if frame was successfully captured.
     */
    bool read(cv::Mat& frame);

    /** Get camera intrinsic matrix (3x3) */
    cv::Mat getIntrinsicMatrix() const;

    /** Get distortion coefficients */
    cv::Mat getDistortionCoeffs() const;

    /** Check if camera is opened */
    bool isOpened() const;

    /** Release camera resources */
    void release();

    /** Get frame width */
    int getWidth() const;

    /** Get frame height */
    int getHeight() const;

    /** Get FPS */
    double getFPS() const;

    /** Get current position in video (milliseconds). For live camera, uses system clock. */
    float getPositionMs() const;

    /** Get total frame count (for video files) */
    int getFrameCount() const;

    /** Set camera config */
    void setConfig(const CameraConfig& config);

    /** Get camera config */
    const CameraConfig& getConfig() const;

private:
    std::string buildGStreamerPipeline(int sensorId, int captureWidth,
                                       int captureHeight, int fps,
                                       int flipMethod) const;

    cv::VideoCapture cap_;
    CameraConfig config_;
    bool isOpened_ = false;
};

} // namespace fcw
