#pragma once
// ==============================================================================
// Camera Module - Capture from CSI/USB camera or video file
// ==============================================================================

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
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

    /**
     * Background capture loop (live camera only, started by openCSI/openUSB).
     * Continuously performs a blocking cap_.read() at the sensor's own pace
     * and publishes only the latest frame (+ its capture timestamp) under
     * frameMutex_, overwriting whatever was there before.
     *
     * This replaces the previous per-call timing-heuristic drain loop
     * (repeated grab() calls, measuring elapsed ms to guess whether the
     * buffer was empty) with the standard dedicated-capture-thread pattern:
     * no guessed thresholds, and read() never adds extra latency when the
     * buffer is already empty (previously the common case, since it's the
     * outcome of the pipeline keeping up with the camera).
     *
     * All access to cap_ after this thread starts happens ONLY here —
     * getFPS()/getFrameCount() are cached before the thread starts, and
     * getPositionMs() reads a timestamp cached alongside the frame — so no
     * other thread ever touches cap_ concurrently (OpenCV's VideoCapture
     * does not guarantee thread-safety, even for read-only property calls).
     */
    void captureLoop();

    /** Start the background capture thread (called at the end of openCSI/openUSB) */
    void startCaptureThread();

    cv::VideoCapture cap_;
    CameraConfig config_;
    bool isOpened_ = false;
    bool isLiveCamera_ = false;  // true for CSI/USB, false for video file

    // Cached at open() time, before the capture thread starts, so these
    // getters never need to touch cap_ from a different thread afterward.
    double cachedFPS_ = 0.0;
    int cachedFrameCount_ = 0;

    // Background capture thread state (live camera only)
    std::thread captureThread_;
    std::atomic<bool> captureThreadRunning_{false};
    mutable std::mutex frameMutex_;
    std::condition_variable frameCv_;
    cv::Mat latestFrame_;
    float latestPositionMs_ = 0.0f;
    bool hasFrame_ = false;
    bool streamEnded_ = false;
};

} // namespace fcw
