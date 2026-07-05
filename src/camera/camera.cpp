// ==============================================================================
// Camera Implementation
// ==============================================================================

#include "camera.h"
#include "logger.h"
#include <sstream>
#include <chrono>

namespace fcw {

Camera::Camera() {}

Camera::~Camera() {
    release();
}

bool Camera::openVideo(const std::string& videoPath) {
    cap_.open(videoPath);
    if (!cap_.isOpened()) {
        LOG_ERROR("Camera", "Failed to open video: " + videoPath);
        return false;
    }
    isOpened_ = true;
    config_.imageWidth = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.imageHeight = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    LOG_INFO("Camera", "Opened video: " + videoPath +
             " (" + std::to_string(config_.imageWidth) + "x" +
             std::to_string(config_.imageHeight) + ")");
    return true;
}

bool Camera::openCSI(int sensorId, int captureWidth,
                     int captureHeight, int fps, int flipMethod) {
    std::string pipeline = buildGStreamerPipeline(sensorId, captureWidth,
                                                  captureHeight, fps, flipMethod);
    LOG_INFO("Camera", "GStreamer pipeline: " + pipeline);

    cap_.open(pipeline, cv::CAP_GSTREAMER);
    if (!cap_.isOpened()) {
        LOG_ERROR("Camera", "Failed to open CSI camera with GStreamer");
        return false;
    }
    isOpened_ = true;
    isLiveCamera_ = true;
    // NOTE: no CAP_PROP_BUFFERSIZE here — buffering is already controlled
    // directly in the GStreamer pipeline string (appsink drop=true
    // max-buffers=1), which is the reliable way to do it for this backend.
    config_.captureWidth = captureWidth;
    config_.captureHeight = captureHeight;
    config_.fps = fps;
    config_.imageWidth = captureWidth;
    config_.imageHeight = captureHeight;

    cachedFPS_ = cap_.get(cv::CAP_PROP_FPS);
    if (cachedFPS_ <= 0.0) cachedFPS_ = fps;  // GStreamer backend often doesn't report this
    cachedFrameCount_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    LOG_INFO("Camera", "CSI camera opened: " +
             std::to_string(captureWidth) + "x" + std::to_string(captureHeight) +
             " @ " + std::to_string(fps) + "fps");
    startCaptureThread();
    return true;
}

bool Camera::openUSB(int deviceId, int width, int height) {
    cap_.open(deviceId);
    if (!cap_.isOpened()) {
        LOG_ERROR("Camera", "Failed to open USB camera: " + std::to_string(deviceId));
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    isOpened_ = true;
    isLiveCamera_ = true;
    cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);

    config_.imageWidth = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.imageHeight = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    cachedFPS_ = cap_.get(cv::CAP_PROP_FPS);
    cachedFrameCount_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));

    LOG_INFO("Camera", "USB camera opened: device " + std::to_string(deviceId));
    startCaptureThread();
    return true;
}

void Camera::startCaptureThread() {
    streamEnded_ = false;
    hasFrame_ = false;
    captureThreadRunning_.store(true);
    captureThread_ = std::thread(&Camera::captureLoop, this);
}

void Camera::captureLoop() {
    // Per-thread timing base for stamping frames (mirrors getPositionMs()'s
    // live-camera fallback, now computed on the one thread that owns cap_).
    auto startTime = std::chrono::high_resolution_clock::now();

    while (captureThreadRunning_.load()) {
        cv::Mat frame;
        bool ok = cap_.read(frame);  // blocks at the sensor's own pace
        if (!ok || frame.empty()) {
            std::lock_guard<std::mutex> lock(frameMutex_);
            streamEnded_ = true;
            frameCv_.notify_all();
            break;
        }

        float posMs = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - startTime).count();
        {
            std::lock_guard<std::mutex> lock(frameMutex_);
            latestFrame_ = frame;  // overwrite: only the newest frame is ever kept
            latestPositionMs_ = posMs;
            hasFrame_ = true;
        }
        frameCv_.notify_one();
    }
}

bool Camera::read(cv::Mat& frame) {
    if (!isOpened_) return false;

    if (isLiveCamera_) {
        std::unique_lock<std::mutex> lock(frameMutex_);
        frameCv_.wait(lock, [this] { return hasFrame_ || streamEnded_; });
        if (!hasFrame_) return false;  // stream ended before any frame arrived
        frame = latestFrame_;
        return true;
    }

    return cap_.read(frame);  // video file: unchanged, every frame processed
}

cv::Mat Camera::getIntrinsicMatrix() const {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = config_.fx;
    K.at<double>(1, 1) = config_.fy;
    K.at<double>(0, 2) = config_.cx;
    K.at<double>(1, 2) = config_.cy;
    return K;
}

cv::Mat Camera::getDistortionCoeffs() const {
    return cv::Mat::zeros(1, 5, CV_64F);
}

bool Camera::isOpened() const {
    return isOpened_ && cap_.isOpened();
}

void Camera::release() {
    // Stop the background capture thread FIRST (if running) so cap_ is only
    // ever touched by one thread at a time before it's released.
    if (captureThreadRunning_.load()) {
        captureThreadRunning_.store(false);
        frameCv_.notify_all();
        if (captureThread_.joinable()) captureThread_.join();
    }
    if (cap_.isOpened()) {
        cap_.release();
    }
    isOpened_ = false;
}

int Camera::getWidth() const { return config_.imageWidth; }
int Camera::getHeight() const { return config_.imageHeight; }

double Camera::getFPS() const {
    // Live camera: cached at open() time, before the capture thread starts
    // touching cap_ — avoids a cross-thread cap_.get() call here.
    return isLiveCamera_ ? cachedFPS_ : cap_.get(cv::CAP_PROP_FPS);
}

float Camera::getPositionMs() const {
    if (isLiveCamera_) {
        // Timestamp cached by captureLoop() alongside the frame it belongs
        // to (see Camera::captureLoop) instead of querying cap_ here, which
        // would race with the capture thread's own cap_ access.
        std::lock_guard<std::mutex> lock(frameMutex_);
        return latestPositionMs_;
    }
    float pos = static_cast<float>(cap_.get(cv::CAP_PROP_POS_MSEC));
    if (pos > 0.0f) return pos;  // Video file: use video timestamp
    // Fallback: use system clock (video files with no timestamp support)
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(now - startTime).count();
}

int Camera::getFrameCount() const {
    return isLiveCamera_ ? cachedFrameCount_ : static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
}

void Camera::setConfig(const CameraConfig& config) {
    config_ = config;
}

const CameraConfig& Camera::getConfig() const {
    return config_;
}

std::string Camera::buildGStreamerPipeline(int sensorId, int captureWidth,
                                            int captureHeight, int fps,
                                            int flipMethod) const {
    // Jetson Nano CSI camera GStreamer pipeline
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensorId << " ! "
       << "video/x-raw(memory:NVMM), "
       << "width=(int)" << captureWidth << ", "
       << "height=(int)" << captureHeight << ", "
       << "framerate=(fraction)" << fps << "/1 ! "
       << "nvvidconv flip-method=" << flipMethod << " ! "
       << "video/x-raw, width=(int)" << captureWidth
       << ", height=(int)" << captureHeight
       << ", format=(string)BGRx ! "
       << "videoconvert ! "
       << "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1";
    return ss.str();
}

} // namespace fcw
