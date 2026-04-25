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
    config_.captureWidth = captureWidth;
    config_.captureHeight = captureHeight;
    config_.fps = fps;
    config_.imageWidth = captureWidth;
    config_.imageHeight = captureHeight;

    LOG_INFO("Camera", "CSI camera opened: " +
             std::to_string(captureWidth) + "x" + std::to_string(captureHeight) +
             " @ " + std::to_string(fps) + "fps");
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

    config_.imageWidth = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.imageHeight = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    LOG_INFO("Camera", "USB camera opened: device " + std::to_string(deviceId));
    return true;
}

bool Camera::read(cv::Mat& frame) {
    if (!isOpened_) return false;
    return cap_.read(frame);
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
    if (cap_.isOpened()) {
        cap_.release();
    }
    isOpened_ = false;
}

int Camera::getWidth() const { return config_.imageWidth; }
int Camera::getHeight() const { return config_.imageHeight; }

double Camera::getFPS() const {
    return cap_.get(cv::CAP_PROP_FPS);
}

float Camera::getPositionMs() const {
    float pos = static_cast<float>(cap_.get(cv::CAP_PROP_POS_MSEC));
    if (pos > 0.0f) return pos;  // Video file: use video timestamp
    // Live camera fallback: use system clock
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(now - startTime).count();
}

int Camera::getFrameCount() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
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
       << "video/x-raw, format=(string)BGR ! appsink";
    return ss.str();
}

} // namespace fcw
