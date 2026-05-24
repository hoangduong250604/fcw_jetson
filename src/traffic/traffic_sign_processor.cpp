// ==============================================================================
// Traffic Sign Processor - Stub implementation
// ==============================================================================
// This processor is currently disabled. The main YOLOv8 model trained on
// BDD100K already detects traffic lights (class 8) and traffic signs (class 9)
// directly, so a secondary classifier is not needed.
// ==============================================================================

#include "traffic_sign_processor.h"
#include "logger.h"

#include <fstream>

namespace fcw {

TrafficSignProcessor::TrafficSignProcessor() = default;
TrafficSignProcessor::~TrafficSignProcessor() = default;

bool TrafficSignProcessor::init(const TrafficSignConfig& config) {
    config_ = config;

    if (!config_.enabled) {
        LOG_INFO("TrafficSign", "Traffic sign processor disabled");
        return false;
    }

    // Check if model file exists
    std::ifstream f(config_.modelPath);
    if (!f.good()) {
        LOG_INFO("TrafficSign", "Secondary model not found: " + config_.modelPath + " (disabled)");
        initialized_ = false;
        return false;
    }

    // Load labels
    if (!loadLabels(config_.labelsPath)) {
        LOG_WARNING("TrafficSign", "Failed to load labels: " + config_.labelsPath);
        initialized_ = false;
        return false;
    }

    LOG_INFO("TrafficSign", "Traffic sign processor initialized (secondary model)");
    initialized_ = true;
    return true;
}

TrafficSignResult TrafficSignProcessor::process(const cv::Mat& frame,
                                                 const DetectionResult& detections,
                                                 int frameId) {
    // If not initialized, return empty result
    if (!initialized_) return TrafficSignResult{};

    // Use cached result if not on processing interval
    if (frameId - cachedResult_.frameId < config_.processInterval && cachedResult_.valid) {
        return cachedResult_;
    }

    TrafficSignResult result;
    result.frameId = frameId;
    result.valid = true;

    // Process traffic light detections from main model
    for (const auto& det : detections.detections) {
        if (det.classId == config_.trafficLightClassId) {
            // Use HSV analysis on the traffic light crop
            cv::Rect roi = clampROI(det.getRect(), frame.cols, frame.rows);
            if (roi.width > 0 && roi.height > 0) {
                cv::Mat crop = frame(roi);
                TrafficLightState state = classifyLightByHSV(crop);
                if (state != TrafficLightState::UNKNOWN) {
                    result.lightState = state;
                    result.lightConfidence = det.confidence;
                }
            }
        }
    }

    cachedResult_ = result;
    return result;
}

// ---- Private helpers ----

std::vector<Detection> TrafficSignProcessor::detectInROI(const cv::Mat& /*roi*/) {
    return {};
}

void TrafficSignProcessor::classifyCrop(const cv::Mat& /*crop*/, int& bestClassId, float& bestConf) {
    bestClassId = -1;
    bestConf = 0.0f;
}

std::vector<Detection> TrafficSignProcessor::decodeOutput(float* /*output*/, int /*numAnchors*/, int /*numClasses*/) {
    return {};
}

std::vector<Detection> TrafficSignProcessor::applyNMS(const std::vector<Detection>& dets, float /*iouThresh*/) {
    return dets;
}

bool TrafficSignProcessor::loadLabels(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    labels_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) labels_.push_back(line);
    }
    numClasses_ = static_cast<int>(labels_.size());
    return !labels_.empty();
}

TrafficLightState TrafficSignProcessor::classToLightState(int classId) const {
    if (classId == CLASS_GREEN_LIGHT) return TrafficLightState::GREEN;
    if (classId == CLASS_RED_LIGHT) return TrafficLightState::RED;
    if (classId == CLASS_YELLOW_LIGHT) return TrafficLightState::YELLOW;
    return TrafficLightState::UNKNOWN;
}

bool TrafficSignProcessor::isTrafficLightClass(int classId) const {
    return classId == CLASS_GREEN_LIGHT || classId == CLASS_RED_LIGHT || classId == CLASS_YELLOW_LIGHT;
}

TrafficLightState TrafficSignProcessor::classifyLightByHSV(const cv::Mat& crop) const {
    if (crop.empty()) return TrafficLightState::UNKNOWN;

    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    // Count pixels in red, yellow, green ranges
    cv::Mat redMask1, redMask2, yellowMask, greenMask;
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), redMask1);
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), redMask2);
    cv::inRange(hsv, cv::Scalar(15, 100, 100), cv::Scalar(35, 255, 255), yellowMask);
    cv::inRange(hsv, cv::Scalar(40, 50, 100), cv::Scalar(90, 255, 255), greenMask);

    int redCount = cv::countNonZero(redMask1) + cv::countNonZero(redMask2);
    int yellowCount = cv::countNonZero(yellowMask);
    int greenCount = cv::countNonZero(greenMask);

    int totalPixels = crop.rows * crop.cols;
    float minRatio = 0.05f;  // At least 5% of pixels

    if (redCount > yellowCount && redCount > greenCount && redCount > totalPixels * minRatio) {
        return TrafficLightState::RED;
    } else if (greenCount > yellowCount && greenCount > redCount && greenCount > totalPixels * minRatio) {
        return TrafficLightState::GREEN;
    } else if (yellowCount > totalPixels * minRatio) {
        return TrafficLightState::YELLOW;
    }

    return TrafficLightState::UNKNOWN;
}

cv::Rect TrafficSignProcessor::clampROI(const cv::Rect& roi, int frameW, int frameH) const {
    int x = std::max(0, roi.x);
    int y = std::max(0, roi.y);
    int w = std::min(roi.width, frameW - x);
    int h = std::min(roi.height, frameH - y);
    if (w <= 0 || h <= 0) return cv::Rect(0, 0, 0, 0);
    return cv::Rect(x, y, w, h);
}

} // namespace fcw
