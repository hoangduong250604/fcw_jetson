// ==============================================================================
// Traffic Sign Processor - HSV color classification (secondary-model path unused)
// ==============================================================================
// This processor is disabled by default. The main YOLOv8 model trained on
// BDD100K already detects traffic lights (class 8) and traffic signs (class 9)
// directly, so a secondary classifier is not needed. When enabled, traffic
// light COLOR is classified via HSV thresholding with CLAHE brightness
// normalization (see classifyLightByHSV) — not by the unimplemented secondary
// YOLOv8n scaffolding also present in this file (see traffic_sign_processor.h).
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

    // NOTE: the active classifier is HSV-based (classifyLightByHSV) and
    // needs neither a model file nor labels — those only matter for the
    // unimplemented secondary-model scaffolding (see header). Previously
    // this function required config_.modelPath to exist even though nothing
    // ever loaded it for inference, which meant enabling this processor did
    // nothing unless an unrelated file happened to be present. Labels are
    // loaded best-effort for that future path but are not required.
    if (!config_.labelsPath.empty()) {
        loadLabels(config_.labelsPath);
    }

    clahe_ = cv::createCLAHE(2.0, cv::Size(4, 4));

    LOG_INFO("TrafficSign", "Traffic sign processor initialized (HSV classifier)");
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

    // Normalize brightness/exposure before thresholding: CLAHE on the V
    // channel only (hue/saturation, which encode color, are untouched).
    // Fixed HSV thresholds are known to be sensitive to glare, backlight,
    // and over/under-exposed crops — this reduces (does not eliminate) that.
    if (clahe_) {
        std::vector<cv::Mat> hsvChannels(3);
        cv::split(hsv, hsvChannels);
        clahe_->apply(hsvChannels[2], hsvChannels[2]);
        cv::merge(hsvChannels, hsv);
    }

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
