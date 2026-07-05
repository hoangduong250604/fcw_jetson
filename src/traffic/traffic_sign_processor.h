#pragma once
// ==============================================================================
// Traffic Sign Processor - traffic light color classification
// ==============================================================================
//
// CURRENT BEHAVIOR: classification is done by HSV color-range analysis on
// cropped traffic-light regions detected by the main BDD100K detector
// (classifyLightByHSV), with CLAHE brightness normalization applied first to
// reduce sensitivity to glare/backlight/exposure. Disabled by default
// (traffic_sign.enabled: false in system_config.yaml) since the main
// detector already localizes traffic lights/signs directly (class 8/9).
//
// The secondary-YOLOv8n-classifier scaffolding below (detectInROI,
// classifyCrop, decodeOutput, applyNMS, the ONNX/dnn session members) is NOT
// implemented — those are stub methods that return empty results. process()
// never calls them; it only ever runs the HSV path. Kept as a documented
// extension point for a future "replace HSV with a trained classifier"
// upgrade, not as a currently-active AI classification path.
// ==============================================================================

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef USE_ONNXRUNTIME
#if defined(__MINGW32__) && !defined(_stdcall)
#define _stdcall __stdcall
#endif
#include <onnxruntime_cxx_api.h>
#endif

#include "detection_result.h"

namespace fcw {

/** Configuration for the traffic sign/light secondary detector */
struct TrafficSignConfig {
    std::string modelPath;                     // Path to yolov8n_traffic.onnx
    std::string labelsPath = "./models/traffic_labels.txt";
    int inputSize = 320;                       // Model input size (320x320)
    float confThreshold = 0.40f;               // Confidence threshold
    float nmsThreshold = 0.45f;                // NMS threshold
    int processInterval = 3;                   // Run every N frames (cache between)
    bool enabled = true;                       // Enable/disable
    float roiPadding = 0.3f;                   // Expand crop ROI by 30% for context
    // Class IDs in the MAIN detector for traffic objects
    // COCO: traffic light=9, stop sign=11
    // BDD100K: traffic light=8, traffic sign=9
    int trafficLightClassId = 8;               // Main detector class for traffic light
    int trafficSignClassId = 9;                // Main detector class for traffic sign
};

/** Result of traffic sign/light classification */
struct TrafficSignResult {
    // Traffic light state (from AI classification)
    TrafficLightState lightState = TrafficLightState::UNKNOWN;
    float lightConfidence = 0.0f;

    // Detected traffic signs
    struct SignInfo {
        std::string name;           // e.g. "Stop", "Speed Limit 60"
        float confidence;
        cv::Rect bbox;              // Position in original frame
    };
    std::vector<SignInfo> signs;

    // Metadata
    bool valid = false;             // Whether result is from current frame
    int frameId = -1;               // Frame when last processed
};

/**
 * Traffic light color classifier. Currently HSV-based (see file header for
 * why); the secondary-model path is unimplemented scaffolding.
 */
class TrafficSignProcessor {
public:
    TrafficSignProcessor();
    ~TrafficSignProcessor();

    /** Initialize with config. Returns false if model not found (non-fatal). */
    bool init(const TrafficSignConfig& config);

    /**
     * Process traffic detections from the main detector.
     *
     * @param frame       Current frame
     * @param detections  Full detection result from main detector
     * @param frameId     Current frame number
     * @return            Classification result (may be cached from previous frame)
     */
    TrafficSignResult process(const cv::Mat& frame,
                               const DetectionResult& detections,
                               int frameId);

    /** Check if processor is initialized and enabled */
    bool isReady() const { return initialized_ && config_.enabled; }

    /** Get last result (for visualization on non-processing frames) */
    const TrafficSignResult& getLastResult() const { return cachedResult_; }

private:
    /** Run YOLOv8n inference on a cropped ROI */
    std::vector<Detection> detectInROI(const cv::Mat& roi);

    /** Classify a crop using the model as classifier (best class across all anchors) */
    void classifyCrop(const cv::Mat& crop, int& bestClassId, float& bestConf);

    /** Decode YOLOv8 output tensor */
    std::vector<Detection> decodeOutput(float* output, int numAnchors, int numClasses);

    /** Apply NMS */
    std::vector<Detection> applyNMS(const std::vector<Detection>& dets, float iouThresh);

    /** Load class labels from file */
    bool loadLabels(const std::string& path);

    /** Map detection class to TrafficLightState */
    TrafficLightState classToLightState(int classId) const;

    /** Check if class ID is a traffic light class */
    bool isTrafficLightClass(int classId) const;

    /** Classify traffic light color using HSV analysis */
    TrafficLightState classifyLightByHSV(const cv::Mat& crop) const;

    /** Clamp ROI to frame bounds */
    cv::Rect clampROI(const cv::Rect& roi, int frameW, int frameH) const;

    TrafficSignConfig config_;
    bool initialized_ = false;

#ifdef USE_ONNXRUNTIME
    Ort::Env ortEnv_{ORT_LOGGING_LEVEL_WARNING, "traffic"};
    std::unique_ptr<Ort::Session> ortSession_;
    Ort::AllocatorWithDefaultOptions ortAllocator_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<std::string> inputNamesOwned_;
    std::vector<std::string> outputNamesOwned_;
#else
    cv::dnn::Net net_;
#endif

    int numClasses_ = 13;
    int numAnchors_ = 2100;   // 320x320 → fewer anchors than 640x640
    std::vector<std::string> labels_;
    TrafficSignResult cachedResult_;

    // CLAHE applied to the HSV V-channel before color thresholding, to
    // reduce classifyLightByHSV's sensitivity to glare/backlight/exposure.
    // Created once in init() and reused (allocating a fresh CLAHE per call
    // would be wasteful even though crops are small).
    cv::Ptr<cv::CLAHE> clahe_;

    // Traffic light class indices in the 13-class merged model:
    // 0=Green Light, 1=Red Light, 2=Yellow Light
    static constexpr int CLASS_GREEN_LIGHT = 0;
    static constexpr int CLASS_RED_LIGHT = 1;
    static constexpr int CLASS_YELLOW_LIGHT = 2;
};

} // namespace fcw
