#pragma once
// ==============================================================================
// Detection Result - Common data structures for detection output
// ==============================================================================

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "math_utils.h"

namespace fcw {

/** Traffic light state detected from image analysis */
enum class TrafficLightState {
    UNKNOWN,
    RED,
    YELLOW,
    GREEN
};

/**
 * Single object detection result from YOLOv8.
 */
struct Detection {
    utils::BBox bbox;           // Bounding box (x1, y1, x2, y2)
    float confidence;           // Detection confidence [0, 1]
    int classId;                // Class ID (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
    std::string className;      // Human-readable class name

    // Convenience methods
    cv::Rect getRect() const { return bbox.toRect(); }
    float getCenterX() const { return bbox.centerX(); }
    float getCenterY() const { return bbox.centerY(); }
    float getWidth() const { return bbox.width(); }
    float getHeight() const { return bbox.height(); }
    float getArea() const { return bbox.area(); }
    float getBottomY() const { return bbox.y2; }
};

/**
 * Detection results for a single frame.
 */
struct DetectionResult {
    std::vector<Detection> detections;
    double inferenceTimeMs = 0.0;
    int frameId = -1;

    int count() const { return static_cast<int>(detections.size()); }
    bool empty() const { return detections.empty(); }

    /** Filter detections by class IDs */
    DetectionResult filterByClass(const std::vector<int>& classIds) const {
        DetectionResult filtered;
        filtered.inferenceTimeMs = inferenceTimeMs;
        filtered.frameId = frameId;
        for (const auto& det : detections) {
            for (int cid : classIds) {
                if (det.classId == cid) {
                    filtered.detections.push_back(det);
                    break;
                }
            }
        }
        return filtered;
    }

    /** Get the detection closest to image center (most likely lead vehicle) */
    const Detection* getLeadVehicle(float imageCenterX) const {
        if (detections.empty()) return nullptr;
        const Detection* lead = nullptr;
        float minDist = std::numeric_limits<float>::max();

        for (const auto& det : detections) {
            // Skip non-vehicle classes (e.g. traffic light = COCO 9)
            if (det.classId == 9) continue;
            float dist = std::abs(det.getCenterX() - imageCenterX);
            // Prefer vehicles that are closest (bottom of bbox is lower)
            // and near the center of the lane
            float score = dist - det.getBottomY() * 0.5f;
            if (score < minDist) {
                minDist = score;
                lead = &det;
            }
        }
        return lead;
    }

    /** Get all traffic light detections (COCO class 9) */
    std::vector<const Detection*> getTrafficLights() const {
        std::vector<const Detection*> lights;
        for (const auto& det : detections) {
            if (det.classId == 9) lights.push_back(&det);
        }
        return lights;
    }
};

} // namespace fcw
