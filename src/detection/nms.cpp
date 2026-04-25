// ==============================================================================
// NMS Implementation - Non-Maximum Suppression
// ==============================================================================

#include "nms.h"
#include "math_utils.h"
#include <algorithm>

namespace fcw {

std::vector<Detection> applyNMS(std::vector<Detection>& detections, float iouThreshold) {
    std::vector<Detection> result;
    if (detections.empty()) return result;

    // Sort by confidence, descending
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    // Track which detections are suppressed
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;

        result.push_back(detections[i]);

        // Suppress overlapping detections of the same class
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;

            float iou = utils::computeIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

} // namespace fcw
