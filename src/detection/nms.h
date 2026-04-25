#pragma once
// ==============================================================================
// NMS - Non-Maximum Suppression for YOLOv8 detections
// ==============================================================================

#include <vector>
#include "detection_result.h"

namespace fcw {

/**
 * Apply Non-Maximum Suppression to remove overlapping detections.
 *
 * Algorithm:
 *   1. Sort detections by confidence (descending)
 *   2. Pick the highest confidence detection
 *   3. Remove all detections with IoU > threshold (same class)
 *   4. Repeat until no detections remain
 *
 * @param detections  Input candidate detections
 * @param iouThreshold  IoU threshold for suppression
 * @return  Filtered detections after NMS
 */
std::vector<Detection> applyNMS(std::vector<Detection>& detections, float iouThreshold);

} // namespace fcw
