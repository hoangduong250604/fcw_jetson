// ==============================================================================
// Distance Estimator Implementation
// ==============================================================================

#include "distance_estimator.h"
#include "math_utils.h"
#include "logger.h"

namespace fcw {

DistanceEstimator::DistanceEstimator() {}

DistanceEstimator::DistanceEstimator(const DistanceConfig& config)
    : config_(config) {}

std::unordered_map<int, DistanceInfo> DistanceEstimator::estimate(
    const std::vector<Track*>& tracks) {

    // Remove entries for tracks that no longer exist
    std::unordered_map<int, DistanceInfo> newDistances;

    for (const Track* track : tracks) {
        int id = track->getId();
        utils::BBox bbox = track->getBBox();

        DistanceInfo info;
        info.trackId = id;
        info.bbox = bbox.toRect();

        // ===== EDGE TRUNCATION SHIELD =====
        // When bbox touches screen edges, height is truncated → distance inflated
        bool touchesLeft = (bbox.x1 <= 10.0f);
        bool touchesRight = (bbox.x2 >= imageWidth_ - 10.0f);
        bool touchesBottom = (bbox.y2 >= imageHeight_ - 10.0f);
        bool isTruncated = (touchesLeft || touchesRight || touchesBottom) &&
                           (bbox.height() > imageHeight_ * 0.25f);

        if (isTruncated) {
            // Vehicle is clipped at screen edge → override with proximity estimate
            float ratio = std::min(1.0f, bbox.height() / static_cast<float>(imageHeight_));
            info.rawDistance = std::max(1.0f, 4.5f - (ratio * 3.0f));
        } else {
            info.rawDistance = computeDistance(bbox);
        }

        // Clamp to valid range
        info.rawDistance = utils::clamp(info.rawDistance,
                                        config_.minDistance, config_.maxDistance);

        // Apply EMA smoothing
        auto prev = trackDistances_.find(id);
        if (prev != trackDistances_.end()) {
            info.previousDistance = prev->second.smoothedDistance;
            info.smoothedDistance = utils::ema(info.rawDistance,
                                               prev->second.smoothedDistance,
                                               config_.smoothingAlpha);
        } else {
            info.previousDistance = info.rawDistance;
            info.smoothedDistance = info.rawDistance;
        }

        info.valid = (info.smoothedDistance >= config_.minDistance &&
                      info.smoothedDistance <= config_.maxDistance);

        // ===== LATERAL OFFSET + EGO-LANE CORRIDOR =====
        // Decompose Euclidean distance into Z (longitudinal) and X (lateral)
        float pixelOffsetX = bbox.centerX() - cameraModel_.cx;
        float angleX = std::atan2(pixelOffsetX, cameraModel_.fx);

        info.longitudinalDist = info.smoothedDistance * std::cos(angleX);
        info.lateralOffset = info.smoothedDistance * std::sin(angleX);

        // Vehicle is in ego path if lateral offset within corridor
        info.inEgoPath = (std::abs(info.lateralOffset) < config_.corridorHalfWidth);

        newDistances[id] = info;
    }

    trackDistances_ = newDistances;
    return trackDistances_;
}

float DistanceEstimator::estimateSingle(const utils::BBox& bbox) const {
    return computeDistance(bbox);
}

float DistanceEstimator::computeDistance(const utils::BBox& bbox) const {
    float distance = -1.0f;

    switch (config_.method) {
        case DistanceMethod::BBOX_HEIGHT: {
            float bboxHeight = bbox.height();
            distance = cameraModel_.estimateDistance(bboxHeight, config_.referenceHeight);
            break;
        }

        case DistanceMethod::GROUND_PLANE: {
            float bottomY = bbox.y2;
            distance = cameraModel_.estimateDistanceGroundPlane(bottomY);
            break;
        }

        case DistanceMethod::COMBINED: {
            float d1 = cameraModel_.estimateDistance(bbox.height(), config_.referenceHeight);
            float d2 = cameraModel_.estimateDistanceGroundPlane(bbox.y2);

            if (d1 > 0.0f && d2 > 0.0f) {
                distance = (d1 + d2) / 2.0f;
            } else if (d1 > 0.0f) {
                distance = d1;
            } else {
                distance = d2;
            }
            break;
        }

        case DistanceMethod::BEV: {
            cv::Point2f bottomCenter(bbox.centerX(), bbox.y2);
            if (bevEstimator_) {
                distance = bevEstimator_->estimateDistance(bottomCenter, imageWidth_, imageHeight_);
                // Fallback to bbox height if BEV not calibrated
                if (distance < 0.0f) {
                    distance = cameraModel_.estimateDistance(bbox.height(), config_.referenceHeight);
                }
            } else {
                distance = cameraModel_.estimateDistance(bbox.height(), config_.referenceHeight);
            }
            break;
        }
    }

    return distance;
}

void DistanceEstimator::setCameraModel(const CameraModel& model) {
    cameraModel_ = model;
}

void DistanceEstimator::setBEVEstimator(const std::string& calibFilePath) {
    if (!bevEstimator_) {
        bevEstimator_ = std::make_unique<BEVDistanceEstimator>();
    }
    bevEstimator_->loadCalibration(calibFilePath);
}

void DistanceEstimator::setConfig(const DistanceConfig& config) {
    config_ = config;
}

DistanceInfo DistanceEstimator::getDistance(int trackId) const {
    auto it = trackDistances_.find(trackId);
    if (it != trackDistances_.end()) {
        return it->second;
    }
    return DistanceInfo();
}

} // namespace fcw
