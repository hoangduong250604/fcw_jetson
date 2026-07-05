// ==============================================================================
// TTC Calculator Implementation
// ==============================================================================
// CORE MODULE: Time-to-Collision estimation
//
// Two complementary methods are used:
//
// Method 1 - Distance-based TTC:
//   TTC = D / v_rel
//   Requires good distance and speed estimates.
//   Sensitive to camera calibration quality.
//
// Method 2 - Scale-based TTC:
//   TTC = s / (ds/dt)
//   where s = bbox area, ds/dt = rate of area change from Kalman filter.
//   Independent of camera calibration.
//   Less accurate at long range (bbox area small, noisy).
//
// Combined TTC:
//   TTC = w * TTC_scale + (1-w) * TTC_dist
//   Provides robustness against failure of either method.
//
// All TTC values are smoothed per-track using EMA to prevent
// alert flickering caused by noisy measurements.
// ==============================================================================

#include "ttc_calculator.h"
#include "math_utils.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fcw {

TTCCalculator::TTCCalculator() {}

TTCCalculator::TTCCalculator(const TTCConfig& config) : config_(config) {}

// ==============================================================================
// Main TTC Calculation
// ==============================================================================
std::unordered_map<int, TTCInfo> TTCCalculator::calculate(
    const std::vector<Track*>& tracks,
    const std::unordered_map<int, DistanceInfo>& distances,
    const std::unordered_map<int, SpeedInfo>& speeds) {

    std::unordered_map<int, TTCInfo> newTTCs;

    for (const Track* track : tracks) {
        int id = track->getId();
        int classId = track->getClassId();

        // Skip static infrastructure — traffic signs and lights cannot cause collisions
        // BDD100K: 8=traffic sign, 9=traffic light
        if (classId == 8 || classId == 9) continue;

        TTCInfo info;
        info.trackId = id;

        // Get distance and speed for this track
        auto distIt = distances.find(id);
        auto speedIt = speeds.find(id);

        if (distIt == distances.end() || speedIt == speeds.end()) continue;

        const DistanceInfo& dist = distIt->second;
        const SpeedInfo& speed = speedIt->second;

        if (!dist.valid) continue;

        info.distance = dist.smoothedDistance;
        info.relativeSpeed = speed.closingSpeedMs;  // Closing speed in m/s
        info.isApproaching = speed.isApproaching;
        info.vehicleState = speed.vehicleState;     // Propagate vehicle state

        // ---- Method 1: Distance-based TTC (raw, unclamped) ----
        info.ttcDistance = computeTTCFromDistance(info.distance, info.relativeSpeed);

        // ---- Method 2: Scale-based TTC (raw, unclamped) ----
        if (config_.useScaleMethod) {
            info.ttcScale = computeTTCFromScale(track);
        }

        // ---- Combined TTC ----
        // Confidence-adaptive fusion: both the disagreement sanity-check and
        // the blend weight are computed on RAW (unclamped) TTC values, so two
        // genuinely divergent estimates can't be masked by both saturating to
        // the same clamp bucket. Only the final result is clamped below.
        info.scaleConfidence = config_.useScaleMethod ? scaleConfidenceFor(track, info.distance) : 0.0f;
        info.ttcCombined = combineTTC(info.ttcDistance, info.ttcScale, info.scaleConfidence);
        if (info.ttcCombined > 0.0f) {
            info.ttcCombined = utils::clamp(info.ttcCombined, config_.minValidTTC, config_.maxTTC);
        }

        // ---- EMA Smoothing ----
        float rawTTC = info.ttcCombined;
        auto prevIt = prevSmoothedTTC_.find(id);
        if (prevIt != prevSmoothedTTC_.end() && prevIt->second > 0.0f) {
            info.ttcSmoothed = utils::ema(rawTTC, prevIt->second, config_.smoothingAlpha);
        } else {
            info.ttcSmoothed = rawTTC;
        }

        // Store for next frame's smoothing
        if (info.ttcSmoothed > 0.0f) {
            prevSmoothedTTC_[id] = info.ttcSmoothed;
        }

        info.valid = (info.ttcSmoothed > 0.0f && info.isApproaching);

        newTTCs[id] = info;
    }

    // Cleanup old entries
    for (auto it = prevSmoothedTTC_.begin(); it != prevSmoothedTTC_.end();) {
        bool exists = false;
        for (const Track* t : tracks) {
            if (t->getId() == it->first) { exists = true; break; }
        }
        if (!exists) it = prevSmoothedTTC_.erase(it);
        else ++it;
    }

    trackTTCs_ = newTTCs;
    return trackTTCs_;
}

// ==============================================================================
// Method 1: Distance-based TTC
// ==============================================================================
float TTCCalculator::computeTTCFromDistance(float distance, float relativeSpeed) const {
    // TTC is only valid when vehicle is approaching (positive relative speed)
    if (relativeSpeed <= 0.0f || distance <= 0.0f) {
        return -1.0f;  // Not approaching or invalid distance
    }

    // Intentionally NOT clamped here: combineTTC needs the raw magnitude to
    // judge disagreement between methods. The caller clamps the final result.
    return distance / relativeSpeed;
}

// ==============================================================================
// Method 2: Scale-based TTC
// ==============================================================================
float TTCCalculator::computeTTCFromScale(const Track* track) const {
    // TTC from bounding box scale change rate
    //
    // Mathematical basis:
    //   bbox_area ∝ 1/distance^2 (inverse square law for monocular camera)
    //   ds/dt = rate of area change (from Kalman filter's vs state)
    //
    // If s is increasing → vehicle is approaching
    //   TTC_scale ≈ s / (ds/dt)
    //
    // This avoids the need for camera calibration.

    float scaleVelocity = track->getScaleVelocity();  // ds/dt from Kalman
    utils::BBox bbox = track->getBBox();
    float area = bbox.area();

    if (area <= 0.0f || scaleVelocity <= 0.0f) {
        return -1.0f;  // Not approaching (scale not increasing)
    }

    // Intentionally NOT clamped here — see computeTTCFromDistance.
    return area / scaleVelocity;
}

// ==============================================================================
// Scale-method confidence (cheap approximation of inverse-variance weighting)
// ==============================================================================
float TTCCalculator::scaleConfidenceFor(const Track* track, float distance) const {
    // Track maturity: the Kalman scale-velocity state (vs) needs a few
    // successful updates to converge; a brand-new track's vs is unreliable.
    float maturity = utils::clamp(
        static_cast<float>(track->getHitCount()) / config_.scaleMatureHits, 0.0f, 1.0f);

    // Range attenuation: bbox area shrinks ~1/distance^2, so area noise
    // dominates the scale signal at long range (matches this file's own
    // header note that scale-based TTC is "less accurate at long range").
    float rangeConf = 1.0f - utils::clamp(distance / config_.scaleMaxUsefulRange, 0.0f, 1.0f);
    rangeConf = std::max(rangeConf, config_.scaleMinConfidenceFloor);

    return maturity * rangeConf;
}

// ==============================================================================
// Combined TTC
// ==============================================================================
float TTCCalculator::combineTTC(float ttcDist, float ttcScale, float scaleConfidence) const {
    bool distValid = ttcDist > 0.0f;
    bool scaleValid = ttcScale > 0.0f;

    if (distValid && scaleValid && config_.useScaleMethod) {
        // Sanity check on RAW values (not yet clamped): if scale-based TTC
        // disagrees too much with distance-based, the scale velocity is
        // unreliable (e.g. lateral motion, new track) – ignore it entirely.
        // Doing this before any clamping prevents two genuinely divergent
        // raw estimates from both saturating into the same clamp bucket and
        // being wrongly read as "agreeing".
        if (std::abs(ttcScale - ttcDist) > config_.scaleSanityThreshold) {
            return ttcDist;
        }
        // Confidence-adaptive blend: scaleWeight is the ceiling, reached only
        // when scaleConfidence == 1 (mature track, close range). This replaces
        // the previous fixed 70/30 split with a weight that tapers down for
        // new tracks and long-range targets, where the scale method is known
        // to be noisier.
        float w = config_.scaleWeight * scaleConfidence;
        return w * ttcScale + (1.0f - w) * ttcDist;
    } else if (distValid) {
        return ttcDist;
    } else if (scaleValid) {
        return ttcScale;
    }

    return -1.0f;  // No valid TTC
}

// ==============================================================================
// Getters
// ==============================================================================
TTCInfo TTCCalculator::getTTC(int trackId) const {
    auto it = trackTTCs_.find(trackId);
    if (it != trackTTCs_.end()) return it->second;
    return TTCInfo();
}

TTCInfo TTCCalculator::getMostCriticalTTC() const {
    TTCInfo critical;
    critical.ttcSmoothed = std::numeric_limits<float>::max();

    for (const auto& [id, info] : trackTTCs_) {
        if (info.valid && info.ttcSmoothed < critical.ttcSmoothed) {
            critical = info;
        }
    }

    if (critical.trackId == -1) {
        critical.ttcSmoothed = -1.0f;  // No valid TTC found
    }

    return critical;
}

void TTCCalculator::setConfig(const TTCConfig& config) {
    config_ = config;
}

} // namespace fcw
