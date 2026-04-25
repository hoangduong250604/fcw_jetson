// ==============================================================================
// Speed Estimator Implementation - TIMESTAMP-BASED
// ==============================================================================
// Closing Speed: Linear regression on (time, distance) → slope = m/s directly.
// Ego Speed: Injected from external source (KITTI OXTS / CAN Bus / GPS).
// Optical Flow computation removed entirely for accuracy + performance.
// ==============================================================================

#include "speed_estimator.h"
#include "math_utils.h"
#include "logger.h"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace fcw {

SpeedEstimator::SpeedEstimator() {}

SpeedEstimator::SpeedEstimator(const SpeedConfig& config) : config_(config) {}

// ==============================================================================
// Main Estimation - Timestamp-based
// ==============================================================================
std::unordered_map<int, SpeedInfo> SpeedEstimator::estimate(
    const std::unordered_map<int, DistanceInfo>& distances,
    float timestampMs) {

    std::unordered_map<int, SpeedInfo> newSpeeds;
    lastTimestampMs_ = timestampMs;
    
    // egoSpeedKmh_ is set externally via setEgoSpeed() before calling this
    float egoSpeedKmh = egoSpeedKmh_;

    for (const auto& [trackId, distInfo] : distances) {
        if (!distInfo.valid) continue;

        SpeedInfo info;
        info.trackId = trackId;
        info.egoSpeedKmh = egoSpeedKmh;

        auto& history = trackHistory_[trackId];
        history.framesSeen++;
        history.currentDistance = distInfo.smoothedDistance;

        // Store distance + timestamp samples
        history.distanceSamples.push_back(distInfo.smoothedDistance);
        history.timeSamples.push_back(timestampMs);
        while (static_cast<int>(history.distanceSamples.size()) > config_.regressionWindow) {
            history.distanceSamples.pop_front();
            history.timeSamples.pop_front();
        }

        // Need minimum samples for regression
        if (static_cast<int>(history.distanceSamples.size()) < 3) {
            info.valid = true;
            newSpeeds[trackId] = info;
            continue;
        }

        // ========== Closing Speed from timestamp-based regression ==========
        float regressionVariance = 0.0f;
        float closingSpeedMs = computeRegressionSpeed(
            history.distanceSamples, history.timeSamples, regressionVariance);
        closingSpeedMs = utils::clamp(closingSpeedMs, -config_.maxSpeed, config_.maxSpeed);

        // Smooth with median filter
        history.closingSpeedHistory.push_back(closingSpeedMs);
        while (static_cast<int>(history.closingSpeedHistory.size()) > config_.medianWindow) {
            history.closingSpeedHistory.pop_front();
        }
        float smoothedClosingSpeed = computeMedian(history.closingSpeedHistory);
        float closingSpeedKmh = utils::msToKmh(smoothedClosingSpeed);

        // ========== TURN SUPPRESSION (from OXTS yaw rate) ==========
        // When ego vehicle is turning, bbox displacement has large lateral component
        // → closing speed estimate becomes unreliable → widen dead zone
        bool egoIsTurning = false;
        if (currentOxts_.valid) {
            float yawRate = std::abs(currentOxts_.wz);  // rad/s
            if (yawRate > config_.turnYawRateThreshold) {
                egoIsTurning = true;
                // Dampen closing speed during turns to avoid false ONCOMING
                float dampFactor = std::max(0.3f, 1.0f - (yawRate - config_.turnYawRateThreshold) * 5.0f);
                smoothedClosingSpeed *= dampFactor;
                closingSpeedKmh = utils::msToKmh(smoothedClosingSpeed);
            }
        }

        // ========== BRAKE DETECTION (from OXTS forward acceleration) ==========
        bool egoIsBraking = false;
        if (currentOxts_.valid && currentOxts_.ax < config_.hardBrakeThreshold) {
            egoIsBraking = true;
        }

        // ========== STICKY LOCK (Stationary Detection with Hysteresis) ==========
        // With ego speed from OXTS now 100% accurate, we detect stationary vehicles
        // by comparing: is V_closing ≈ V_ego? (allowing for bbox jitter)
        bool isCurrentlyStationary = false;
        
        if (egoSpeedKmh > 3.0f && closingSpeedKmh > 0.0f) {
            float speedDiff = std::abs(closingSpeedKmh - egoSpeedKmh);
            // Allowed error: ±(ego_speed * ratio) + 5 km/h baseline noise from bbox jitter
            float allowedError = (egoSpeedKmh * config_.stationaryMatchRatio) + 5.0f;
            
            if (speedDiff < allowedError) {
                isCurrentlyStationary = true;
            }
        }

        // Hysteresis logic: accumulate evidence before committing to state change
        if (isCurrentlyStationary) {
            history.stationaryFrames++;
            history.movingFrames = std::max(0, history.movingFrames - 2);  // Decay moving counter
        } else {
            history.movingFrames++;
            history.stationaryFrames = std::max(0, history.stationaryFrames - 1);  // Decay stationary counter
        }

        // Clamp counters to prevent overflow
        history.stationaryFrames = std::min(history.stationaryFrames, 15);
        history.movingFrames = std::min(history.movingFrames, 15);

        // Lock / Unlock state machine
        if (!history.lockedStationary && history.stationaryFrames >= config_.stickyLockThreshold) {
            history.lockedStationary = true;
        }
        if (history.lockedStationary && history.movingFrames >= config_.stickyUnlockThreshold) {
            history.lockedStationary = false;
        }

        // ========== STATE CLASSIFICATION ==========
        VehicleState state = VehicleState::UNKNOWN;
        float estimatedTargetKmh = 0.0f;
        
        // If locked to stationary, force that state
        if (history.lockedStationary) {
            state = VehicleState::STATIONARY;
            estimatedTargetKmh = 0.0f;
            closingSpeedKmh = egoSpeedKmh;  // Lock V_closing = V_ego for clean TTC
            smoothedClosingSpeed = utils::kmhToMs(egoSpeedKmh);
        }
        else if (egoSpeedKmh > 5.0f && closingSpeedKmh > egoSpeedKmh + config_.oncomingThreshold) {
            // Oncoming vehicle
            state = VehicleState::ONCOMING;
            estimatedTargetKmh = closingSpeedKmh - egoSpeedKmh;
        } 
        else if (closingSpeedKmh > utils::msToKmh(config_.minClosingSpeedMs)) {
            // Same direction, ahead is slower
            state = VehicleState::SAME_DIRECTION;
            estimatedTargetKmh = std::max(0.0f, egoSpeedKmh - closingSpeedKmh);
        } 
        else {
            // Same direction, ahead is faster or same speed
            state = VehicleState::SAME_DIRECTION;
            estimatedTargetKmh = egoSpeedKmh - closingSpeedKmh;
        }

        // ========== TTC (Time To Collision) ==========
        float ttcSeconds = -1.0f;
        if (smoothedClosingSpeed > config_.minClosingSpeedMs) {
            ttcSeconds = distInfo.smoothedDistance / smoothedClosingSpeed;
            if (ttcSeconds > 100.0f) ttcSeconds = -1.0f;
        }

        // ========== OUTPUT ==========
        info.closingSpeedMs = smoothedClosingSpeed;
        info.closingSpeedKmh = closingSpeedKmh;
        info.estimatedTargetKmh = estimatedTargetKmh;
        info.vehicleState = state;
        info.ttcSeconds = ttcSeconds;
        info.egoIsBraking = egoIsBraking;
        info.isApproaching = (smoothedClosingSpeed > config_.minClosingSpeedMs) 
                             && (ttcSeconds > 0 && ttcSeconds < config_.ttcThreshold);
        
        // If ego is braking hard AND approaching target → escalate urgency
        if (egoIsBraking && info.isApproaching && ttcSeconds > 0 && ttcSeconds < config_.ttcThreshold * 1.5f) {
            info.isApproaching = true;  // Widen approach window during hard braking
        }

        info.valid = true;
        newSpeeds[trackId] = info;
    }

    // Cleanup stale tracks
    for (auto it = trackHistory_.begin(); it != trackHistory_.end();) {
        if (distances.find(it->first) == distances.end()) {
            it = trackHistory_.erase(it);
        } else {
            ++it;
        }
    }

    trackSpeeds_ = newSpeeds;
    return trackSpeeds_;
}

// ==============================================================================
// Timestamp-based Linear Regression
// ==============================================================================
// X = real time (seconds), Y = distance (meters)
// slope = m/s directly, no FPS needed
// Closing speed = -slope (positive = approaching)
// ==============================================================================
float SpeedEstimator::computeRegressionSpeed(
    const std::deque<float>& distances,
    const std::deque<float>& times,
    float& outVariance) const {
    
    int N = static_cast<int>(distances.size());
    if (N < 3 || static_cast<int>(times.size()) < N) {
        outVariance = 0.0f;
        return 0.0f;
    }

    float t0 = times[0];  // Use first timestamp as origin (avoid float precision loss)

    float sumT = 0.0f, sumD = 0.0f, sumTD = 0.0f, sumT2 = 0.0f;
    for (int i = 0; i < N; i++) {
        float t = (times[i] - t0) / 1000.0f;  // ms → seconds
        float d = distances[i];
        sumT += t;
        sumD += d;
        sumTD += t * d;
        sumT2 += t * t;
    }

    float denom = N * sumT2 - sumT * sumT;
    if (std::abs(denom) < 1e-6f) { outVariance = 0.0f; return 0.0f; }

    float slope = (N * sumTD - sumT * sumD) / denom;
    float intercept = (sumD - slope * sumT) / N;

    // Compute residual variance
    float residualSum = 0.0f;
    for (int i = 0; i < N; i++) {
        float t = (times[i] - t0) / 1000.0f;
        float predicted = intercept + slope * t;
        float residual = distances[i] - predicted;
        residualSum += residual * residual;
    }
    outVariance = residualSum / N;

    // Closing speed = -slope (distance decreasing = approaching = positive)
    return -slope;
}

// ==============================================================================
// Helpers
// ==============================================================================
float SpeedEstimator::computeMedian(const std::deque<float>& values) const {
    if (values.empty()) return 0.0f;
    std::vector<float> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    int n = static_cast<int>(sorted.size());
    if (n % 2 == 0) return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0f;
    return sorted[n / 2];
}

SpeedInfo SpeedEstimator::getSpeed(int trackId) const {
    auto it = trackSpeeds_.find(trackId);
    if (it != trackSpeeds_.end()) return it->second;
    return SpeedInfo();
}

void SpeedEstimator::setConfig(const SpeedConfig& config) {
    config_ = config;
}

} // namespace fcw

