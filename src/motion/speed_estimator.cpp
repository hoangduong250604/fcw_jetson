// ==============================================================================
// Speed Estimator Implementation
// ==============================================================================
// Two selectable methods (SpeedConfig::useKalmanSpeed):
//
// A. Kalman Filter [D, v_closing]  (default, recommended)
//    Predict: D_{k+1} = D_k - v*dt,  v_{k+1} = v_k (constant velocity model)
//    Process noise Q: Singer model from vehicle acceleration std dev sigma_a
//    Update:  z = D_measured from pinhole bbox-height estimate
//    Warm-start: v_0 estimated from first 2-frame distance difference
//      → converges in 2 frames vs. regression needing 16+ samples for accuracy
//
// B. Weighted Linear Regression  (fallback when useKalmanSpeed=false)
//    w_i = exp(-lambda * age_seconds) — recent samples get higher weight
//    Followed by median filter for additional robustness
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
// Main Estimation
// ==============================================================================
std::unordered_map<int, SpeedInfo> SpeedEstimator::estimate(
    const std::unordered_map<int, DistanceInfo>& distances,
    float timestampMs) {

    std::unordered_map<int, SpeedInfo> newSpeeds;
    lastTimestampMs_ = timestampMs;

    float egoSpeedKmh = egoSpeedKmh_;

    for (const auto& [trackId, distInfo] : distances) {
        if (!distInfo.valid) continue;

        SpeedInfo info;
        info.trackId     = trackId;
        info.egoSpeedKmh = egoSpeedKmh;

        auto& history = trackHistory_[trackId];
        history.framesSeen++;
        history.currentDistance = distInfo.smoothedDistance;

        // Store samples (for regression fallback & warm-start)
        history.distanceSamples.push_back(distInfo.smoothedDistance);
        history.timeSamples.push_back(timestampMs);
        while (static_cast<int>(history.distanceSamples.size()) > config_.regressionWindow) {
            history.distanceSamples.pop_front();
            history.timeSamples.pop_front();
        }

        // ── Speed estimation ────────────────────────────────────────────────────
        float smoothedClosingSpeed = 0.0f;
        float h_px = static_cast<float>(distInfo.bbox.height);

        if (config_.usePixelKalman && h_px > 1.0f) {
            // ── Method A: Pixel-space Kalman [h, dh/dt] ─────────────────────────
            // Measurement noise R = σ_h² = 4 px² (constant — homoscedastic)
            // v_closing = D_smooth * (dh_kalman / h_kalman)
            // TTC_direct = h_kalman / dh_kalman  (H_real-independent)
            KalmanBboxState& kb = history.kalmanBbox;

            if (!kb.initialized) {
                if (history.distanceSamples.size() < 2) {
                    info.valid = true;
                    newSpeeds[trackId] = info;
                    continue;
                }
                // Warm-start dh from D history: dh/dt = (h/D) * v_closing_init
                float D_now  = history.distanceSamples.back();
                float D_prev = history.distanceSamples[history.distanceSamples.size() - 2];
                float t_prev = history.timeSamples[history.timeSamples.size() - 2];
                float dt_init = std::max(0.001f, (timestampMs - t_prev) / 1000.0f);
                float v_init  = (D_prev - D_now) / dt_init;   // + = approaching

                kb.h  = h_px;
                // dh/dt = (h/D) * v_closing   (from D=fy*H/h → dD/dt = -(D/h)*dh/dt)
                kb.dh = utils::clamp((h_px / std::max(D_now, 0.1f)) * v_init,
                                     -300.0f, 300.0f);
                kb.P[0] = config_.kalmanBboxInitVarH;  kb.P[1] = 0.0f;
                kb.P[2] = 0.0f;  kb.P[3] = config_.kalmanBboxInitVarDH;
                kb.lastTimestampMs = timestampMs;
                kb.initialized     = true;

                smoothedClosingSpeed = utils::clamp(v_init, -config_.maxSpeed, config_.maxSpeed);

            } else {
                float dt = (timestampMs - kb.lastTimestampMs) / 1000.0f;
                kb.lastTimestampMs = timestampMs;

                if (dt > 2.0f) {
                    kb.initialized = false;
                    info.valid = true;
                    newSpeeds[trackId] = info;
                    continue;
                }
                if (dt > 0.005f) kalmanBboxPredict(kb, dt);
                kalmanBboxUpdate(kb, h_px);

                // v_closing = D_smooth * dh/h
                float h_sm = std::max(kb.h, 1.0f);
                smoothedClosingSpeed = distInfo.smoothedDistance * kb.dh / h_sm;
                smoothedClosingSpeed = utils::clamp(smoothedClosingSpeed,
                                                    -config_.maxSpeed, config_.maxSpeed);
            }

        } else if (config_.useKalmanSpeed) {
            // ── Method B: D-space Kalman [D, v_closing] ─────────────────────────
            KalmanSpeedState& ks = history.kalman;

            if (!ks.initialized) {
                if (history.distanceSamples.size() < 2) {
                    info.valid = true;
                    newSpeeds[trackId] = info;
                    continue;
                }
                float D_now  = history.distanceSamples.back();
                float D_prev = history.distanceSamples[history.distanceSamples.size() - 2];
                float t_prev = history.timeSamples[history.timeSamples.size() - 2];
                float dt_init = std::max(0.001f, (timestampMs - t_prev) / 1000.0f);

                ks.D = D_now;
                ks.v = utils::clamp((D_prev - D_now) / dt_init, -config_.maxSpeed, config_.maxSpeed);
                ks.P[0] = config_.kalmanInitVarD; ks.P[1] = 0.0f;
                ks.P[2] = 0.0f;                   ks.P[3] = config_.kalmanInitVarV;
                ks.lastTimestampMs = timestampMs;
                ks.initialized     = true;
                smoothedClosingSpeed = ks.v;

            } else {
                float dt = (timestampMs - ks.lastTimestampMs) / 1000.0f;
                ks.lastTimestampMs = timestampMs;

                if (dt > 2.0f) {
                    ks.initialized = false;
                    info.valid = true;
                    newSpeeds[trackId] = info;
                    continue;
                }
                if (dt > 0.005f) kalmanPredict(ks, dt);
                kalmanUpdate(ks, distInfo.smoothedDistance, h_px);
                smoothedClosingSpeed = ks.v;
            }

        } else {
            // ── Method C: Weighted linear regression + median filter ─────────────
            if (static_cast<int>(history.distanceSamples.size()) < 3) {
                info.valid = true;
                newSpeeds[trackId] = info;
                continue;
            }
            float regressionVariance = 0.0f;
            float rawSpeed = computeRegressionSpeed(
                history.distanceSamples, history.timeSamples, regressionVariance);
            rawSpeed = utils::clamp(rawSpeed, -config_.maxSpeed, config_.maxSpeed);

            history.closingSpeedHistory.push_back(rawSpeed);
            while (static_cast<int>(history.closingSpeedHistory.size()) > config_.medianWindow) {
                history.closingSpeedHistory.pop_front();
            }
            smoothedClosingSpeed = computeMedian(history.closingSpeedHistory);
        }

        smoothedClosingSpeed = utils::clamp(smoothedClosingSpeed, -config_.maxSpeed, config_.maxSpeed);
        float closingSpeedKmh = utils::msToKmh(smoothedClosingSpeed);

        // ── Turn suppression (OXTS yaw rate) ────────────────────────────────────
        if (currentOxts_.valid) {
            float yawRate = std::abs(currentOxts_.wz);
            if (yawRate > config_.turnYawRateThreshold) {
                float dampFactor = std::max(0.3f, 1.0f - (yawRate - config_.turnYawRateThreshold) * 5.0f);
                smoothedClosingSpeed *= dampFactor;
                closingSpeedKmh = utils::msToKmh(smoothedClosingSpeed);
            }
        }

        // ── Brake detection (OXTS forward acceleration) ──────────────────────────
        bool egoIsBraking = false;
        if (currentOxts_.valid && currentOxts_.ax < config_.hardBrakeThreshold) {
            egoIsBraking = true;
        }

        // ── Sticky Lock: stationary detection with hysteresis ────────────────────
        bool isCurrentlyStationary = false;
        if (egoSpeedKmh > 3.0f && closingSpeedKmh > 0.0f) {
            float speedDiff    = std::abs(closingSpeedKmh - egoSpeedKmh);
            float allowedError = (egoSpeedKmh * config_.stationaryMatchRatio) + 5.0f;
            if (speedDiff < allowedError) isCurrentlyStationary = true;
        }

        if (isCurrentlyStationary) {
            history.stationaryFrames++;
            history.movingFrames = std::max(0, history.movingFrames - 2);
        } else {
            history.movingFrames++;
            history.stationaryFrames = std::max(0, history.stationaryFrames - 1);
        }
        history.stationaryFrames = std::min(history.stationaryFrames, 15);
        history.movingFrames     = std::min(history.movingFrames, 15);

        if (!history.lockedStationary && history.stationaryFrames >= config_.stickyLockThreshold)
            history.lockedStationary = true;
        if ( history.lockedStationary && history.movingFrames  >= config_.stickyUnlockThreshold)
            history.lockedStationary = false;

        // ── State classification ─────────────────────────────────────────────────
        VehicleState state            = VehicleState::UNKNOWN;
        float        estimatedTargetKmh = 0.0f;

        if (history.lockedStationary) {
            state                = VehicleState::STATIONARY;
            estimatedTargetKmh   = 0.0f;
            closingSpeedKmh      = egoSpeedKmh;
            smoothedClosingSpeed = utils::kmhToMs(egoSpeedKmh);
        } else if (egoSpeedKmh > 5.0f && closingSpeedKmh > egoSpeedKmh + config_.oncomingThreshold) {
            state              = VehicleState::ONCOMING;
            estimatedTargetKmh = closingSpeedKmh - egoSpeedKmh;
        } else if (closingSpeedKmh > utils::msToKmh(config_.minClosingSpeedMs)) {
            state              = VehicleState::SAME_DIRECTION;
            estimatedTargetKmh = std::max(0.0f, egoSpeedKmh - closingSpeedKmh);
        } else {
            state              = VehicleState::SAME_DIRECTION;
            estimatedTargetKmh = egoSpeedKmh - closingSpeedKmh;
        }

        // ── TTC ──────────────────────────────────────────────────────────────────
        float ttcSeconds = -1.0f;
        if (smoothedClosingSpeed > config_.minClosingSpeedMs) {
            ttcSeconds = distInfo.smoothedDistance / smoothedClosingSpeed;
            if (ttcSeconds > 100.0f) ttcSeconds = -1.0f;
        }

        // ── Output ───────────────────────────────────────────────────────────────
        info.closingSpeedMs     = smoothedClosingSpeed;
        info.closingSpeedKmh    = closingSpeedKmh;
        info.estimatedTargetKmh = estimatedTargetKmh;
        info.vehicleState       = state;
        info.ttcSeconds         = ttcSeconds;
        info.egoIsBraking       = egoIsBraking;
        info.isApproaching      = (smoothedClosingSpeed > config_.minClosingSpeedMs)
                                  && (ttcSeconds > 0 && ttcSeconds < config_.ttcThreshold);

        // Hard braking: extend detection window to 1.5x (9s)
        if (egoIsBraking && smoothedClosingSpeed > config_.minClosingSpeedMs
                         && ttcSeconds > 0 && ttcSeconds < config_.ttcThreshold * 1.5f) {
            info.isApproaching = true;
        }

        info.valid = true;
        newSpeeds[trackId] = info;
    }

    // Cleanup stale tracks
    for (auto it = trackHistory_.begin(); it != trackHistory_.end();) {
        if (distances.find(it->first) == distances.end())
            it = trackHistory_.erase(it);
        else
            ++it;
    }

    trackSpeeds_ = newSpeeds;
    return trackSpeeds_;
}

// ==============================================================================
// Method A — Pixel-space Kalman: Predict
// ==============================================================================
// State: x = [h_px, dh/dt]   F = [[1, dt], [0, 1]]
//   h_new  = h + dh*dt
//   dh_new = dh   (constant velocity; acceleration is process noise)
// Q = sigmaA² * [[dt⁴/4, dt³/2], [dt³/2, dt²]]
// ==============================================================================
void SpeedEstimator::kalmanBboxPredict(KalmanBboxState& s, float dt) const {
    // Predict state
    s.h = s.h + s.dh * dt;
    // s.dh unchanged

    // P_new = F*P*F^T + Q
    const float sa  = config_.kalmanBboxSigmaA;
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;

    // F*P  (F = [[1,dt],[0,1]])
    float fp00 = s.P[0] + dt * s.P[2];
    float fp01 = s.P[1] + dt * s.P[3];
    float fp10 = s.P[2];
    float fp11 = s.P[3];

    // (F*P)*F^T + Q  (F^T = [[1,0],[dt,1]])
    float q00 = sa * sa * dt4 / 4.0f;
    float q01 = sa * sa * dt3 / 2.0f;
    float q11 = sa * sa * dt2;

    s.P[0] = fp00 + dt * fp01 + q00;
    s.P[1] = fp01             + q01;
    s.P[2] = fp10 + dt * fp11 + q01;
    s.P[3] = fp11             + q11;
}

// ==============================================================================
// Method A — Pixel-space Kalman: Update
// ==============================================================================
// Measurement: z = h_meas,  H = [1, 0],  R = σ_h² (constant, px²)
//   S = P[0] + R
//   K = [P[0], P[2]]^T / S
//   x += K * (z - h_predicted)
//   P = (I - K*H) * P
// ==============================================================================
void SpeedEstimator::kalmanBboxUpdate(KalmanBboxState& s, float hMeas) const {
    float R = config_.kalmanBboxMeasNoise;
    float S = s.P[0] + R;
    if (S < 1e-6f) S = 1e-6f;

    float K0 = s.P[0] / S;
    float K1 = s.P[2] / S;

    float innov = hMeas - s.h;
    s.h  += K0 * innov;
    s.dh += K1 * innov;
    s.dh  = utils::clamp(s.dh, -500.0f, 500.0f);

    // P = (I - K*H) * P
    float p00 = s.P[0], p01 = s.P[1];
    float p10 = s.P[2], p11 = s.P[3];
    s.P[0] = (1.0f - K0) * p00;
    s.P[1] = (1.0f - K0) * p01;
    s.P[2] = p10 - K1 * p00;
    s.P[3] = p11 - K1 * p01;

    if (s.P[0] < 1e-3f) s.P[0] = 1e-3f;
    if (s.P[3] < 1e-3f) s.P[3] = 1e-3f;
}

// ==============================================================================
// Method B — D-space Kalman: Predict
// ==============================================================================
// State transition: F = [[1, -dt], [0, 1]]
//   D_new = D - v*dt
//   v_new = v  (constant velocity; acceleration is process noise)
//
// Process noise (Singer constant-acceleration model):
//   Q = sigma_a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
// ==============================================================================
void SpeedEstimator::kalmanPredict(KalmanSpeedState& s, float dt) const {
    const float sa  = config_.kalmanSigmaA;
    const float dt2 = dt * dt;
    const float dt3 = dt2 * dt;
    const float dt4 = dt2 * dt2;

    // Predict state
    s.D = s.D - s.v * dt;
    // s.v unchanged (constant velocity model)

    // Expand F*P:
    //   F = [[1,-dt],[0,1]]
    //   (F*P)[0][*] = P[0][*] - dt*P[2][*]
    //   (F*P)[1][*] = P[2][*]  (second row of F is [0,1])
    float fp00 = s.P[0] - dt * s.P[2];
    float fp01 = s.P[1] - dt * s.P[3];
    float fp10 = s.P[2];
    float fp11 = s.P[3];

    // Expand (F*P)*F^T + Q:
    //   F^T = [[1,0],[-dt,1]]
    //   (FP*F^T)[0][0] = fp00*1 + fp01*(-dt) = fp00 - dt*fp01
    //   (FP*F^T)[0][1] = fp00*0 + fp01*1     = fp01
    //   (FP*F^T)[1][0] = fp10*1 + fp11*(-dt) = fp10 - dt*fp11
    //   (FP*F^T)[1][1] = fp10*0 + fp11*1     = fp11
    float q00 = sa * sa * dt4 / 4.0f;
    float q01 = sa * sa * dt3 / 2.0f;
    float q11 = sa * sa * dt2;

    s.P[0] = fp00 - dt * fp01 + q00;
    s.P[1] = fp01             + q01;
    s.P[2] = fp10 - dt * fp11 + q01;   // symmetry: P[1] == P[2]
    s.P[3] = fp11             + q11;
}

// ==============================================================================
// Method B — D-space Kalman: Update
// ==============================================================================
// Measurement: z = D_measured,  H = [1, 0]
//   R = kalmanMeasNoise (constant; ≈ (0.8m)² = 0.64 m²)
//   S = P[0] + R,  K = P*H^T / S
//   x += K*(z - D_pred),  P = (I - K*H)*P
// ==============================================================================
float SpeedEstimator::kalmanUpdate(KalmanSpeedState& s, float measuredDist, float /*h_px*/) const {
    float R = config_.kalmanMeasNoise;
    float S  = s.P[0] + R;
    if (S < 1e-6f) S = 1e-6f;

    float K0 = s.P[0] / S;   // Kalman gain for D state
    float K1 = s.P[2] / S;   // Kalman gain for v state

    float innov = measuredDist - s.D;
    s.D += K0 * innov;
    s.v += K1 * innov;

    // P = (I - K*H) * P  where K*H = [[K0,0],[K1,0]]
    float p00 = s.P[0], p01 = s.P[1];
    float p10 = s.P[2], p11 = s.P[3];
    s.P[0] = (1.0f - K0) * p00;
    s.P[1] = (1.0f - K0) * p01;
    s.P[2] = p10 - K1 * p00;
    s.P[3] = p11 - K1 * p01;

    // Guard against numerical collapse of diagonal variances
    if (s.P[0] < 1e-4f) s.P[0] = 1e-4f;
    if (s.P[3] < 1e-4f) s.P[3] = 1e-4f;

    return s.v;
}

// ==============================================================================
// Method C — Weighted Linear Regression
// ==============================================================================
// w_i = exp(-lambda * age_seconds)   (recent samples get higher weight)
// Weighted least-squares on (t, d) samples → slope = dd/dt
// Closing speed = -slope   (positive = distance decreasing = approaching)
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

    float t0     = times[0];
    float t_last = times[N - 1];
    float lambda = config_.regressionLambda;

    float sumW = 0.0f, sumWT = 0.0f, sumWD = 0.0f, sumWTD = 0.0f, sumWT2 = 0.0f;
    for (int i = 0; i < N; i++) {
        float t   = (times[i] - t0)     / 1000.0f;   // ms -> s, origin at t0
        float d   = distances[i];
        float age = (t_last - times[i]) / 1000.0f;   // age of sample in seconds
        float w   = std::exp(-lambda * age);

        sumW   += w;
        sumWT  += w * t;
        sumWD  += w * d;
        sumWTD += w * t * d;
        sumWT2 += w * t * t;
    }

    float denom = sumW * sumWT2 - sumWT * sumWT;
    if (std::abs(denom) < 1e-6f) { outVariance = 0.0f; return 0.0f; }

    float slope     = (sumW * sumWTD - sumWT * sumWD) / denom;
    float intercept = (sumWD - slope * sumWT) / sumW;

    // Weighted residual variance
    float residualSum = 0.0f;
    for (int i = 0; i < N; i++) {
        float t   = (times[i] - t0)     / 1000.0f;
        float age = (t_last - times[i]) / 1000.0f;
        float w   = std::exp(-lambda * age);
        float res = distances[i] - (intercept + slope * t);
        residualSum += w * res * res;
    }
    outVariance = (sumW > 0.0f) ? residualSum / sumW : 0.0f;

    return -slope;  // positive = approaching
}

// ==============================================================================
// Helpers
// ==============================================================================
float SpeedEstimator::computeMedian(const std::deque<float>& values) const {
    if (values.empty()) return 0.0f;
    std::vector<float> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    int n = static_cast<int>(sorted.size());
    return (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0f : sorted[n/2];
}

SpeedInfo SpeedEstimator::getSpeed(int trackId) const {
    auto it = trackSpeeds_.find(trackId);
    return (it != trackSpeeds_.end()) ? it->second : SpeedInfo();
}

void SpeedEstimator::setConfig(const SpeedConfig& config) {
    config_ = config;
}

} // namespace fcw
