#pragma once
// ==============================================================================
// Speed Estimator - TIMESTAMP-BASED
// ==============================================================================
// Speed estimation method (selectable via SpeedConfig::useKalmanSpeed):
//
// Method A — Kalman Filter [D, v_closing]  (default, recommended)
//   State:    x = [distance, closing_speed]
//   Dynamics: D_{k+1} = D_k − v·Δt,  v_{k+1} = v_k + ε_a
//   Process noise Q: Singer constant-acceleration model (σ_a from config)
//   Measurement: z = D_bbox (pinhole estimate), H = [1, 0]
//   Warm-start: v_0 estimated from first 2-frame distance difference
//   Benefits: FPS-agnostic (uses real Δt), smooth velocity, models dynamics
//
// Method B — Weighted Linear Regression  (fallback, useKalmanSpeed=false)
//   Weight: w_i = exp(−λ · age_seconds) — recent samples weighted higher
//   Followed by median filter for additional robustness
//
// Ego Speed: Injected externally (KITTI OXTS / CAN Bus / GPS / fixed 50 km/h)
// TTC = Distance / |ClosingSpeed|
// ==============================================================================

#include <unordered_map>
#include <vector>
#include <deque>
#include <opencv2/core.hpp>
#include "distance_estimator.h"
#include "kitti_oxts_reader.h"

namespace fcw {

enum class VehicleState {
    UNKNOWN,
    SAME_DIRECTION,
    STATIONARY,
    ONCOMING
};

// ---------------------------------------------------------------------------
// Method A — Pixel-space Kalman: x = [h_px, dh/dt]
// h_px  = bbox height in pixels, dh/dt in px/s (positive = growing = approaching)
// Measurement noise R = σ_h² ≈ (2px)² = 4 px²  — constant everywhere (homoscedastic)
// v_closing derived as: v = D_smooth * (dh/h)   (no H_real needed for v itself)
// TTC direct:  TTC = h / dh  (purely from pixel divergence, H_real-independent)
// ---------------------------------------------------------------------------
struct KalmanBboxState {
    float h  = 0.0f;                              // Estimated bbox height (px)
    float dh = 0.0f;                              // dh/dt (px/s), + = growing = approaching
    float P[4] = {100.0f, 0.0f, 0.0f, 400.0f};  // 2×2 covariance, row-major
    bool  initialized     = false;
    float lastTimestampMs = -1.0f;
};

// ---------------------------------------------------------------------------
// Method B — D-space Kalman: x = [distance (m), closing_speed (m/s)]
// Covariance P stored row-major: [P00, P01, P10, P11]
// ---------------------------------------------------------------------------
struct KalmanSpeedState {
    float D  = 0.0f;                           // Estimated distance (m)
    float v  = 0.0f;                           // Estimated closing speed (m/s, + = approaching)
    float P[4] = {4.0f, 0.0f, 0.0f, 25.0f};  // 2x2 covariance, row-major
    bool  initialized     = false;
    float lastTimestampMs = -1.0f;
};

struct SpeedConfig {
    // Sample buffer (kept for regression fallback & debug)
    int   regressionWindow      = 16;         // Max samples in history deque
    int   medianWindow          = 5;          // Median filter window (regression only)
    float maxSpeed              = 50.0f;      // Hard clamp m/s (180 km/h)

    // Closing speed thresholds
    float minClosingSpeedMs     = 0.2f;       // Dead zone (< 0.72 km/h → not approaching)
    float ttcThreshold          = 6.0f;       // TTC < 6s = approaching window

    // State classification
    float oncomingThreshold     = 15.0f;      // km/h: V_closing > V_ego + this → ONCOMING

    // Sticky Lock (stationary detection hysteresis)
    float stationaryMatchRatio  = 0.25f;
    int   stickyLockThreshold   = 5;
    int   stickyUnlockThreshold = 8;

    // OXTS-based helpers
    float turnYawRateThreshold  = 0.05f;      // rad/s
    float hardBrakeThreshold    = -3.0f;      // m/s²

    // ── Speed estimation method ──────────────────────────────────────────────
    bool  usePixelKalman        = false;      // Method A: pixel-space Kalman [h, dh/dt] (experimental)
    bool  useKalmanSpeed        = true;       // Method B: D-space Kalman [D, v] (default — best)
                                              // false for both → Method C: weighted regression

    // Method A — Pixel-space Kalman [h, dh/dt]  (best: homoscedastic noise)
    float kalmanBboxSigmaA      = 10.0f;     // Process noise: bbox height accel std (px/s²)
    float kalmanBboxMeasNoise   = 4.0f;      // R = σ_h² ≈ (2px)² — constant everywhere
    float kalmanBboxInitVarH    = 100.0f;    // Initial P00 (px²)
    float kalmanBboxInitVarDH   = 400.0f;    // Initial P11 ((px/s)²)

    // Method B — D-space Kalman [D, v]  (previous default)
    float kalmanSigmaA          = 2.0f;       // Process noise: accel std dev (m/s²)
    float kalmanMeasNoise       = 0.64f;      // Measurement variance R (m²) ~= (0.8m)^2
    float kalmanInitVarD        = 4.0f;       // Initial distance variance P00 (m²)
    float kalmanInitVarV        = 25.0f;      // Initial velocity variance P11 ((m/s)^2)

    // Method C — Weighted regression  (fallback)
    float regressionLambda      = 0.8f;       // Exp decay rate (1/s); higher = more weight on recent
};

struct SpeedInfo {
    int          trackId             = -1;
    float        closingSpeedMs      = 0.0f;
    float        closingSpeedKmh     = 0.0f;
    float        estimatedTargetKmh  = 0.0f;
    VehicleState vehicleState        = VehicleState::UNKNOWN;
    float        ttcSeconds          = -1.0f;
    float        egoSpeedKmh         = 0.0f;
    bool         isApproaching       = false;
    bool         egoIsBraking        = false;
    bool         valid               = false;
};

class SpeedEstimator {
public:
    SpeedEstimator();
    explicit SpeedEstimator(const SpeedConfig& config);

    void setEgoSpeed(float egoSpeedKmh)    { egoSpeedKmh_ = egoSpeedKmh; }
    void setOxtsData(const OxtsData& data) { currentOxts_ = data; }

    std::unordered_map<int, SpeedInfo> estimate(
        const std::unordered_map<int, DistanceInfo>& distances,
        float timestampMs);

    float     getEgoSpeedKmh() const { return egoSpeedKmh_; }
    SpeedInfo getSpeed(int trackId) const;
    void      setConfig(const SpeedConfig& config);

private:
    SpeedConfig config_;
    float       egoSpeedKmh_     = 0.0f;
    float       lastTimestampMs_ = -1.0f;
    OxtsData    currentOxts_;
    float       prevYaw_         = 0.0f;
    bool        hasPrevYaw_      = false;

    struct TrackSpeedHistory {
        std::deque<float> distanceSamples;      // Smoothed distance history (m)
        std::deque<float> timeSamples;          // Timestamps (ms)
        std::deque<float> closingSpeedHistory;  // For median filter (regression only)
        float currentDistance  = 0.0f;
        int   framesSeen       = 0;
        int   stationaryFrames = 0;
        int   movingFrames     = 0;
        bool  lockedStationary = false;

        KalmanBboxState  kalmanBbox;           // Method A: pixel-space Kalman [h, dh/dt]
        KalmanSpeedState kalman;               // Method B: D-space Kalman [D, v_closing]
    };

    std::unordered_map<int, TrackSpeedHistory> trackHistory_;
    std::unordered_map<int, SpeedInfo>         trackSpeeds_;

    // Method A: Pixel-space Kalman [h, dh/dt]
    void  kalmanBboxPredict(KalmanBboxState& s, float dt) const;
    void  kalmanBboxUpdate(KalmanBboxState& s, float hMeas) const;

    // Method B: D-space Kalman [D, v] with adaptive R(D) for heteroscedastic noise
    void  kalmanPredict(KalmanSpeedState& s, float dt) const;
    float kalmanUpdate(KalmanSpeedState& s, float measuredDist, float h_px) const;

    // Method C: Weighted linear regression
    float computeRegressionSpeed(const std::deque<float>& distances,
                                 const std::deque<float>& times,
                                 float& outVariance) const;
    float computeMedian(const std::deque<float>& values) const;
};

} // namespace fcw
