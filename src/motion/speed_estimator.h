#pragma once
// ==============================================================================
// Speed Estimator - TIMESTAMP-BASED (Industrial Standard)
// ==============================================================================
// - Closing Speed: Linear Regression on (Distance, Timestamp)
//   X-axis = real time (seconds), Y-axis = distance (meters)
//   → slope is directly m/s, NO FPS multiplication needed.
// - Ego Speed: Injected from external source (KITTI OXTS / CAN Bus / GPS)
//   NOT computed from Optical Flow (physically limited with monocular camera).
// - TTC = Distance / |ClosingSpeed|
// - Completely immune to FPS fluctuations.
// ==============================================================================

#include <unordered_map>
#include <vector>
#include <deque>
#include <opencv2/core.hpp>
#include "distance_estimator.h"
#include "kitti_oxts_reader.h"

namespace fcw {

// Target vehicle state (inferred from V_closing vs V_ego)
enum class VehicleState {
    UNKNOWN,          // Insufficient data
    SAME_DIRECTION,   // Same direction (slower or faster)
    STATIONARY,       // Stationary
    ONCOMING          // Oncoming
};

struct SpeedConfig {
    // Timestamp-based regression parameters
    int regressionWindow = 16;           // Max samples for linear regression
    int medianWindow = 5;                // Samples for median filtering
    float maxSpeed = 50.0f;              // Max m/s (180 km/h)
    
    // Closing speed thresholds
    float minClosingSpeedMs = 0.2f;      // Dead zone (0.72 km/h)
    float ttcThreshold = 3.0f;           // TTC < 3s = approaching (warning threshold)
    
    // State classification thresholds
    float oncomingThreshold = 15.0f;     // km/h - V_closing > V_ego + this → oncoming
    
    // Sticky Lock (stationary detection with hysteresis)
    float stationaryMatchRatio = 0.25f;  // |V_closing - V_ego| < V_ego * ratio + 5 → candidate stationary
    int stickyLockThreshold = 5;         // Frames to lock stationary
    int stickyUnlockThreshold = 8;       // Frames to unlock stationary
    
    // Turn suppression (yaw rate from OXTS)
    float turnYawRateThreshold = 0.05f;  // rad/s - above this = ego is turning
    
    // Brake detection (forward acceleration from OXTS)
    float hardBrakeThreshold = -3.0f;    // m/s² - ego hard braking
};

struct SpeedInfo {
    int trackId = -1;
    
    // Closing Speed: rate of distance decrease (positive = approaching)
    float closingSpeedMs = 0.0f;         // m/s
    float closingSpeedKmh = 0.0f;        // km/h
    
    // Estimated target vehicle speed (inferred from V_closing + V_ego)
    float estimatedTargetKmh = 0.0f;     // km/h
    
    // Target vehicle state
    VehicleState vehicleState = VehicleState::UNKNOWN;
    
    // Time To Collision
    float ttcSeconds = -1.0f;            // seconds (-1 = not applicable)
    
    // Ego speed (from external sensor, for reference)
    float egoSpeedKmh = 0.0f;
    
    // Status
    bool isApproaching = false;          // true if TTC < threshold
    bool egoIsBraking = false;           // true if ego ax < hardBrakeThreshold
    bool valid = false;
};

class SpeedEstimator {
public:
    SpeedEstimator();
    explicit SpeedEstimator(const SpeedConfig& config);

    /// Set ego speed from external source (OXTS / CAN Bus / GPS)
    void setEgoSpeed(float egoSpeedKmh) { egoSpeedKmh_ = egoSpeedKmh; }
    
    /// Set full OXTS data for current frame (yaw, acceleration, angular rate)
    void setOxtsData(const OxtsData& data) { currentOxts_ = data; }

    /// Main estimation: compute closing speed + TTC for each tracked vehicle.
    std::unordered_map<int, SpeedInfo> estimate(
        const std::unordered_map<int, DistanceInfo>& distances,
        float timestampMs);

    float getEgoSpeedKmh() const { return egoSpeedKmh_; }
    SpeedInfo getSpeed(int trackId) const;
    void setConfig(const SpeedConfig& config);

private:
    SpeedConfig config_;

    float egoSpeedKmh_ = 0.0f;          // Set externally via setEgoSpeed()
    float lastTimestampMs_ = -1.0f;
    OxtsData currentOxts_;               // Full OXTS data for current frame
    float prevYaw_ = 0.0f;              // Previous frame yaw for delta computation
    bool hasPrevYaw_ = false;

    struct TrackSpeedHistory {
        std::deque<float> distanceSamples;     // Distance (meters)
        std::deque<float> timeSamples;         // Real timestamp (ms)
        std::deque<float> closingSpeedHistory; // Smoothed closing speed history
        float currentDistance = 0.0f;
        int framesSeen = 0;
        
        // Sticky Lock (stationary detection hysteresis)
        int stationaryFrames = 0;    // Frames meeting stationary criteria
        int movingFrames = 0;        // Frames NOT meeting stationary criteria
        bool lockedStationary = false; // Current lock state
    };

    std::unordered_map<int, TrackSpeedHistory> trackHistory_;
    std::unordered_map<int, SpeedInfo> trackSpeeds_;

    // Timestamp-based linear regression → returns m/s directly
    float computeRegressionSpeed(const std::deque<float>& distances,
                                 const std::deque<float>& times,
                                 float& outVariance) const;
    float computeMedian(const std::deque<float>& values) const;
};

} // namespace fcw

