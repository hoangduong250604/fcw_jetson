#pragma once
// ==============================================================================
// TTC Calculator - Time-to-Collision Estimation
// ==============================================================================
// CORE MODULE: The central safety computation of the FCW system.
//
// Time-to-Collision (TTC) is defined as the time remaining before a
// collision occurs, assuming constant relative velocity:
//
//   TTC = D / v_relative
//
// Where:
//   D = distance to the lead vehicle (meters)
//   v_relative = relative approaching speed (m/s, positive = approaching)
//
// TTC is only meaningful when:
//   - v_relative > 0 (vehicle is approaching)
//   - D > 0 (valid distance measurement)
//
// Alternative TTC methods implemented:
//   1. Distance-based:  TTC = D / v_rel (primary method)
//   2. Scale-based:     TTC from bbox scale change rate
//   3. Combined:        Weighted average for robustness
//
// TTC values are smoothed per-track using EMA to prevent alert flickering.
// ==============================================================================

#include <unordered_map>
#include <vector>
#include "distance_estimator.h"
#include "speed_estimator.h"
#include "track.h"

namespace fcw {

struct TTCConfig {
    float maxTTC = 20.0f;           // Max TTC to report (seconds)
    float minValidTTC = 0.5f;       // Below this = imminent collision
    float smoothingAlpha = 0.3f;    // EMA smoothing
    bool useScaleMethod = true;     // Also compute scale-based TTC
    float scaleWeight = 0.3f;       // Weight for scale-based TTC in combined
};

/**
 * Per-track TTC estimation.
 */
struct TTCInfo {
    int trackId = -1;
    float distance = 0.0f;              // Distance to vehicle (m)
    float relativeSpeed = 0.0f;         // Approaching speed (m/s)

    float ttcDistance = -1.0f;           // TTC from distance method (seconds)
    float ttcScale = -1.0f;             // TTC from scale method (seconds)
    float ttcCombined = -1.0f;          // Combined TTC (seconds)
    float ttcSmoothed = -1.0f;          // Final smoothed TTC (seconds)

    bool isApproaching = false;          // Vehicle is getting closer
    bool valid = false;                  // TTC computation is valid
};

class TTCCalculator {
public:
    TTCCalculator();
    explicit TTCCalculator(const TTCConfig& config);

    /**
     * Calculate TTC for all tracked vehicles.
     *
     * Combines distance information and relative speed to estimate
     * how many seconds until collision.
     *
     * @param tracks     Active tracked objects
     * @param distances  Distance estimates per track
     * @param speeds     Speed estimates per track
     * @return Map of trackId → TTCInfo
     */
    std::unordered_map<int, TTCInfo> calculate(
        const std::vector<Track*>& tracks,
        const std::unordered_map<int, DistanceInfo>& distances,
        const std::unordered_map<int, SpeedInfo>& speeds);

    /** Get TTC for specific track */
    TTCInfo getTTC(int trackId) const;

    /** Get the most critical (lowest) TTC among all tracks */
    TTCInfo getMostCriticalTTC() const;

    /** Set config */
    void setConfig(const TTCConfig& config);

private:
    /**
     * Primary method: TTC = D / v_relative
     * Uses distance and relative speed from tracker.
     */
    float computeTTCFromDistance(float distance, float relativeSpeed) const;

    /**
     * Secondary method: TTC from bounding box scale change.
     * 
     * If a vehicle is approaching, its bbox gets larger.
     * TTC_scale = s / (ds/dt)
     * where s = bbox area, ds/dt = rate of area change.
     *
     * This is independent of camera calibration.
     */
    float computeTTCFromScale(const Track* track) const;

    /**
     * Combine distance-based and scale-based TTC.
     */
    float combineTTC(float ttcDist, float ttcScale) const;

    TTCConfig config_;
    std::unordered_map<int, TTCInfo> trackTTCs_;
    std::unordered_map<int, float> prevSmoothedTTC_;
};

} // namespace fcw
