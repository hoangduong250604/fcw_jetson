#pragma once
// ==============================================================================
// Risk State - Collision risk level definitions
// ==============================================================================

#include <string>

namespace fcw {

/**
 * Risk levels for collision warning.
 * Ordered by severity (higher enum value = higher risk).
 */
enum class RiskLevel {
    SAFE = 0,       // No collision risk
    CAUTION = 1,    // Monitor - vehicle approaching
    DANGER = 2,     // High risk - prepare to brake
    CRITICAL = 3    // Imminent collision - brake NOW
};

inline std::string riskLevelToString(RiskLevel level) {
    switch (level) {
        case RiskLevel::SAFE:     return "SAFE";
        case RiskLevel::CAUTION:  return "CAUTION";
        case RiskLevel::DANGER:   return "DANGER";
        case RiskLevel::CRITICAL: return "CRITICAL";
        default:                  return "UNKNOWN";
    }
}

/**
 * Per-track risk assessment with all relevant data.
 */
struct RiskAssessment {
    int trackId = -1;
    RiskLevel level = RiskLevel::SAFE;
    float ttc = -1.0f;              // Time to collision (seconds)
    float distance = 0.0f;          // Distance (meters)
    float relativeSpeed = 0.0f;     // Relative approaching speed (m/s)
    bool isLeadVehicle = false;     // Whether this is the lead vehicle in lane
    bool inEgoPath = true;          // Whether vehicle is in ego-lane corridor
    int consecutiveFrames = 0;      // Consecutive frames at this risk level
};

} // namespace fcw
