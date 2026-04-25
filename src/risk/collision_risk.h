#pragma once
// ==============================================================================
// Collision Risk Assessment
// ==============================================================================

#include <unordered_map>
#include <vector>
#include "risk_state.h"
#include "ttc_calculator.h"

namespace fcw {

struct RiskConfig {
    float criticalTTC = 1.5f;       // TTC threshold for CRITICAL
    float dangerTTC = 3.0f;         // TTC threshold for DANGER
    float cautionTTC = 5.0f;        // TTC threshold for CAUTION
    bool enableSmoothing = true;    // Prevent risk level flickering
    int smoothingWindow = 5;        // Frames for smoothing
    int minConsecutive = 3;         // Min frames before triggering alert
};

class CollisionRisk {
public:
    CollisionRisk();
    explicit CollisionRisk(const RiskConfig& config);

    /**
     * Assess collision risk for all tracked vehicles.
     * @param ttcResults  TTC information per track
     * @param distances   Distance info per track (for ego-lane filtering)
     * @return Map of trackId → RiskAssessment
     */
    std::unordered_map<int, RiskAssessment> assess(
        const std::unordered_map<int, TTCInfo>& ttcResults,
        const std::unordered_map<int, DistanceInfo>& distances = {});

    /** Get the highest risk level across all tracks */
    RiskAssessment getHighestRisk() const;

    /** Get risk for specific track */
    RiskAssessment getRisk(int trackId) const;

    /** Set config */
    void setConfig(const RiskConfig& config);

private:
    RiskLevel classifyRisk(float ttc) const;

    RiskConfig config_;
    std::unordered_map<int, RiskAssessment> risks_;
    std::unordered_map<int, std::vector<RiskLevel>> riskHistory_;
};

} // namespace fcw
