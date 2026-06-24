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
    float criticalTTC = 1.5f;          // TTC threshold for CRITICAL
    float dangerTTC = 3.0f;            // TTC threshold for DANGER
    float cautionTTC = 5.0f;           // TTC threshold for CAUTION

    bool enableSmoothing = true;       // Prevent risk level flickering
    int smoothingWindow = 5;           // History window (frames)

    // Per-level minimum consecutive frames before triggering
    // Faster for urgent levels (imminent collision), slower for early warning
    int minConsecutiveCritical = 2;    // ~0.25s at 8fps – fast response
    int minConsecutiveDanger = 4;      // ~0.50s – reduce noise-induced false alarms
    int minConsecutiveCaution = 3;     // ~0.37s – early warning fires reasonably fast

    // CAUTION distance gate: only warn when actually close
    // DANGER/CRITICAL have no distance cap (TTC alone is sufficient for urgent levels)
    float cautionMaxDistM = 12.0f;     // CAUTION fires only when distance < 12m

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
    int getMinConsecutive(RiskLevel level) const;

    RiskConfig config_;
    std::unordered_map<int, RiskAssessment> risks_;
    std::unordered_map<int, std::vector<RiskLevel>> riskHistory_;
};

} // namespace fcw
