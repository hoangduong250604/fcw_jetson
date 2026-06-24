// ==============================================================================
// Collision Risk Implementation
// ==============================================================================

#include "collision_risk.h"
#include "logger.h"

#include <algorithm>

namespace fcw {

CollisionRisk::CollisionRisk() {}
CollisionRisk::CollisionRisk(const RiskConfig& config) : config_(config) {}

std::unordered_map<int, RiskAssessment> CollisionRisk::assess(
    const std::unordered_map<int, TTCInfo>& ttcResults,
    const std::unordered_map<int, DistanceInfo>& distances) {

    std::unordered_map<int, RiskAssessment> newRisks;

    for (const auto& [trackId, ttcInfo] : ttcResults) {
        RiskAssessment ra;
        ra.trackId = trackId;
        ra.ttc = ttcInfo.ttcSmoothed;
        ra.distance = ttcInfo.distance;
        ra.relativeSpeed = ttcInfo.relativeSpeed;

        // Check if vehicle is in ego-lane corridor
        bool isEdgeTruncated = false;
        auto distIt = distances.find(trackId);
        if (distIt != distances.end()) {
            ra.inEgoPath = distIt->second.inEgoPath;
            isEdgeTruncated = distIt->second.isEdgeTruncated;
        }

        // === CLASSIFY RISK FROM TTC ===
        if (ttcInfo.valid && ttcInfo.isApproaching) {
            ra.level = classifyRisk(ttcInfo.ttcSmoothed);

            // === ONCOMING VEHICLE SUPPRESSION ===
            // Oncoming in opposite lane: SAFE. Oncoming head-on (in ego path): cap DANGER.
            if (ttcInfo.vehicleState == VehicleState::ONCOMING) {
                if (!ra.inEgoPath) {
                    ra.level = RiskLevel::SAFE;
                } else if (ra.level > RiskLevel::DANGER) {
                    ra.level = RiskLevel::DANGER;
                }
            }

            // === EDGE TRUNCATION SUPPRESSION ===
            if (isEdgeTruncated) {
                ra.level = RiskLevel::SAFE;
            }

            // === EGO-LANE ONLY ===
            // FCW only warns for vehicles inside our lane corridor.
            // Vehicles in adjacent lanes, opposite lane, or roadside are SAFE.
            // This prevents false alarms for: parked cars, adjacent traffic,
            // crossing vehicles at intersections, and vehicles passing on sides.
            if (!ra.inEgoPath) {
                ra.level = RiskLevel::SAFE;
            }

            // === CAUTION DISTANCE GATE ===
            // CAUTION at long range (> 12m) means TTC < 5s only because of slow
            // relative closing speed — not an urgent situation in typical urban driving.
            // DANGER and CRITICAL are NOT gated: high closing speed at any distance is urgent.
            if (ra.level == RiskLevel::CAUTION && ra.distance > config_.cautionMaxDistM) {
                ra.level = RiskLevel::SAFE;
            }
        } else {
            ra.level = RiskLevel::SAFE;
        }
        // NOTE: No proximity floor — avoids false alarms when ego is stopped
        // and a vehicle crosses in front (new track, inEgoPath briefly true, no TTC yet).

        // === SMOOTHING: per-level consecutive frames required ===
        if (config_.enableSmoothing) {
            auto& history = riskHistory_[trackId];
            history.push_back(ra.level);
            while (static_cast<int>(history.size()) > config_.smoothingWindow) {
                history.erase(history.begin());
            }

            int consecutive = 0;
            for (auto it = history.rbegin(); it != history.rend(); ++it) {
                if (*it >= ra.level) consecutive++;
                else break;
            }
            ra.consecutiveFrames = consecutive;

            if (ra.level > RiskLevel::SAFE && consecutive < getMinConsecutive(ra.level)) {
                auto prevIt = risks_.find(trackId);
                ra.level = (prevIt != risks_.end()) ? prevIt->second.level : RiskLevel::SAFE;
            }
        }

        newRisks[trackId] = ra;
    }

    // Cleanup history for removed tracks
    for (auto it = riskHistory_.begin(); it != riskHistory_.end();) {
        if (ttcResults.find(it->first) == ttcResults.end()) {
            it = riskHistory_.erase(it);
        } else {
            ++it;
        }
    }

    risks_ = newRisks;
    return risks_;
}

RiskLevel CollisionRisk::classifyRisk(float ttc) const {
    if (ttc <= config_.criticalTTC) return RiskLevel::CRITICAL;
    if (ttc <= config_.dangerTTC) return RiskLevel::DANGER;
    if (ttc <= config_.cautionTTC) return RiskLevel::CAUTION;
    return RiskLevel::SAFE;
}

int CollisionRisk::getMinConsecutive(RiskLevel level) const {
    switch (level) {
        case RiskLevel::CRITICAL: return config_.minConsecutiveCritical;
        case RiskLevel::DANGER:   return config_.minConsecutiveDanger;
        case RiskLevel::CAUTION:  return config_.minConsecutiveCaution;
        default:                  return 1;
    }
}

RiskAssessment CollisionRisk::getHighestRisk() const {
    RiskAssessment highest;
    highest.level = RiskLevel::SAFE;
    highest.ttc = 999.0f;

    for (const auto& [id, ra] : risks_) {
        if (ra.level > highest.level ||
            (ra.level == highest.level && ra.ttc < highest.ttc)) {
            highest = ra;
        }
    }
    return highest;
}

RiskAssessment CollisionRisk::getRisk(int trackId) const {
    auto it = risks_.find(trackId);
    if (it != risks_.end()) return it->second;
    return RiskAssessment();
}

void CollisionRisk::setConfig(const RiskConfig& config) {
    config_ = config;
}

} // namespace fcw
