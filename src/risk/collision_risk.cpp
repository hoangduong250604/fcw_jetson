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

        // Classify risk based on TTC
        if (ttcInfo.valid && ttcInfo.isApproaching) {
            ra.level = classifyRisk(ttcInfo.ttcSmoothed);
            
            // === ONCOMING VEHICLE SUPPRESSION ===
            // Oncoming vehicles have high closing speed but will pass by, not collide
            // Only allow warning if vehicle is directly in ego path (head-on scenario)
            if (ttcInfo.vehicleState == VehicleState::ONCOMING) {
                if (!ra.inEgoPath) {
                    ra.level = RiskLevel::SAFE;  // Not in our lane → no risk
                } else {
                    // In ego path + oncoming = potential head-on, keep warning but cap at DANGER
                    if (ra.level > RiskLevel::DANGER) {
                        ra.level = RiskLevel::DANGER;
                    }
                }
            }
            
            // === EDGE TRUNCATION SUPPRESSION ===
            // Vehicle leaving frame has unreliable distance/bbox → suppress false alarm
            if (isEdgeTruncated) {
                ra.level = RiskLevel::SAFE;  // Don't warn for vehicles exiting the frame
            }
            
            // Vehicles NOT in ego path: cap risk at CAUTION (no false alarms)
            if (!ra.inEgoPath && ra.level > RiskLevel::CAUTION) {
                ra.level = RiskLevel::CAUTION;
            }
        } else {
            ra.level = RiskLevel::SAFE;
        }

        // Smoothing: track risk level history
        if (config_.enableSmoothing) {
            auto& history = riskHistory_[trackId];
            history.push_back(ra.level);

            // Trim history
            while (static_cast<int>(history.size()) > config_.smoothingWindow) {
                history.erase(history.begin());
            }

            // Count consecutive frames at current level or higher
            int consecutive = 0;
            for (auto it = history.rbegin(); it != history.rend(); ++it) {
                if (*it >= ra.level) consecutive++;
                else break;
            }
            ra.consecutiveFrames = consecutive;

            // Only escalate if enough consecutive frames
            if (ra.level > RiskLevel::SAFE && consecutive < config_.minConsecutive) {
                // Check previous risk level
                auto prevIt = risks_.find(trackId);
                if (prevIt != risks_.end()) {
                    ra.level = prevIt->second.level;
                } else {
                    ra.level = RiskLevel::SAFE;
                }
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
