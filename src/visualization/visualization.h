#pragma once
// ==============================================================================
// Visualization - HUD overlay for FCW system
// ==============================================================================

#include <opencv2/core.hpp>
#include <vector>
#include <unordered_map>
#include "detection_result.h"
#include "track.h"
#include "distance_estimator.h"
#include "speed_estimator.h"
#include "ttc_calculator.h"
#include "risk_state.h"
#include "bev_distance.h"

namespace fcw {

struct VisConfig {
    bool showBBox = true;
    bool showTrackId = true;
    bool showDistance = true;
    bool showTTC = true;
    bool showSpeed = true;
    bool showFPS = true;
    bool showRiskOverlay = true;
    bool showDangerZone = true;        // Dynamic danger zone overlay
    bool showTrajectory = true;        // Track trajectory history
    float overlayAlpha = 0.3f;
    float dangerZoneAlpha = 0.2f;
    // Detection zone config
    float cautionDistanceM = 15.0f;    // Caution zone distance from ego (meters)
    float maxViewDistanceM = 50.0f;    // Maximum visible distance for zone mapping
    float dangerZoneReactionTime = 1.5f; // Reaction time (seconds) for danger zone size
    // BGR colors
    cv::Scalar colorSafe = {0, 255, 0};
    cv::Scalar colorCaution = {0, 255, 255};
    cv::Scalar colorDanger = {0, 165, 255};
    cv::Scalar colorCritical = {0, 0, 255};
    cv::Scalar detectionZoneColor = {0, 0, 255};  // Red (BGR)
};

class Visualization {
public:
    Visualization();
    explicit Visualization(const VisConfig& config);

    /**
     * Draw full HUD overlay on frame.
     */
    void draw(cv::Mat& frame,
              const std::vector<Track*>& tracks,
              const std::unordered_map<int, DistanceInfo>& distances,
              const std::unordered_map<int, SpeedInfo>& speeds,
              const std::unordered_map<int, TTCInfo>& ttcs,
              const std::unordered_map<int, RiskAssessment>& risks,
              double fps,
              const DetectionResult& detections = DetectionResult(),
              float egoSpeedKmh = 0.0f);

    /** Set config */
    void setConfig(const VisConfig& config);

    /** Set BEV estimator for danger zone visualization */
    void setBEVEstimator(const BEVDistanceEstimator* bev);

private:
    void drawBoundingBox(cv::Mat& frame, const Track* track,
                         const RiskAssessment& risk) const;
    void drawInfoLabel(cv::Mat& frame, const Track* track,
                       const DistanceInfo& dist, const SpeedInfo& speed,
                       const TTCInfo& ttc, const RiskAssessment& risk) const;
    void drawRiskOverlay(cv::Mat& frame, RiskLevel level) const;
    void drawDangerZone(cv::Mat& frame, float dangerDistance) const;
    void drawDangerZoneBySpeed(cv::Mat& frame, float egoSpeedKmh, RiskLevel riskLevel) const;
    void drawTrajectory(cv::Mat& frame, const Track* track, const cv::Scalar& color) const;
    void drawFPS(cv::Mat& frame, double fps) const;
    void drawDashboard(cv::Mat& frame, const RiskAssessment& highestRisk) const;
    void drawDetectionZone(cv::Mat& frame, const RiskAssessment& risk) const;
    void drawDetectionZoneStatic(cv::Mat& frame) const;
    void drawTrafficLightPanel(cv::Mat& frame, const DetectionResult& detections) const;
    void drawEgoSpeedPanel(cv::Mat& frame, float egoSpeedKmh) const;
    TrafficLightState analyzeTrafficLightColor(const cv::Mat& frame, const Detection& det) const;

    cv::Scalar getRiskColor(RiskLevel level) const;

    VisConfig config_;
    const BEVDistanceEstimator* bevEstimator_ = nullptr;
};

} // namespace fcw
