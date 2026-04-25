// ==============================================================================
// Visualization Implementation - HUD overlay
// ==============================================================================

#include "visualization.h"
#include "math_utils.h"

#include <opencv2/imgproc.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>

namespace fcw {

Visualization::Visualization() {}
Visualization::Visualization(const VisConfig& config) : config_(config) {}

void Visualization::draw(cv::Mat& frame,
                          const std::vector<Track*>& tracks,
                          const std::unordered_map<int, DistanceInfo>& distances,
                          const std::unordered_map<int, SpeedInfo>& speeds,
                          const std::unordered_map<int, TTCInfo>& ttcs,
                          const std::unordered_map<int, RiskAssessment>& risks,
                          double fps,
                          const DetectionResult& detections,
                          float egoSpeedKmh) {
    if (frame.empty()) return;

    // Find highest risk for overall overlay
    RiskAssessment highestRisk;
    highestRisk.level = RiskLevel::SAFE;
    for (const auto& [id, ra] : risks) {
        if (ra.level > highestRisk.level) highestRisk = ra;
    }

    // Draw risk overlay on entire frame
    if (config_.showRiskOverlay && highestRisk.level > RiskLevel::SAFE) {
        drawRiskOverlay(frame, highestRisk.level);
    }

    // Draw BEV danger zone
    if (config_.showDangerZone && bevEstimator_ && bevEstimator_->isCalibrated()) {
        float dangerDist = (highestRisk.ttc > 0 && highestRisk.relativeSpeed > 0)
                            ? highestRisk.relativeSpeed * 1.5f  // 1.5 sec reaction time
                            : 15.0f;  // default 15m
        drawDangerZone(frame, dangerDist);
    }

    // Draw dynamic danger zone based on ego speed (always visible)
    if (config_.showDangerZone) {
        drawDangerZoneBySpeed(frame, egoSpeedKmh, highestRisk.level);
    }

    // Draw per-track visualizations
    for (const Track* track : tracks) {
        int id = track->getId();

        RiskAssessment risk;
        auto riskIt = risks.find(id);
        if (riskIt != risks.end()) risk = riskIt->second;

        // Draw bounding box
        if (config_.showBBox) {
            drawBoundingBox(frame, track, risk);
        }

        // Draw trajectory
        if (config_.showTrajectory) {
            drawTrajectory(frame, track, getRiskColor(risk.level));
        }

        // Draw info labels
        DistanceInfo dist;
        SpeedInfo speed;
        TTCInfo ttc;
        auto distIt = distances.find(id);
        if (distIt != distances.end()) dist = distIt->second;
        auto speedIt = speeds.find(id);
        if (speedIt != speeds.end()) speed = speedIt->second;
        auto ttcIt = ttcs.find(id);
        if (ttcIt != ttcs.end()) ttc = ttcIt->second;

        drawInfoLabel(frame, track, dist, speed, ttc, risk);
    }

    // Draw FPS
    if (config_.showFPS) {
        drawFPS(frame, fps);
    }

    // Draw dashboard
    drawDashboard(frame, highestRisk);

    // Draw traffic light panel
    if (!detections.empty()) {
        drawTrafficLightPanel(frame, detections);
    }

    // Draw ego speed panel
    drawEgoSpeedPanel(frame, egoSpeedKmh);
}

void Visualization::drawBoundingBox(cv::Mat& frame, const Track* track,
                                     const RiskAssessment& risk) const {
    utils::BBox bbox = track->getBBox();
    cv::Rect rect = bbox.toRect();
    cv::Scalar color = getRiskColor(risk.level);
    int thickness = (risk.level >= RiskLevel::DANGER) ? 3 : 2;

    cv::rectangle(frame, rect, color, thickness);

    // Draw track ID
    if (config_.showTrackId) {
        std::string idText = "ID:" + std::to_string(track->getId());
        cv::putText(frame, idText,
                    cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

void Visualization::drawInfoLabel(cv::Mat& frame, const Track* track,
                                   const DistanceInfo& dist, const SpeedInfo& speed,
                                   const TTCInfo& ttc, const RiskAssessment& risk) const {
    utils::BBox bbox = track->getBBox();
    int x = static_cast<int>(bbox.x2) + 5;
    int y = static_cast<int>(bbox.y1);
    cv::Scalar color = getRiskColor(risk.level);
    int lineStep = 18;

    // Distance
    if (config_.showDistance && dist.valid) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << dist.smoothedDistance << "m";
        cv::putText(frame, oss.str(), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        y += lineStep;
    }

    // Speed display removed - only calculated internally, not shown on bbox
    // Ego speed is displayed in separate panel

    // TTC
    if (config_.showTTC && ttc.valid) {
        std::ostringstream oss;
        oss << "TTC:" << std::fixed << std::setprecision(1) << ttc.ttcSmoothed << "s";
        cv::putText(frame, oss.str(), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        y += lineStep;
    }

    // Risk level text
    if (risk.level > RiskLevel::SAFE) {
        cv::putText(frame, riskLevelToString(risk.level), cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

void Visualization::drawRiskOverlay(cv::Mat& frame, RiskLevel level) const {
    cv::Scalar color = getRiskColor(level);
    cv::Mat overlay = frame.clone();

    // Top and bottom warning bars
    int barHeight = 30;
    cv::rectangle(overlay, cv::Rect(0, 0, frame.cols, barHeight), color, -1);
    cv::rectangle(overlay, cv::Rect(0, frame.rows - barHeight, frame.cols, barHeight),
                  color, -1);

    // Blend
    cv::addWeighted(overlay, config_.overlayAlpha, frame, 1.0 - config_.overlayAlpha,
                    0, frame);

    // Warning text at top center
    std::string warningText = "!! " + riskLevelToString(level) + " !!";
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(warningText, cv::FONT_HERSHEY_SIMPLEX,
                                         1.0, 2, &baseline);
    cv::Point textPos((frame.cols - textSize.width) / 2, 22);
    cv::putText(frame, warningText, textPos,
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
}

void Visualization::drawFPS(cv::Mat& frame, double fps) const {
    std::ostringstream oss;
    oss << "FPS: " << std::fixed << std::setprecision(1) << fps;
    cv::putText(frame, oss.str(), cv::Point(10, frame.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
}

void Visualization::drawDashboard(cv::Mat& frame, const RiskAssessment& highestRisk) const {
    // Dashboard panel at bottom-right
    int panelW = 250, panelH = 80;
    int x = frame.cols - panelW - 10;
    int y = frame.rows - panelH - 40;

    // Semi-transparent background
    cv::Mat roi = frame(cv::Rect(x, y, panelW, panelH));
    cv::Mat overlay(panelH, panelW, frame.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(overlay, 0.6, roi, 0.4, 0, roi);

    cv::Scalar color = getRiskColor(highestRisk.level);

    // Risk level
    cv::putText(frame, "Risk: " + riskLevelToString(highestRisk.level),
                cv::Point(x + 10, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

    if (highestRisk.ttc > 0) {
        std::ostringstream oss;
        oss << "TTC: " << std::fixed << std::setprecision(1) << highestRisk.ttc << "s";
        cv::putText(frame, oss.str(), cv::Point(x + 10, y + 45),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        std::ostringstream oss2;
        oss2 << "Dist: " << std::fixed << std::setprecision(1) << highestRisk.distance << "m";
        cv::putText(frame, oss2.str(), cv::Point(x + 10, y + 65),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

void Visualization::drawDangerZone(cv::Mat& frame, float dangerDistance) const {
    if (!bevEstimator_ || !bevEstimator_->isCalibrated()) return;

    cv::Mat mask = bevEstimator_->getDangerZoneMask(frame.size(), dangerDistance);
    if (mask.empty()) return;

    // Draw danger zone as semi-transparent red overlay
    cv::Mat colorMask;
    cv::cvtColor(mask, colorMask, cv::COLOR_GRAY2BGR);
    // Make danger zone red
    cv::Mat redZone(frame.size(), CV_8UC3, cv::Scalar(0, 0, 200));
    redZone.copyTo(colorMask, mask);

    cv::addWeighted(colorMask, config_.dangerZoneAlpha, frame, 1.0, 0, frame);
}

void Visualization::drawTrajectory(cv::Mat& frame, const Track* track,
                                    const cv::Scalar& color) const {
    const auto& history = track->getHistory();
    if (history.size() < 2) return;

    for (size_t i = 1; i < history.size(); i++) {
        cv::Point p1(static_cast<int>(history[i - 1].centerX()),
                     static_cast<int>(history[i - 1].y2));
        cv::Point p2(static_cast<int>(history[i].centerX()),
                     static_cast<int>(history[i].y2));

        // Fade older points
        float alpha = static_cast<float>(i) / static_cast<float>(history.size());
        cv::Scalar fadedColor = color * static_cast<double>(alpha);
        cv::line(frame, p1, p2, fadedColor, 2);
    }
}

cv::Scalar Visualization::getRiskColor(RiskLevel level) const {
    switch (level) {
        case RiskLevel::SAFE:     return config_.colorSafe;
        case RiskLevel::CAUTION:  return config_.colorCaution;
        case RiskLevel::DANGER:   return config_.colorDanger;
        case RiskLevel::CRITICAL: return config_.colorCritical;
        default:                  return config_.colorSafe;
    }
}

void Visualization::setConfig(const VisConfig& config) {
    config_ = config;
}

void Visualization::setBEVEstimator(const BEVDistanceEstimator* bev) {
    bevEstimator_ = bev;
}

// ==============================================================================
// Dynamic Danger Zone - trapezoid that changes size & color with ego speed + risk
// ==============================================================================
// Inspired by open-adas: dangerDistance = speed(m/s) × reactionTime
// The zone grows with ego speed, and color changes with risk level:
//   SAFE     → green border (monitoring zone)
//   CAUTION  → yellow semi-transparent
//   DANGER   → orange semi-transparent  
//   CRITICAL → red flashing semi-transparent
// ==============================================================================
void Visualization::drawDangerZoneBySpeed(cv::Mat& frame, float egoSpeedKmh,
                                          RiskLevel riskLevel) const {
    int W = frame.cols;
    int H = frame.rows;

    // Calculate danger distance based on ego speed
    // dangerDistance = speed(m/s) × reaction_time
    float speedMs = std::abs(egoSpeedKmh) / 3.6f;
    float dangerDistM = speedMs * config_.dangerZoneReactionTime;

    // Clamp to reasonable range
    if (dangerDistM < 3.0f) dangerDistM = 3.0f;
    if (dangerDistM > config_.maxViewDistanceM) dangerDistM = config_.maxViewDistanceM;

    // Map danger distance to image Y coordinate
    // Bottom of frame = 0m (ego position), higher Y = further distance
    // Using perspective mapping: Y = H - (dist / maxDist) * H * scaleFactor
    float distRatio = dangerDistM / config_.maxViewDistanceM;
    int topY = static_cast<int>(H - distRatio * H * 0.6f);
    if (topY < static_cast<int>(H * 0.3f)) topY = static_cast<int>(H * 0.3f);

    int botY = H - 5;

    // Trapezoid: wide at bottom (near), narrow at top (far) - perspective effect
    float botWidthRatio = 0.76f;  // 76% of image width at bottom
    float perspectiveRatio = 0.25f + 0.20f * (1.0f - distRatio);  // Narrows with distance

    int botLeft  = static_cast<int>(W * (0.5f - botWidthRatio / 2.0f));
    int botRight = static_cast<int>(W * (0.5f + botWidthRatio / 2.0f));
    int topLeft  = static_cast<int>(W * (0.5f - perspectiveRatio / 2.0f));
    int topRight = static_cast<int>(W * (0.5f + perspectiveRatio / 2.0f));

    std::vector<cv::Point> trapezoid = {
        cv::Point(topLeft, topY),
        cv::Point(topRight, topY),
        cv::Point(botRight, botY),
        cv::Point(botLeft, botY)
    };

    // Color based on risk level
    cv::Scalar zoneColor;
    float alpha;
    switch (riskLevel) {
        case RiskLevel::CRITICAL:
            zoneColor = cv::Scalar(0, 0, 255);     // Red
            alpha = 0.30f;
            break;
        case RiskLevel::DANGER:
            zoneColor = cv::Scalar(0, 100, 255);   // Orange
            alpha = 0.25f;
            break;
        case RiskLevel::CAUTION:
            zoneColor = cv::Scalar(0, 200, 255);   // Yellow
            alpha = 0.18f;
            break;
        default:
            zoneColor = cv::Scalar(0, 180, 0);     // Green
            alpha = 0.10f;
            break;
    }

    // Draw filled trapezoid with transparency
    cv::Mat overlay = frame.clone();
    std::vector<std::vector<cv::Point>> pts = {trapezoid};
    cv::fillPoly(overlay, pts, zoneColor);
    cv::addWeighted(overlay, alpha, frame, 1.0f - alpha, 0, frame);

    // Border (thicker for higher risk)
    int borderThick = (riskLevel >= RiskLevel::DANGER) ? 2 : 1;
    cv::polylines(frame, trapezoid, true, zoneColor, borderThick);

    // Distance label at top of zone
    char distLabel[32];
    snprintf(distLabel, sizeof(distLabel), "%.0fm", dangerDistM);
    int baseline = 0;
    cv::Size textSz = cv::getTextSize(distLabel, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
    cv::Point labelPos((topLeft + topRight) / 2 - textSz.width / 2, topY - 5);
    cv::putText(frame, distLabel, labelPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.45, zoneColor, 1);
}

// ==============================================================================
// Detection Zone - trapezoid in front of ego vehicle covering CAUTION distance
// ==============================================================================
void Visualization::drawDetectionZone(cv::Mat& frame, const RiskAssessment& risk) const {
    if (!bevEstimator_ || !bevEstimator_->isCalibrated()) {
        // Fallback to static trapezoid if BEV not available
        drawDetectionZoneStatic(frame);
        return;
    }

    int W = frame.cols;
    int H = frame.rows;

    // Use config's caution distance to define zone
    float zoneDistance = config_.cautionDistanceM;  // e.g., 15 meters

    // Use BEV to get trapezoid points at this distance
    // Trapezoid represents: bottom (near) and top (far) edges of detection zone
    
    // Near edge: 0m from ego (bottom of frame)
    int botY = H - 5;
    int botLeft = static_cast<int>(W * 0.12);
    int botRight = static_cast<int>(W * 0.88);
    
    // Far edge: at cautionDistanceM from ego
    // Project world distance to image Y coordinate
    // Simple mapping: assuming bottom = 0m, top_full_view = ~40m
    // Linear interpolation: Y = H - (distance / maxDistance) * H
    float maxViewDistance = 50.0f;  // Maximum visible distance in BEV
    float topY = static_cast<int>(H - (zoneDistance / maxViewDistance) * H * 0.6f);
    
    // Perspective: far trapezoid is narrower
    float perspectiveRatio = 0.45f;  // Relative width at far edge
    int topLeft = static_cast<int>(W * (0.5f - perspectiveRatio / 2));
    int topRight = static_cast<int>(W * (0.5f + perspectiveRatio / 2));

    std::vector<cv::Point> trapezoid = {
        cv::Point(topLeft, topY),
        cv::Point(topRight, topY),
        cv::Point(botRight, botY),
        cv::Point(botLeft, botY)
    };

    // Draw red filled trapezoid (constant red color)
    cv::Mat overlay = frame.clone();
    std::vector<std::vector<cv::Point>> pts = {trapezoid};
    cv::fillPoly(overlay, pts, config_.detectionZoneColor);  // Red
    cv::addWeighted(overlay, 0.25f, frame, 0.75f, 0, frame);

    // Red border
    cv::polylines(frame, trapezoid, true, config_.detectionZoneColor, 1);
}

void Visualization::drawDetectionZoneStatic(cv::Mat& frame) const {
    int W = frame.cols;
    int H = frame.rows;

    // Small trapezoid close to bottom of frame = right in front of ego vehicle on the road
    int botY = H - 5;
    int topY = static_cast<int>(H * 0.82);   // thin strip near bottom
    int botLeft = static_cast<int>(W * 0.12);
    int botRight = static_cast<int>(W * 0.88);
    int topLeft = static_cast<int>(W * 0.30);
    int topRight = static_cast<int>(W * 0.70);

    std::vector<cv::Point> trapezoid = {
        cv::Point(topLeft, topY),
        cv::Point(topRight, topY),
        cv::Point(botRight, botY),
        cv::Point(botLeft, botY)
    };

    // Draw light red filled trapezoid (constant red color)
    cv::Mat overlay = frame.clone();
    std::vector<std::vector<cv::Point>> pts = {trapezoid};
    cv::fillPoly(overlay, pts, config_.detectionZoneColor);  // Red
    cv::addWeighted(overlay, 0.25f, frame, 0.75f, 0, frame);

    // Red border
    cv::polylines(frame, trapezoid, true, config_.detectionZoneColor, 1);
}

// ==============================================================================
// Traffic Light Color Analysis
// ==============================================================================
TrafficLightState Visualization::analyzeTrafficLightColor(const cv::Mat& frame,
                                                           const Detection& det) const {
    cv::Rect roi = det.getRect();
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (roi.width < 5 || roi.height < 5) return TrafficLightState::UNKNOWN;

    cv::Mat crop = frame(roi);
    cv::Mat hsv;
    cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

    // Split into top/middle/bottom thirds
    int h3 = crop.rows / 3;
    if (h3 < 2) return TrafficLightState::UNKNOWN;

    cv::Mat topROI = hsv(cv::Rect(0, 0, crop.cols, h3));
    cv::Mat midROI = hsv(cv::Rect(0, h3, crop.cols, h3));
    cv::Mat botROI = hsv(cv::Rect(0, 2 * h3, crop.cols, crop.rows - 2 * h3));

    // Calculate mean brightness (V channel) of each third
    cv::Scalar topMean = cv::mean(topROI);
    cv::Scalar midMean = cv::mean(midROI);
    cv::Scalar botMean = cv::mean(botROI);

    double topV = topMean[2];  // V channel
    double midV = midMean[2];
    double botV = botMean[2];

    // The active light is the brightest region
    double maxV = std::max({topV, midV, botV});
    double threshold = 80.0;  // minimum brightness to consider active

    if (maxV < threshold) return TrafficLightState::UNKNOWN;

    if (topV == maxV && topV > midV * 1.2 && topV > botV * 1.2) {
        return TrafficLightState::RED;
    } else if (botV == maxV && botV > topV * 1.2 && botV > midV * 1.2) {
        return TrafficLightState::GREEN;
    } else if (midV == maxV && midV > topV * 1.2 && midV > botV * 1.2) {
        return TrafficLightState::YELLOW;
    }

    // Fallback: analyze hue of brightest region
    // Red: H in [0,10] or [170,180], Green: H in [35,85], Yellow: H in [20,35]
    double topH = topMean[0], midH = midMean[0], botH = botMean[0];
    if (topV >= midV && topV >= botV) {
        if (topH < 10 || topH > 170) return TrafficLightState::RED;
    }
    if (botV >= topV && botV >= midV) {
        if (botH > 35 && botH < 85) return TrafficLightState::GREEN;
    }
    if (midV >= topV && midV >= botV) {
        if (midH > 15 && midH < 40) return TrafficLightState::YELLOW;
    }

    return TrafficLightState::UNKNOWN;
}

// ==============================================================================
// Traffic Light Panel - right side display
// ==============================================================================
void Visualization::drawTrafficLightPanel(cv::Mat& frame,
                                           const DetectionResult& detections) const {
    auto lights = detections.getTrafficLights();
    if (lights.empty()) return;

    // Find the most confident / largest traffic light
    const Detection* bestLight = lights[0];
    for (const auto* light : lights) {
        if (light->confidence > bestLight->confidence) bestLight = light;
    }

    TrafficLightState state = analyzeTrafficLightColor(frame, *bestLight);

    // Draw panel on the right side (below dashboard area)
    int panelW = 160, panelH = 110;
    int x = frame.cols - panelW - 10;
    int y = 10;  // top-right corner

    // Semi-transparent background
    cv::Rect panelRect(x, y, panelW, panelH);
    panelRect &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (panelRect.width <= 0 || panelRect.height <= 0) return;

    cv::Mat roi = frame(panelRect);
    cv::Mat bgOverlay(panelRect.size(), frame.type(), cv::Scalar(30, 30, 30));
    cv::addWeighted(bgOverlay, 0.7, roi, 0.3, 0, roi);

    // Border
    cv::rectangle(frame, panelRect, cv::Scalar(200, 200, 200), 1);

    // Title
    cv::putText(frame, "TRAFFIC LIGHT", cv::Point(x + 10, y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    // Draw traffic light icon (3 circles)
    int iconX = x + 35;
    int iconY = y + 50;
    int radius = 10;
    int spacing = 25;

    // Red circle (top)
    cv::Scalar redColor = (state == TrafficLightState::RED)
        ? cv::Scalar(0, 0, 255) : cv::Scalar(50, 50, 80);
    cv::circle(frame, cv::Point(iconX, iconY), radius, redColor, -1);

    // Yellow circle (middle)
    cv::Scalar yellowColor = (state == TrafficLightState::YELLOW)
        ? cv::Scalar(0, 255, 255) : cv::Scalar(50, 60, 80);
    cv::circle(frame, cv::Point(iconX, iconY + spacing), radius, yellowColor, -1);

    // Green circle (bottom)
    cv::Scalar greenColor = (state == TrafficLightState::GREEN)
        ? cv::Scalar(0, 255, 0) : cv::Scalar(50, 80, 50);
    cv::circle(frame, cv::Point(iconX, iconY + 2 * spacing), radius, greenColor, -1);

    // State text
    std::string stateText;
    cv::Scalar textColor;
    switch (state) {
        case TrafficLightState::RED:
            stateText = "RED";
            textColor = cv::Scalar(0, 0, 255);
            break;
        case TrafficLightState::GREEN:
            stateText = "GREEN";
            textColor = cv::Scalar(0, 255, 0);
            break;
        case TrafficLightState::YELLOW:
            stateText = "YELLOW";
            textColor = cv::Scalar(0, 255, 255);
            break;
        default:
            stateText = "---";
            textColor = cv::Scalar(150, 150, 150);
            break;
    }
    cv::putText(frame, stateText, cv::Point(iconX + 20, iconY + spacing + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);

    // Also draw bbox around the detected traffic light in the frame
    cv::Rect tlRect = bestLight->getRect();
    cv::rectangle(frame, tlRect, cv::Scalar(0, 255, 255), 2);
    cv::putText(frame, "TL", cv::Point(tlRect.x, tlRect.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
}

// ==============================================================================
// Ego Speed Panel - display current vehicle speed
// ==============================================================================
void Visualization::drawEgoSpeedPanel(cv::Mat& frame, float egoSpeedKmh) const {
    int panelW = 160, panelH = 60;
    int x = frame.cols - panelW - 10;
    int y = 130;  // below traffic light panel

    // Semi-transparent background
    cv::Rect panelRect(x, y, panelW, panelH);
    panelRect &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (panelRect.width <= 0 || panelRect.height <= 0) return;

    cv::Mat roi = frame(panelRect);
    cv::Mat bgOverlay(panelRect.size(), frame.type(), cv::Scalar(30, 30, 30));
    cv::addWeighted(bgOverlay, 0.7, roi, 0.3, 0, roi);

    cv::rectangle(frame, panelRect, cv::Scalar(200, 200, 200), 1);

    // Title
    cv::putText(frame, "EGO SPEED", cv::Point(x + 10, y + 18),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);

    // Speed value
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << std::abs(egoSpeedKmh) << " km/h";
    cv::Scalar speedColor = (egoSpeedKmh > 80.0f) ? cv::Scalar(0, 100, 255) :
                            (egoSpeedKmh > 50.0f) ? cv::Scalar(0, 200, 255) :
                                                     cv::Scalar(0, 255, 0);
    cv::putText(frame, oss.str(), cv::Point(x + 15, y + 48),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, speedColor, 2);
}

} // namespace fcw
