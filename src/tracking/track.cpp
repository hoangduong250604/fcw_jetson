// ==============================================================================
// Track Implementation - Kalman Filter-based single object track
// ==============================================================================
// CORE MODULE
//
// Kalman Filter for bounding box tracking:
//
//   State (7D):       x = [cx, cy, s, r, vx, vy, vs]^T
//   Measurement (4D): z = [cx, cy, s, r]^T
//
//   Transition model (constant velocity):
//     cx' = cx + vx
//     cy' = cy + vy
//     s'  = s + vs
//     r'  = r
//     vx' = vx
//     vy' = vy
//     vs' = vs
//
//   Where:
//     cx, cy = bbox center coordinates
//     s = bbox area (width * height)
//     r = aspect ratio (width / height)
//     vx, vy = center velocity
//     vs = scale velocity
//
// This gives us velocity estimation for free - critical for TTC calculation.
// ==============================================================================

#include "track.h"
#include "logger.h"
#include <cmath>

namespace fcw {

// Static member initialization
int Track::maxLost_ = 30;
int Track::minHits_ = 3;

// ==============================================================================
// Constructor
// ==============================================================================
Track::Track(const Detection& det, int trackId)
    : trackId_(trackId), lastDetection_(det), currentBBox_(det.bbox) {
    initKalmanFilter(det.bbox);
    hitCount_ = 1;
    age_ = 1;
    bboxHistory_.push_back(det.bbox);
}

// ==============================================================================
// Kalman Filter Initialization
// ==============================================================================
void Track::initKalmanFilter(const utils::BBox& bbox) {
    // 7 state dimensions, 4 measurement dimensions, 0 control dimensions
    kf_ = cv::KalmanFilter(7, 4, 0);

    // Transition matrix F (constant velocity model)
    // [1, 0, 0, 0, 1, 0, 0]   cx' = cx + vx
    // [0, 1, 0, 0, 0, 1, 0]   cy' = cy + vy
    // [0, 0, 1, 0, 0, 0, 1]   s'  = s + vs
    // [0, 0, 0, 1, 0, 0, 0]   r'  = r
    // [0, 0, 0, 0, 1, 0, 0]   vx' = vx
    // [0, 0, 0, 0, 0, 1, 0]   vy' = vy
    // [0, 0, 0, 0, 0, 0, 1]   vs' = vs
    kf_.transitionMatrix = cv::Mat::eye(7, 7, CV_32F);
    kf_.transitionMatrix.at<float>(0, 4) = 1.0f;  // cx += vx
    kf_.transitionMatrix.at<float>(1, 5) = 1.0f;  // cy += vy
    kf_.transitionMatrix.at<float>(2, 6) = 1.0f;  // s += vs

    // Measurement matrix H (we only observe [cx, cy, s, r])
    kf_.measurementMatrix = cv::Mat::zeros(4, 7, CV_32F);
    kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf_.measurementMatrix.at<float>(2, 2) = 1.0f;
    kf_.measurementMatrix.at<float>(3, 3) = 1.0f;

    // Process noise covariance Q
    kf_.processNoiseCov = cv::Mat::eye(7, 7, CV_32F);
    kf_.processNoiseCov.at<float>(0, 0) = 1.0f;     // cx
    kf_.processNoiseCov.at<float>(1, 1) = 1.0f;     // cy
    kf_.processNoiseCov.at<float>(2, 2) = 1.0f;     // s
    kf_.processNoiseCov.at<float>(3, 3) = 1.0f;     // r
    kf_.processNoiseCov.at<float>(4, 4) = 0.01f;    // vx (lower = smoother)
    kf_.processNoiseCov.at<float>(5, 5) = 0.01f;    // vy
    kf_.processNoiseCov.at<float>(6, 6) = 0.0001f;  // vs (area velocity, small)

    // Measurement noise covariance R
    kf_.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F);
    kf_.measurementNoiseCov.at<float>(0, 0) = 1.0f;   // cx noise
    kf_.measurementNoiseCov.at<float>(1, 1) = 1.0f;   // cy noise
    kf_.measurementNoiseCov.at<float>(2, 2) = 10.0f;  // s noise (area, more uncertain)
    kf_.measurementNoiseCov.at<float>(3, 3) = 10.0f;  // r noise

    // Error covariance P (initial uncertainty)
    kf_.errorCovPost = cv::Mat::eye(7, 7, CV_32F);
    kf_.errorCovPost.at<float>(4, 4) = 10.0f;    // High initial velocity uncertainty
    kf_.errorCovPost.at<float>(5, 5) = 10.0f;
    kf_.errorCovPost.at<float>(6, 6) = 10.0f;

    // Initial state from first detection
    float cx = bbox.centerX();
    float cy = bbox.centerY();
    float s = bbox.area();
    float r = bbox.width() / std::max(bbox.height(), 1.0f);

    kf_.statePost.at<float>(0) = cx;
    kf_.statePost.at<float>(1) = cy;
    kf_.statePost.at<float>(2) = s;
    kf_.statePost.at<float>(3) = r;
    kf_.statePost.at<float>(4) = 0.0f;  // vx = 0 initially
    kf_.statePost.at<float>(5) = 0.0f;  // vy = 0
    kf_.statePost.at<float>(6) = 0.0f;  // vs = 0
}

// ==============================================================================
// Predict
// ==============================================================================
utils::BBox Track::predict() {
    // Prevent negative area
    if (kf_.statePost.at<float>(2) + kf_.statePost.at<float>(6) <= 0) {
        kf_.statePost.at<float>(6) = 0.0f;
    }

    cv::Mat prediction = kf_.predict();
    currentBBox_ = stateToBBox();
    age_++;

    return currentBBox_;
}

// ==============================================================================
// Update (matched detection)
// ==============================================================================
void Track::update(const Detection& det) {
    // Correct Kalman state with measurement
    cv::Mat measurement = bboxToMeasurement(det.bbox);
    kf_.correct(measurement);

    // Update track info
    currentBBox_ = det.bbox;
    lastDetection_ = det;
    hitCount_++;
    lostCount_ = 0;

    // State transition
    if (state_ == TrackState::TENTATIVE && hitCount_ >= minHits_) {
        state_ = TrackState::CONFIRMED;
    } else if (state_ == TrackState::LOST) {
        state_ = TrackState::CONFIRMED;
    }

    // Update history
    bboxHistory_.push_back(currentBBox_);
    if (static_cast<int>(bboxHistory_.size()) > MAX_HISTORY) {
        bboxHistory_.erase(bboxHistory_.begin());
    }
}

// ==============================================================================
// Mark Missed
// ==============================================================================
void Track::markMissed() {
    lostCount_++;

    if (state_ == TrackState::TENTATIVE) {
        // Delete tentative tracks immediately if unmatched
        state_ = TrackState::DELETED;
    } else if (lostCount_ > maxLost_) {
        // Too many missed frames → delete
        state_ = TrackState::DELETED;
    } else if (state_ == TrackState::CONFIRMED) {
        state_ = TrackState::LOST;
    }
}

// ==============================================================================
// Getters
// ==============================================================================
cv::Point2f Track::getVelocity() const {
    float vx = kf_.statePost.at<float>(4);
    float vy = kf_.statePost.at<float>(5);
    return cv::Point2f(vx, vy);
}

float Track::getScaleVelocity() const {
    return kf_.statePost.at<float>(6);
}

// ==============================================================================
// Helpers
// ==============================================================================
cv::Mat Track::bboxToMeasurement(const utils::BBox& bbox) const {
    cv::Mat z(4, 1, CV_32F);
    z.at<float>(0) = bbox.centerX();
    z.at<float>(1) = bbox.centerY();
    z.at<float>(2) = bbox.area();
    z.at<float>(3) = bbox.width() / std::max(bbox.height(), 1.0f);
    return z;
}

utils::BBox Track::stateToBBox() const {
    float cx = kf_.statePost.at<float>(0);
    float cy = kf_.statePost.at<float>(1);
    float s = std::max(kf_.statePost.at<float>(2), 1.0f);  // area
    float r = kf_.statePost.at<float>(3);                    // aspect ratio

    float w = std::sqrt(s * r);
    float h = s / std::max(w, 1.0f);

    utils::BBox bbox;
    bbox.x1 = cx - w / 2.0f;
    bbox.y1 = cy - h / 2.0f;
    bbox.x2 = cx + w / 2.0f;
    bbox.y2 = cy + h / 2.0f;
    return bbox;
}

} // namespace fcw
