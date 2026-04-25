#pragma once
// ==============================================================================
// Track - Single object track with Kalman Filter state estimation
// ==============================================================================
// CORE MODULE: Each Track represents one tracked vehicle across frames.
//
// State vector (Kalman Filter):
//   [cx, cy, s, r, vx, vy, vs]
//   cx, cy = bounding box center
//   s      = scale (area)
//   r      = aspect ratio (w/h)
//   vx, vy = velocity of center
//   vs     = velocity of scale
//
// Based on SORT (Simple Online and Realtime Tracking) algorithm.
// ==============================================================================

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "detection_result.h"
#include "math_utils.h"

namespace fcw {

enum class TrackState {
    TENTATIVE = 0,   // Not yet confirmed (< min_hits)
    CONFIRMED = 1,   // Active confirmed track
    LOST = 2,        // Lost for some frames (prediction only)
    DELETED = 3      // Marked for deletion
};

/**
 * Single object track using Kalman Filter for state prediction.
 * 
 * Lifecycle:
 *   TENTATIVE → (min_hits reached) → CONFIRMED → (no match) → LOST → (max_age) → DELETED
 *
 * Kalman Filter predicts position when detection is missing,
 * enabling smooth tracking through brief occlusions.
 */
class Track {
public:
    /**
     * Create a new track from an initial detection.
     * @param det     Initial detection
     * @param trackId Unique track identifier
     */
    Track(const Detection& det, int trackId);
    ~Track() = default;

    // ---- Kalman Filter Operations ----

    /**
     * Predict next state using Kalman Filter.
     * Called at the beginning of each frame before association.
     * Returns predicted bounding box.
     */
    utils::BBox predict();

    /**
     * Update track with a matched detection.
     * Corrects the Kalman Filter state with measurement.
     */
    void update(const Detection& det);

    /**
     * Mark track as unmatched for this frame.
     * Increments lost counter and may change state.
     */
    void markMissed();

    // ---- Getters ----

    int getId() const { return trackId_; }
    TrackState getState() const { return state_; }
    utils::BBox getBBox() const { return currentBBox_; }
    Detection getLastDetection() const { return lastDetection_; }
    int getHitCount() const { return hitCount_; }
    int getLostCount() const { return lostCount_; }
    int getAge() const { return age_; }
    float getConfidence() const { return lastDetection_.confidence; }
    int getClassId() const { return lastDetection_.classId; }

    /** Check if track is confirmed and active */
    bool isConfirmed() const { return state_ == TrackState::CONFIRMED; }

    /** Check if track should be deleted */
    bool isDeleted() const { return state_ == TrackState::DELETED; }

    /** Get velocity of bbox center (pixels/frame) */
    cv::Point2f getVelocity() const;

    /** Get bbox height change rate (pixels/frame) - proxy for distance change */
    float getScaleVelocity() const;

    /** Get history of bounding boxes (for trajectory visualization) */
    const std::vector<utils::BBox>& getHistory() const { return bboxHistory_; }

    // ---- Configuration ----
    static void setMaxLost(int maxLost) { maxLost_ = maxLost; }
    static void setMinHits(int minHits) { minHits_ = minHits; }

private:
    /**
     * Initialize Kalman Filter.
     * State: [cx, cy, s, r, vx, vy, vs] (7D)
     * Measurement: [cx, cy, s, r] (4D)
     */
    void initKalmanFilter(const utils::BBox& bbox);

    /** Convert bbox to measurement vector [cx, cy, s, r] */
    cv::Mat bboxToMeasurement(const utils::BBox& bbox) const;

    /** Convert Kalman state to BBox */
    utils::BBox stateToBBox() const;

    // Track identity
    int trackId_;
    TrackState state_ = TrackState::TENTATIVE;

    // Kalman Filter
    cv::KalmanFilter kf_;

    // Detection info
    Detection lastDetection_;
    utils::BBox currentBBox_;

    // Track statistics
    int age_ = 0;          // Total frames since creation
    int hitCount_ = 0;     // Total successful associations
    int lostCount_ = 0;    // Consecutive frames without association

    // History for trajectory
    std::vector<utils::BBox> bboxHistory_;
    static constexpr int MAX_HISTORY = 60;

    // Configuration (shared across all tracks)
    static int maxLost_;    // Max frames before deletion
    static int minHits_;    // Min hits before confirmation
};

} // namespace fcw
