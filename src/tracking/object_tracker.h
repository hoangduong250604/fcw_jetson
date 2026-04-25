#pragma once
// ==============================================================================
// Object Tracker - Multi-object tracking manager (SORT algorithm)
// ==============================================================================
// CORE MODULE: Manages all active tracks and performs frame-by-frame association.
//
// SORT Algorithm Steps (per frame):
//   1. Predict all existing tracks forward (Kalman predict)
//   2. Compute cost matrix (IoU) between predictions and new detections
//   3. Solve assignment using Hungarian algorithm
//   4. Update matched tracks with new detections (Kalman update)
//   5. Create new tracks for unmatched detections
//   6. Mark unmatched tracks as lost / delete old tracks
//
// Outputs: List of active Track objects with IDs, bounding boxes, and velocities.
// These are consumed by distance estimation, speed estimation, and TTC.
// ==============================================================================

#include <vector>
#include <memory>
#include "track.h"
#include "detection_result.h"

namespace fcw {

struct TrackerConfig {
    float maxDistance = 100.0f;      // Max center distance for association
    int maxLost = 30;               // Max frames before deleting lost track
    int minHits = 3;                // Min hits before confirming track
    float iouThreshold = 0.3f;      // Min IoU for valid association
    bool useKalman = true;          // Use Kalman filter for prediction
};

/**
 * Multi-object tracker using SORT (Simple Online Realtime Tracking).
 * 
 * This tracker:
 *   - Assigns a unique ID to each detected vehicle
 *   - Maintains track continuity across frames
 *   - Provides velocity estimates via Kalman Filter
 *   - Handles temporary occlusions through prediction
 */
class ObjectTracker {
public:
    ObjectTracker();
    explicit ObjectTracker(const TrackerConfig& config);
    ~ObjectTracker() = default;

    /**
     * Update tracker with new detections for the current frame.
     * 
     * This is the main entry point per frame:
     *   1. Predict existing tracks
     *   2. Associate detections with tracks
     *   3. Update matched, create new, handle unmatched
     *
     * @param detections  New detections from YOLOv8
     * @return            List of active confirmed tracks
     */
    std::vector<Track*> update(const DetectionResult& detections);

    /** Get all active tracks (confirmed only) */
    std::vector<Track*> getActiveTracks() const;

    /** Get all tracks including tentative and lost */
    std::vector<Track*> getAllTracks() const;

    /** Get specific track by ID */
    Track* getTrack(int trackId) const;

    /** Get total number of tracks ever created */
    int getTotalTrackCount() const { return nextTrackId_; }

    /** Reset tracker state */
    void reset();

    /** Set config */
    void setConfig(const TrackerConfig& config);

private:
    // ---- Association ----

    /**
     * Compute IoU cost matrix between predicted track positions and detections.
     * cost[i][j] = 1 - IoU(track_i, detection_j)
     */
    std::vector<std::vector<float>> computeCostMatrix(
        const std::vector<Track*>& tracks,
        const std::vector<Detection>& detections) const;

    /**
     * Hungarian algorithm for optimal assignment.
     * Solves the linear assignment problem on the cost matrix.
     * 
     * @param costMatrix  N x M cost matrix
     * @return Vector of (track_idx, detection_idx) pairs
     */
    std::vector<std::pair<int, int>> hungarianAssignment(
        const std::vector<std::vector<float>>& costMatrix) const;

    /**
     * Simple greedy association (fallback, faster than Hungarian).
     * For each detection, find the best matching track with IoU above threshold.
     */
    std::vector<std::pair<int, int>> greedyAssignment(
        const std::vector<std::vector<float>>& costMatrix) const;

    // ---- Track Management ----
    
    /** Create a new track from unmatched detection */
    void createTrack(const Detection& det);

    /** Remove deleted tracks */
    void pruneDeletedTracks();

    // ---- Members ----
    TrackerConfig config_;
    std::vector<std::unique_ptr<Track>> tracks_;
    int nextTrackId_ = 0;
};

} // namespace fcw
