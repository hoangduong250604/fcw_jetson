// ==============================================================================
// Object Tracker Implementation - SORT Algorithm
// ==============================================================================
// CORE MODULE
//
// Implements SORT (Simple Online and Realtime Tracking):
//   Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
//
// Per-frame algorithm:
//   1. PREDICT: Use Kalman Filter to predict each track's position
//   2. ASSOCIATE: Build IoU cost matrix, solve with Hungarian/greedy
//   3. UPDATE: Update matched tracks, create new for unmatched detections
//   4. PRUNE: Delete tracks that have been lost too long
//
// The tracker provides per-track velocity via Kalman Filter, which is
// essential for the downstream TTC calculation module.
// ==============================================================================

#include "object_tracker.h"
#include "hungarian.h"
#include "math_utils.h"
#include "logger.h"

#include <algorithm>
#include <numeric>
#include <limits>
#include <set>

namespace fcw {

// ==============================================================================
// Constructor
// ==============================================================================
ObjectTracker::ObjectTracker() {
    Track::setMaxLost(config_.maxLost);
    Track::setMinHits(config_.minHits);
}

ObjectTracker::ObjectTracker(const TrackerConfig& config) : config_(config) {
    Track::setMaxLost(config.maxLost);
    Track::setMinHits(config.minHits);
}

// ==============================================================================
// Main Update (called once per frame)
// ==============================================================================
std::vector<Track*> ObjectTracker::update(const DetectionResult& detections) {
    // =============================================
    // Step 1: PREDICT all existing tracks
    // =============================================
    for (auto& track : tracks_) {
        track->predict();
    }

    // =============================================
    // Step 2: ASSOCIATE detections with tracks
    // =============================================
    // Get list of existing tracks (non-deleted)
    std::vector<Track*> existingTracks;
    for (auto& t : tracks_) {
        if (!t->isDeleted()) {
            existingTracks.push_back(t.get());
        }
    }

    const auto& dets = detections.detections;

    // Track matched/unmatched indices
    std::set<int> matchedTrackIdx;
    std::set<int> matchedDetIdx;
    std::vector<std::pair<int, int>> matches;

    if (!existingTracks.empty() && !dets.empty()) {
        // Compute IoU cost matrix
        auto costMatrix = computeCostMatrix(existingTracks, dets);

        // Solve assignment using Hungarian algorithm (optimal)
        matches = hungarianAssignment(costMatrix);

        for (auto& [tIdx, dIdx] : matches) {
            matchedTrackIdx.insert(tIdx);
            matchedDetIdx.insert(dIdx);
        }
    }

    // =============================================
    // Step 3: UPDATE matched tracks
    // =============================================
    for (auto& [tIdx, dIdx] : matches) {
        existingTracks[tIdx]->update(dets[dIdx]);
    }

    // =============================================
    // Step 4: Handle UNMATCHED TRACKS (mark missed)
    // =============================================
    for (int i = 0; i < static_cast<int>(existingTracks.size()); i++) {
        if (matchedTrackIdx.find(i) == matchedTrackIdx.end()) {
            existingTracks[i]->markMissed();
        }
    }

    // =============================================
    // Step 5: Create NEW TRACKS for unmatched detections
    // =============================================
    for (int i = 0; i < static_cast<int>(dets.size()); i++) {
        if (matchedDetIdx.find(i) == matchedDetIdx.end()) {
            createTrack(dets[i]);
        }
    }

    // =============================================
    // Step 6: PRUNE deleted tracks
    // =============================================
    pruneDeletedTracks();

    // Return active (confirmed) tracks
    return getActiveTracks();
}

// ==============================================================================
// Cost Matrix Computation
// ==============================================================================
std::vector<std::vector<float>> ObjectTracker::computeCostMatrix(
    const std::vector<Track*>& tracks,
    const std::vector<Detection>& detections) const {

    int numTracks = static_cast<int>(tracks.size());
    int numDets = static_cast<int>(detections.size());

    std::vector<std::vector<float>> cost(numTracks, std::vector<float>(numDets, 1.0f));

    for (int i = 0; i < numTracks; i++) {
        utils::BBox trackBBox = tracks[i]->getBBox();
        for (int j = 0; j < numDets; j++) {
            float iou = utils::computeIoU(trackBBox, detections[j].bbox);

            // Cost = 1 - IoU (lower cost = better match)
            cost[i][j] = 1.0f - iou;
        }
    }

    return cost;
}

// ==============================================================================
// Greedy Assignment (fast approximation of Hungarian)
// ==============================================================================
std::vector<std::pair<int, int>> ObjectTracker::greedyAssignment(
    const std::vector<std::vector<float>>& costMatrix) const {

    std::vector<std::pair<int, int>> matches;
    if (costMatrix.empty()) return matches;

    int numTracks = static_cast<int>(costMatrix.size());
    int numDets = static_cast<int>(costMatrix[0].size());

    // Collect all (cost, trackIdx, detIdx) and sort by cost ascending
    struct CostEntry {
        float cost;
        int trackIdx;
        int detIdx;
    };

    std::vector<CostEntry> entries;
    entries.reserve(numTracks * numDets);

    for (int i = 0; i < numTracks; i++) {
        for (int j = 0; j < numDets; j++) {
            entries.push_back({costMatrix[i][j], i, j});
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const CostEntry& a, const CostEntry& b) {
                  return a.cost < b.cost;
              });

    std::set<int> usedTracks, usedDets;

    for (const auto& entry : entries) {
        if (usedTracks.count(entry.trackIdx) || usedDets.count(entry.detIdx)) continue;

        // Only accept if IoU is above threshold (cost < 1 - iouThreshold)
        float iou = 1.0f - entry.cost;
        if (iou < config_.iouThreshold) continue;

        matches.push_back({entry.trackIdx, entry.detIdx});
        usedTracks.insert(entry.trackIdx);
        usedDets.insert(entry.detIdx);
    }

    return matches;
}

// ==============================================================================
// Hungarian Algorithm Implementation (Kuhn-Munkres)
// ==============================================================================
std::vector<std::pair<int, int>> ObjectTracker::hungarianAssignment(
    const std::vector<std::vector<float>>& costMatrix) const {

    if (costMatrix.empty()) return {};

    // Use Munkres O(N^3) algorithm for optimal assignment
    std::vector<int> assignment;
    HungarianAlgorithm::solve(costMatrix, assignment);

    // Convert assignment vector to matched pairs, filtering by IoU threshold
    std::vector<std::pair<int, int>> matches;
    int numDets = static_cast<int>(costMatrix[0].size());

    for (int i = 0; i < static_cast<int>(assignment.size()); i++) {
        int j = assignment[i];
        if (j < 0 || j >= numDets) continue;

        float iou = 1.0f - costMatrix[i][j];
        if (iou >= config_.iouThreshold) {
            matches.push_back({i, j});
        }
    }

    return matches;
}

// ==============================================================================
// Track Management
// ==============================================================================
void ObjectTracker::createTrack(const Detection& det) {
    auto track = std::make_unique<Track>(det, nextTrackId_++);
    tracks_.push_back(std::move(track));
}

void ObjectTracker::pruneDeletedTracks() {
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const std::unique_ptr<Track>& t) {
                           return t->isDeleted();
                       }),
        tracks_.end());
}

std::vector<Track*> ObjectTracker::getActiveTracks() const {
    std::vector<Track*> active;
    for (const auto& t : tracks_) {
        if (t->isConfirmed()) {
            active.push_back(t.get());
        }
    }
    return active;
}

std::vector<Track*> ObjectTracker::getAllTracks() const {
    std::vector<Track*> all;
    for (const auto& t : tracks_) {
        if (!t->isDeleted()) {
            all.push_back(t.get());
        }
    }
    return all;
}

Track* ObjectTracker::getTrack(int trackId) const {
    for (const auto& t : tracks_) {
        if (t->getId() == trackId && !t->isDeleted()) {
            return t.get();
        }
    }
    return nullptr;
}

void ObjectTracker::reset() {
    tracks_.clear();
    nextTrackId_ = 0;
}

void ObjectTracker::setConfig(const TrackerConfig& config) {
    config_ = config;
    Track::setMaxLost(config.maxLost);
    Track::setMinHits(config.minHits);
}

} // namespace fcw
