#pragma once
// ==============================================================================
// FCWState - Thread-safe shared state for multi-threaded pipeline
// ==============================================================================
// Inspired by open-adas CarStatus pattern. Provides mutex-protected access
// to shared data between pipeline threads (capture, detection, tracking, etc).
//
// Each setter/getter acquires the appropriate mutex, ensuring data consistency
// when detection, tracking, and visualization run in parallel.
// ==============================================================================

#include <mutex>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "detection_result.h"
#include "track.h"
#include "distance_estimator.h"
#include "speed_estimator.h"
#include "ttc_calculator.h"
#include "risk_state.h"

namespace fcw {

class FCWState {
public:
    FCWState() = default;

    // ---- Frame ----
    void setCurrentFrame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(frameMutex_);
        frame.copyTo(currentFrame_);
        frameId_++;
    }

    cv::Mat getCurrentFrame() const {
        std::lock_guard<std::mutex> lock(frameMutex_);
        return currentFrame_.clone();
    }

    int getFrameId() const { return frameId_.load(); }

    // ---- Detection Results ----
    void setDetections(const DetectionResult& dets) {
        std::lock_guard<std::mutex> lock(detectionMutex_);
        detections_ = dets;
    }

    DetectionResult getDetections() const {
        std::lock_guard<std::mutex> lock(detectionMutex_);
        return detections_;
    }

    // ---- Active Track Data (snapshot) ----
    struct TrackSnapshot {
        int id;
        utils::BBox bbox;
        int classId;
        float confidence;
        cv::Point2f velocity;
        float scaleVelocity;
        TrackState state;
        int age;
        std::vector<utils::BBox> history;
    };

    void setTrackSnapshots(const std::vector<Track*>& tracks) {
        std::lock_guard<std::mutex> lock(trackMutex_);
        trackSnapshots_.clear();
        for (const Track* t : tracks) {
            TrackSnapshot snap;
            snap.id = t->getId();
            snap.bbox = t->getBBox();
            snap.classId = t->getClassId();
            snap.confidence = t->getConfidence();
            snap.velocity = t->getVelocity();
            snap.scaleVelocity = t->getScaleVelocity();
            snap.state = t->getState();
            snap.age = t->getAge();
            snap.history = t->getHistory();
            trackSnapshots_.push_back(snap);
        }
    }

    std::vector<TrackSnapshot> getTrackSnapshots() const {
        std::lock_guard<std::mutex> lock(trackMutex_);
        return trackSnapshots_;
    }

    // ---- Distance ----
    void setDistances(const std::unordered_map<int, DistanceInfo>& dists) {
        std::lock_guard<std::mutex> lock(distanceMutex_);
        distances_ = dists;
    }

    std::unordered_map<int, DistanceInfo> getDistances() const {
        std::lock_guard<std::mutex> lock(distanceMutex_);
        return distances_;
    }

    // ---- Speed ----
    void setSpeeds(const std::unordered_map<int, SpeedInfo>& speeds) {
        std::lock_guard<std::mutex> lock(speedMutex_);
        speeds_ = speeds;
    }

    std::unordered_map<int, SpeedInfo> getSpeeds() const {
        std::lock_guard<std::mutex> lock(speedMutex_);
        return speeds_;
    }

    // ---- TTC ----
    void setTTCs(const std::unordered_map<int, TTCInfo>& ttcs) {
        std::lock_guard<std::mutex> lock(ttcMutex_);
        ttcs_ = ttcs;
    }

    std::unordered_map<int, TTCInfo> getTTCs() const {
        std::lock_guard<std::mutex> lock(ttcMutex_);
        return ttcs_;
    }

    // ---- Risk ----
    void setRisks(const std::unordered_map<int, RiskAssessment>& risks) {
        std::lock_guard<std::mutex> lock(riskMutex_);
        risks_ = risks;
    }

    std::unordered_map<int, RiskAssessment> getRisks() const {
        std::lock_guard<std::mutex> lock(riskMutex_);
        return risks_;
    }

    void setHighestRisk(const RiskAssessment& risk) {
        std::lock_guard<std::mutex> lock(riskMutex_);
        highestRisk_ = risk;
    }

    RiskAssessment getHighestRisk() const {
        std::lock_guard<std::mutex> lock(riskMutex_);
        return highestRisk_;
    }

    // ---- Performance ----
    void setFPS(double fps) { fps_.store(fps); }
    double getFPS() const { return fps_.load(); }

    void setDetectionTimeMs(double ms) { detectionTimeMs_.store(ms); }
    double getDetectionTimeMs() const { return detectionTimeMs_.load(); }

    // ---- Pipeline Control ----
    void requestStop() { stopRequested_.store(true); }
    bool isStopRequested() const { return stopRequested_.load(); }
    void reset() { stopRequested_.store(false); frameId_.store(0); }

private:
    // Frame
    mutable std::mutex frameMutex_;
    cv::Mat currentFrame_;
    std::atomic<int> frameId_{0};

    // Detection
    mutable std::mutex detectionMutex_;
    DetectionResult detections_;

    // Tracking
    mutable std::mutex trackMutex_;
    std::vector<TrackSnapshot> trackSnapshots_;

    // Distance
    mutable std::mutex distanceMutex_;
    std::unordered_map<int, DistanceInfo> distances_;

    // Speed
    mutable std::mutex speedMutex_;
    std::unordered_map<int, SpeedInfo> speeds_;

    // TTC
    mutable std::mutex ttcMutex_;
    std::unordered_map<int, TTCInfo> ttcs_;

    // Risk
    mutable std::mutex riskMutex_;
    std::unordered_map<int, RiskAssessment> risks_;
    RiskAssessment highestRisk_;

    // Performance
    std::atomic<double> fps_{0.0};
    std::atomic<double> detectionTimeMs_{0.0};

    // Control
    std::atomic<bool> stopRequested_{false};
};

} // namespace fcw
