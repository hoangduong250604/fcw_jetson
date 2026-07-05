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

    /**
     * Bundled output of ONE processing-thread cycle: the exact frame that was
     * analyzed plus every downstream result computed from it. All fields are
     * published together under a single lock (setFrameResult/getFrameResult)
     * so a consumer (the display thread) can never see a mix of, e.g., the
     * newest captured frame paired with TTC/risk data from an older cycle —
     * a real correctness issue for a collision-warning HUD, since the
     * capture thread runs at full camera FPS while processing is
     * detection-bound and therefore slower.
     */
    struct FrameResult {
        int frameId = -1;
        cv::Mat frame;
        DetectionResult detections;
        std::vector<TrackSnapshot> trackSnapshots;
        std::unordered_map<int, DistanceInfo> distances;
        std::unordered_map<int, SpeedInfo> speeds;
        std::unordered_map<int, TTCInfo> ttcs;
        std::unordered_map<int, RiskAssessment> risks;
        RiskAssessment highestRisk;
        double fps = 0.0;
    };

    void setFrameResult(FrameResult&& result) {
        std::lock_guard<std::mutex> lock(resultMutex_);
        result_ = std::move(result);
    }

    FrameResult getFrameResult() const {
        std::lock_guard<std::mutex> lock(resultMutex_);
        return result_;
    }

    int getResultFrameId() const {
        std::lock_guard<std::mutex> lock(resultMutex_);
        return result_.frameId;
    }

    // ---- Pipeline Control ----
    void requestStop() { stopRequested_.store(true); }
    bool isStopRequested() const { return stopRequested_.load(); }
    void reset() { stopRequested_.store(false); }

private:
    mutable std::mutex resultMutex_;
    FrameResult result_;

    // Control
    std::atomic<bool> stopRequested_{false};
};

} // namespace fcw
