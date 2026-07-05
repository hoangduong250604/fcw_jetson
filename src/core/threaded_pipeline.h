#pragma once
// ==============================================================================
// ThreadedPipeline - Multi-threaded FCW pipeline for Jetson Nano
// ==============================================================================
// Inspired by open-adas multi-threaded architecture.
//
// Thread architecture:
//   Thread 1 (Capture):     Camera → frame buffer
//   Thread 2 (Detection):   Frame → YOLOv8 detect → Track → Distance → TTC → Risk
//   Thread 3 (Display):     Visualization + Warning + Output
//
// Threads communicate through FCWState (mutex-protected shared state).
// This allows capture to run at full camera FPS while detection
// runs at its own (slower) rate without blocking the display.
// ==============================================================================

#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>

#include "pipeline.h"
#include "fcw_state.h"

namespace fcw {

struct ThreadedPipelineConfig {
    PipelineConfig baseConfig;
    bool enableThreading = true;    // Use threaded mode
    int captureQueueSize = 2;       // Frame buffer size
};

class ThreadedPipeline {
public:
    ThreadedPipeline();
    ~ThreadedPipeline();

    /**
     * Initialize with config. Sets up all modules.
     */
    bool init(const ThreadedPipelineConfig& config);

    /**
     * Load config from YAML files, then initialize.
     */
    bool loadAndInit(const std::string& systemConfigPath,
                     const std::string& cameraConfigPath = "",
                     const std::string& warningConfigPath = "");

    /**
     * Run the multi-threaded pipeline.
     * Blocks until stop() is called or input ends.
     */
    void run();

    /**
     * Stop all threads gracefully.
     */
    void stop();

    bool isRunning() const { return running_.load(); }

private:
    // Thread entry points
    void captureThread();
    void processingThread();
    void displayThread();

    // Helper for display thread drawing without Track* pointers
    void drawFromSnapshots(cv::Mat& frame,
                           const std::vector<FCWState::TrackSnapshot>& snapshots,
                           const std::unordered_map<int, DistanceInfo>& distances,
                           const std::unordered_map<int, SpeedInfo>& speeds,
                           const std::unordered_map<int, TTCInfo>& ttcs,
                           const std::unordered_map<int, RiskAssessment>& risks,
                           double fps);

    // Modules (owned by this class)
    Camera camera_;
    YOLOv8Detector detector_;
    ObjectTracker tracker_;
    DistanceEstimator distanceEstimator_;
    SpeedEstimator speedEstimator_;
    TTCCalculator ttcCalculator_;
    CollisionRisk riskAssessor_;
    WarningSystem warningSystem_;
    Visualization visualization_;
    utils::Timer timer_;

    // Shared state
    FCWState state_;

    // Frame buffer with condition variable
    cv::Mat frameBuffer_;
    std::mutex frameBufferMutex_;
    std::condition_variable frameReady_;
    bool hasNewFrame_ = false;

    // Threads
    std::thread captureThread_;
    std::thread processingThread_;
    std::thread displayThread_;

    // Config
    ThreadedPipelineConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> initialized_{false};

    // Video output
    cv::VideoWriter videoWriter_;
};

} // namespace fcw
