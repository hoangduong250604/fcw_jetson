// ==============================================================================
// ThreadedPipeline Implementation - Multi-threaded FCW Pipeline
// ==============================================================================
// Three-thread architecture for maximizing throughput on Jetson Nano:
//
//   Capture Thread:    Grabs frames at camera FPS, stores latest frame
//   Processing Thread: Takes latest frame → detect → track → TTC → risk
//   Display Thread:    Reads latest results → visualize → show/save
//
// This decouples frame capture from heavy detection computation,
// ensuring the system never drops camera frames while waiting for
// the neural network to finish.
// ==============================================================================

#include "threaded_pipeline.h"
#include "logger.h"

#include <opencv2/highgui.hpp>
#ifdef HAVE_YAML_CPP
  #include <yaml-cpp/yaml.h>
#endif
#include <experimental/filesystem>
#include <chrono>

namespace fs = std::experimental::filesystem;

namespace fcw {

ThreadedPipeline::ThreadedPipeline() {}

ThreadedPipeline::~ThreadedPipeline() {
    stop();
}

// ==============================================================================
// Initialization
// ==============================================================================
bool ThreadedPipeline::loadAndInit(const std::string& systemConfigPath,
                                    const std::string& cameraConfigPath,
                                    const std::string& warningConfigPath) {
    // Use Pipeline's config loader to parse YAML
    Pipeline tempPipeline;
    if (!tempPipeline.loadConfig(systemConfigPath, cameraConfigPath, warningConfigPath)) {
        return false;
    }

    ThreadedPipelineConfig tConfig;
    tConfig.baseConfig = tempPipeline.getConfig();
    return init(tConfig);
}

bool ThreadedPipeline::init(const ThreadedPipelineConfig& config) {
    config_ = config;
    const auto& cfg = config.baseConfig;

    LOG_INFO("ThreadedPipeline", "Initializing multi-threaded FCW pipeline...");

    fs::create_directories("./results/logs");
    fs::create_directories("./results/videos");

    utils::Logger::getInstance().init(cfg.logPath);

    // Open input
    if (cfg.inputType == "video") {
        if (!camera_.openVideo(cfg.inputSource)) {
            LOG_ERROR("ThreadedPipeline", "Failed to open video source");
            return false;
        }
    } else if (cfg.inputType == "camera") {
        if (!camera_.openCSI()) {
            LOG_ERROR("ThreadedPipeline", "Failed to open camera");
            return false;
        }
    }

    // Initialize detector
    if (cfg.enableDetection) {
        if (!detector_.init(cfg.detectorConfig)) {
            LOG_ERROR("ThreadedPipeline", "Failed to initialize detector");
            return false;
        }
    }

    // Initialize other modules
    tracker_.setConfig(cfg.trackerConfig);
    distanceEstimator_.setConfig(cfg.distanceConfig);
    distanceEstimator_.setCameraModel(cfg.cameraModel);

    SpeedConfig speedCfg;
    speedEstimator_.setConfig(speedCfg);

    ttcCalculator_.setConfig(cfg.ttcConfig);
    riskAssessor_.setConfig(cfg.riskConfig);
    warningSystem_.setConfig(cfg.warningConfig);
    warningSystem_.startThread();  // Start dedicated warning thread
    visualization_.setConfig(cfg.visConfig);

    // Video writer
    if (cfg.saveVideo) {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = camera_.getFPS() > 0 ? camera_.getFPS() : 15.0;
        videoWriter_.open(cfg.videoOutputPath, fourcc, fps,
                          cv::Size(camera_.getWidth(), camera_.getHeight()));
    }

    initialized_.store(true);
    LOG_INFO("ThreadedPipeline", "Multi-threaded pipeline initialized!");
    return true;
}

// ==============================================================================
// Main Run
// ==============================================================================
void ThreadedPipeline::run() {
    if (!initialized_.load()) {
        LOG_ERROR("ThreadedPipeline", "Pipeline not initialized!");
        return;
    }

    running_.store(true);
    state_.reset();

    LOG_INFO("ThreadedPipeline", "Starting 3-thread pipeline...");

    // Launch threads
    captureThread_ = std::thread(&ThreadedPipeline::captureThread, this);
    processingThread_ = std::thread(&ThreadedPipeline::processingThread, this);
    displayThread_ = std::thread(&ThreadedPipeline::displayThread, this);

    // Wait for all threads to finish
    if (captureThread_.joinable()) captureThread_.join();
    if (processingThread_.joinable()) processingThread_.join();
    if (displayThread_.joinable()) displayThread_.join();

    timer_.printSummary();
    stop();
}

// ==============================================================================
// Capture Thread
// ==============================================================================
void ThreadedPipeline::captureThread() {
    LOG_INFO("ThreadedPipeline", "Capture thread started");

    while (running_.load()) {
        cv::Mat frame;
        if (!camera_.read(frame) || frame.empty()) {
            LOG_INFO("ThreadedPipeline", "End of input stream");
            state_.requestStop();
            running_.store(false);
            frameReady_.notify_all();
            break;
        }

        // Update shared state
        state_.setCurrentFrame(frame);

        // Signal frame buffer for processing thread
        {
            std::lock_guard<std::mutex> lock(frameBufferMutex_);
            frame.copyTo(frameBuffer_);
            hasNewFrame_ = true;
        }
        frameReady_.notify_one();
    }

    LOG_INFO("ThreadedPipeline", "Capture thread ended");
}

// ==============================================================================
// Processing Thread (Detection + Tracking + TTC)
// ==============================================================================
void ThreadedPipeline::processingThread() {
    LOG_INFO("ThreadedPipeline", "Processing thread started");
    const auto& cfg = config_.baseConfig;

    while (running_.load() && !state_.isStopRequested()) {
        // Wait for a new frame
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frameBufferMutex_);
            frameReady_.wait(lock, [this] {
                return hasNewFrame_ || !running_.load() || state_.isStopRequested();
            });
            if (!running_.load() || state_.isStopRequested()) break;
            frameBuffer_.copyTo(frame);
            hasNewFrame_ = false;
        }

        if (frame.empty()) continue;

        timer_.frameTick();

        // Step 1: Detection
        DetectionResult detections;
        if (cfg.enableDetection) {
            utils::ScopedTimer st(timer_, "detection");
            detections = detector_.detect(frame);
        }
        state_.setDetections(detections);
        state_.setDetectionTimeMs(detections.inferenceTimeMs);

        // Step 2: Tracking
        std::vector<Track*> activeTracks;
        if (cfg.enableTracking) {
            utils::ScopedTimer st(timer_, "tracking");
            activeTracks = tracker_.update(detections);
        }

        // Step 3: Distance
        std::unordered_map<int, DistanceInfo> distances;
        if (cfg.enableDistance && !activeTracks.empty()) {
            utils::ScopedTimer st(timer_, "distance");
            distances = distanceEstimator_.estimate(activeTracks);
        }
        state_.setDistances(distances);

        // Step 4: Speed (timestamp-based)
        float timestampMs = camera_.getPositionMs();
        std::unordered_map<int, SpeedInfo> speeds;
        if (cfg.enableDistance && !distances.empty()) {
            utils::ScopedTimer st(timer_, "speed");
            speeds = speedEstimator_.estimate(distances, timestampMs);
        }
        state_.setSpeeds(speeds);

        // Step 5: TTC
        std::unordered_map<int, TTCInfo> ttcs;
        if (cfg.enableTTC && !activeTracks.empty()) {
            utils::ScopedTimer st(timer_, "ttc");
            ttcs = ttcCalculator_.calculate(activeTracks, distances, speeds);
        }
        state_.setTTCs(ttcs);

        // Step 6: Risk
        std::unordered_map<int, RiskAssessment> risks;
        if (cfg.enableTTC && !ttcs.empty()) {
            utils::ScopedTimer st(timer_, "risk");
            risks = riskAssessor_.assess(ttcs, distances);
        }
        state_.setRisks(risks);
        state_.setHighestRisk(riskAssessor_.getHighestRisk());

        // Store track snapshots
        state_.setTrackSnapshots(activeTracks);
        state_.setFPS(timer_.getFPS());
    }

    LOG_INFO("ThreadedPipeline", "Processing thread ended");
}

// ==============================================================================
// Display Thread (Visualization + Warning)
// ==============================================================================
void ThreadedPipeline::displayThread() {
    LOG_INFO("ThreadedPipeline", "Display thread started");
    const auto& cfg = config_.baseConfig;
    int lastFrameId = -1;

    while (running_.load() && !state_.isStopRequested()) {
        int currentFrameId = state_.getFrameId();
        if (currentFrameId == lastFrameId) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        lastFrameId = currentFrameId;

        cv::Mat frame = state_.getCurrentFrame();
        if (frame.empty()) continue;

        // Warning
        if (cfg.enableWarning) {
            RiskAssessment highest = state_.getHighestRisk();
            if (highest.level > RiskLevel::SAFE) {
                warningSystem_.trigger(highest);
            }
        }

        // Visualization - need to reconstruct track pointers from snapshots
        // We draw directly using the snapshot data
        if (cfg.enableVisualization) {
            auto distances = state_.getDistances();
            auto speeds = state_.getSpeeds();
            auto ttcs = state_.getTTCs();
            auto risks = state_.getRisks();
            auto snapshots = state_.getTrackSnapshots();
            double fps = state_.getFPS();

            // Draw using snapshot data directly
            drawFromSnapshots(frame, snapshots, distances, speeds, ttcs, risks, fps);

            cv::imshow("FCW System", frame);
        }

        // Save video
        if (cfg.saveVideo && videoWriter_.isOpened()) {
            videoWriter_.write(frame);
        }

        // Check for quit key
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            state_.requestStop();
            running_.store(false);
            frameReady_.notify_all();
            break;
        }
    }

    LOG_INFO("ThreadedPipeline", "Display thread ended");
}

// ==============================================================================
// Helper: Draw from snapshots (no Track* needed)
// ==============================================================================
void ThreadedPipeline::drawFromSnapshots(
    cv::Mat& frame,
    const std::vector<FCWState::TrackSnapshot>& snapshots,
    const std::unordered_map<int, DistanceInfo>& distances,
    const std::unordered_map<int, SpeedInfo>& speeds,
    const std::unordered_map<int, TTCInfo>& ttcs,
    const std::unordered_map<int, RiskAssessment>& risks,
    double fps) {

    // Find highest risk for overlay
    RiskAssessment highestRisk;
    highestRisk.level = RiskLevel::SAFE;
    for (const auto& [id, ra] : risks) {
        if (ra.level > highestRisk.level) highestRisk = ra;
    }

    // Risk overlay
    if (highestRisk.level > RiskLevel::SAFE) {
        cv::Mat overlay = frame.clone();
        cv::Scalar color;
        switch (highestRisk.level) {
            case RiskLevel::CAUTION:  color = cv::Scalar(0, 255, 255); break;
            case RiskLevel::DANGER:   color = cv::Scalar(0, 165, 255); break;
            case RiskLevel::CRITICAL: color = cv::Scalar(0, 0, 255); break;
            default: color = cv::Scalar(0, 255, 0); break;
        }
        int barH = 30;
        cv::rectangle(overlay, cv::Rect(0, 0, frame.cols, barH), color, -1);
        cv::rectangle(overlay, cv::Rect(0, frame.rows - barH, frame.cols, barH), color, -1);
        cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);

        std::string text = "!! " + riskLevelToString(highestRisk.level) + " !!";
        int baseline;
        cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::putText(frame, text, cv::Point((frame.cols - sz.width) / 2, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    }

    // Per-track visualization
    for (const auto& snap : snapshots) {
        int id = snap.id;
        RiskLevel riskLevel = RiskLevel::SAFE;
        auto riskIt = risks.find(id);
        if (riskIt != risks.end()) riskLevel = riskIt->second.level;

        cv::Scalar color;
        switch (riskLevel) {
            case RiskLevel::SAFE:     color = cv::Scalar(0, 255, 0); break;
            case RiskLevel::CAUTION:  color = cv::Scalar(0, 255, 255); break;
            case RiskLevel::DANGER:   color = cv::Scalar(0, 165, 255); break;
            case RiskLevel::CRITICAL: color = cv::Scalar(0, 0, 255); break;
        }

        cv::Rect rect = snap.bbox.toRect();
        int thickness = (riskLevel >= RiskLevel::DANGER) ? 3 : 2;
        cv::rectangle(frame, rect, color, thickness);

        // ID label
        std::string idText = "ID:" + std::to_string(id);
        cv::putText(frame, idText, cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

        // Info labels
        int x = rect.x + rect.width + 5;
        int y = rect.y;
        int step = 18;

        auto distIt = distances.find(id);
        if (distIt != distances.end() && distIt->second.valid) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1fm", distIt->second.smoothedDistance);
            cv::putText(frame, buf, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            y += step;
        }

        // Speed calculation is kept but not displayed on bbox
        // Target speed will only be shown in ego speed panel

        auto ttcIt = ttcs.find(id);
        if (ttcIt != ttcs.end() && ttcIt->second.valid) {
            char buf[32];
            snprintf(buf, sizeof(buf), "TTC:%.1fs", ttcIt->second.ttcSmoothed);
            cv::putText(frame, buf, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            y += step;
        }

        if (riskLevel > RiskLevel::SAFE) {
            cv::putText(frame, riskLevelToString(riskLevel), cv::Point(x, y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
    }

    // FPS
    {
        char buf[32];
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(frame, buf, cv::Point(10, frame.rows - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

// ==============================================================================
// Stop
// ==============================================================================
void ThreadedPipeline::stop() {
    running_.store(false);
    state_.requestStop();
    frameReady_.notify_all();
    warningSystem_.stopThread();  // Stop warning thread

    if (captureThread_.joinable()) captureThread_.join();
    if (processingThread_.joinable()) processingThread_.join();
    if (displayThread_.joinable()) displayThread_.join();

    camera_.release();
    detector_.cleanup();
    if (videoWriter_.isOpened()) videoWriter_.release();
    cv::destroyAllWindows();

    LOG_INFO("ThreadedPipeline", "Pipeline stopped");
}

} // namespace fcw
