#pragma once
// ==============================================================================
// Pipeline - Main processing pipeline orchestrator
// ==============================================================================
// Orchestrates the full FCW processing chain:
//
//   Camera → Preprocess → Detect → Track → Distance → Speed → TTC → Risk → Warn → Viz
//
// Each frame goes through all stages sequentially.
// The pipeline manages all module instances and configuration.
// ==============================================================================

#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "camera.h"
#include "image_preprocess.h"
#include "yolov8_detector.h"
#include "object_tracker.h"
#include "distance_estimator.h"
#include "speed_estimator.h"
#include "kitti_oxts_reader.h"
#include "ttc_calculator.h"
#include "collision_risk.h"
#include "warning_system.h"
#include "visualization.h"
#include "timer.h"

namespace fcw {

struct PipelineConfig {
    // Input
    std::string inputType = "video";     // "camera", "video", "image_dir"
    std::string inputSource;
    std::string cameraType = "csi";      // "csi" or "usb"
    int inputWidth = 1280;
    int inputHeight = 720;

    // CSI camera settings (Jetson Nano)
    int csiSensorId = 0;
    int csiCaptureWidth = 1280;
    int csiCaptureHeight = 720;
    int csiFps = 30;
    int csiFlipMethod = 0;

    // KITTI OXTS ground truth (for ego speed)
    std::string kittiRoot;               // Path to KITTI/ folder (auto-detect OXTS)
    std::string oxtsDataFolder;          // Direct path to oxts/data/ folder (override)

    // Modules enable/disable
    bool enableDetection = true;
    bool enableTracking = true;
    bool enableDistance = true;
    bool enableTTC = true;
    bool enableWarning = true;
    bool enableVisualization = true;

    // Output
    bool saveVideo = false;
    std::string videoOutputPath;
    bool saveLog = true;
    std::string logPath = "./results/logs/system.log";

    // Sub-module configs
    DetectorConfig detectorConfig;
    TrackerConfig trackerConfig;
    DistanceConfig distanceConfig;
    SpeedConfig speedConfig;
    TTCConfig ttcConfig;
    RiskConfig riskConfig;
    WarningConfig warningConfig;
    VisConfig visConfig;
    CameraModel cameraModel;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    /**
     * Initialize all modules from config.
     */
    bool init(const PipelineConfig& config);

    /**
     * Initialize pipeline using config already loaded by loadConfig().
     */
    bool initFromLoadedConfig();

    /**
     * Load configuration from YAML files.
     */
    bool loadConfig(const std::string& systemConfigPath,
                    const std::string& cameraConfigPath = "",
                    const std::string& warningConfigPath = "");

    /**
     * Run the pipeline (main loop).
     * Processes frames until input ends or user quits.
     */
    void run();

    /**
     * Process a single frame through the entire pipeline.
     * @return false if no more frames available
     */
    bool processFrame();

    /** Stop the pipeline */
    void stop();

    /** Check if pipeline is running */
    bool isRunning() const { return running_; }

    /** Get current config (for ThreadedPipeline) */
    const PipelineConfig& getConfig() const { return config_; }

private:
    PipelineConfig config_;
    bool initialized_ = false;
    bool running_ = false;
    int frameCount_ = 0;

    // Modules
    Camera camera_;
    YOLOv8Detector detector_;
    ObjectTracker tracker_;
    DistanceEstimator distanceEstimator_;
    SpeedEstimator speedEstimator_;
    KittiOxtsReader oxtsReader_;
    TTCCalculator ttcCalculator_;
    CollisionRisk riskAssessor_;
    WarningSystem warningSystem_;
    Visualization visualization_;
    utils::Timer timer_;

    // Video writer for output
    cv::VideoWriter videoWriter_;
};

} // namespace fcw
