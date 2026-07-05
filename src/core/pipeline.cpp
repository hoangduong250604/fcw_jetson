// ==============================================================================
// Pipeline Implementation - Full FCW processing chain
// ==============================================================================
//
// Frame processing order:
//   1. Capture frame from camera/video
//   2. Run YOLOv8 detection → DetectionResult
//   3. Update tracker → list of Track*
//   4. Estimate distance per track → DistanceInfo
//   5. Estimate relative speed → SpeedInfo
//   6. Calculate TTC → TTCInfo
//   7. Assess collision risk → RiskAssessment
//   8. Trigger warning if needed
//   9. Draw visualization overlay
//   10. Display / save output
// ==============================================================================

#include "pipeline.h"
#include "logger.h"

#include <opencv2/highgui.hpp>
#ifdef HAVE_YAML_CPP
  #include <yaml-cpp/yaml.h>
#endif
#include <iostream>
#include <algorithm>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace fcw {

Pipeline::Pipeline() {}

Pipeline::~Pipeline() {
    stop();
}

// ==============================================================================
// Configuration Loading
// ==============================================================================
bool Pipeline::loadConfig(const std::string& systemConfigPath,
                           const std::string& cameraConfigPath,
                           const std::string& warningConfigPath) {
    (void)cameraConfigPath;   // Suppress unused parameter warning
    (void)warningConfigPath;  // Suppress unused parameter warning
    
#ifdef HAVE_YAML_CPP
    try {
        LOG_INFO("Pipeline", "Loading config: " + systemConfigPath);
        YAML::Node sysConfig = YAML::LoadFile(systemConfigPath);

        // System settings
        auto sys = sysConfig["system"];
        config_.inputType = sys["input"]["type"].as<std::string>("video");
        config_.inputSource = sys["input"]["source"].as<std::string>("");
        config_.inputWidth = sys["input"]["width"].as<int>(1280);
        config_.inputHeight = sys["input"]["height"].as<int>(720);
        config_.saveVideo = sys["output"]["save_video"].as<bool>(false);
        config_.videoOutputPath = sys["output"]["video_path"].as<std::string>("./results/videos/output.avi");
        config_.logPath = sys["output"]["log_path"].as<std::string>("./results/logs/system.log");

        // Pipeline enables
        auto pipeline = sys["pipeline"];
        config_.enableDetection = pipeline["enable_detection"].as<bool>(true);
        config_.enableTracking = pipeline["enable_tracking"].as<bool>(true);
        config_.enableDistance = pipeline["enable_distance"].as<bool>(true);
        config_.enableTTC = pipeline["enable_ttc"].as<bool>(true);
        config_.enableWarning = pipeline["enable_warning"].as<bool>(true);
        config_.enableVisualization = pipeline["enable_visualization"].as<bool>(true);

        // Performance: detect_interval (clamped >=1 to avoid a modulo-by-zero
        // in processFrame(); fallback of 2 matches this member's own
        // pre-existing default from before this was wired up)
        config_.detectInterval = std::max(1, sys["performance"]["detect_interval"].as<int>(2));

        // Detection config
        auto det = sysConfig["detection"];
        config_.detectorConfig.modelPath = det["model_path"].as<std::string>("./models/yolov8n.engine");
        config_.detectorConfig.labelsPath = det["labels_path"].as<std::string>("./models/labels.txt");
        config_.detectorConfig.inputWidth = det["input_width"].as<int>(640);
        config_.detectorConfig.inputHeight = det["input_height"].as<int>(640);
        config_.detectorConfig.confThreshold = det["conf_threshold"].as<float>(0.45f);
        config_.detectorConfig.nmsThreshold = det["nms_threshold"].as<float>(0.50f);
        config_.detectorConfig.maxDetections = det["max_detections"].as<int>(100);
        config_.detectorConfig.useFP16 = det["use_fp16"].as<bool>(true);

        auto targetClasses = det["target_classes"];
        config_.detectorConfig.targetClasses.clear();
        for (const auto& cls : targetClasses) {
            config_.detectorConfig.targetClasses.push_back(cls.as<int>());
        }

        // Tracking config
        auto trk = sysConfig["tracking"];
        config_.trackerConfig.maxDistance = trk["max_distance"].as<float>(100.0f);
        config_.trackerConfig.maxLost = trk["max_lost"].as<int>(30);
        config_.trackerConfig.minHits = trk["min_hits"].as<int>(3);
        config_.trackerConfig.iouThreshold = trk["iou_threshold"].as<float>(0.3f);
        config_.trackerConfig.useKalman = trk["use_kalman"].as<bool>(true);

        // Distance config
        auto dist = sysConfig["distance"];
        std::string distMethod = dist["method"].as<std::string>("bbox_height");
        if (distMethod == "bbox_height") config_.distanceConfig.method = DistanceMethod::BBOX_HEIGHT;
        else if (distMethod == "ground_plane") config_.distanceConfig.method = DistanceMethod::GROUND_PLANE;
        else if (distMethod == "bev") config_.distanceConfig.method = DistanceMethod::BEV;
        else config_.distanceConfig.method = DistanceMethod::COMBINED;
        config_.distanceConfig.referenceHeight = dist["reference_height"].as<float>(1.5f);
        config_.distanceConfig.maxDistance = dist["max_distance"].as<float>(100.0f);
        config_.distanceConfig.minDistance = dist["min_distance"].as<float>(2.0f);

        // Camera config
        if (!cameraConfigPath.empty()) {
            YAML::Node camConfig = YAML::LoadFile(cameraConfigPath);
            auto cam = camConfig["camera"];
            config_.cameraModel.fx = cam["intrinsic"]["fx"].as<float>(721.5377f);
            config_.cameraModel.fy = cam["intrinsic"]["fy"].as<float>(721.5377f);
            config_.cameraModel.cx = cam["intrinsic"]["cx"].as<float>(609.5593f);
            config_.cameraModel.cy = cam["intrinsic"]["cy"].as<float>(172.854f);
            config_.cameraModel.mountHeight = cam["mounting"]["height"].as<float>(1.65f);
            config_.cameraModel.pitchAngle = cam["mounting"]["pitch_angle"].as<float>(0.0f)
                                              * static_cast<float>(M_PI) / 180.0f;

            // Load CSI camera settings for Jetson Nano
            if (cam["csi"]) {
                config_.csiSensorId = cam["csi"]["sensor_id"].as<int>(0);
                config_.csiCaptureWidth = cam["csi"]["capture_width"].as<int>(1280);
                config_.csiCaptureHeight = cam["csi"]["capture_height"].as<int>(720);
                config_.csiFps = cam["csi"]["framerate"].as<int>(30);
                config_.csiFlipMethod = cam["csi"]["flip_method"].as<int>(0);
            }
        }

        // Warning config
        if (!warningConfigPath.empty()) {
            YAML::Node warnConfig = YAML::LoadFile(warningConfigPath);
            auto warn = warnConfig["warning"];
            config_.riskConfig.criticalTTC = warn["ttc"]["critical_threshold"].as<float>(1.5f);
            config_.riskConfig.dangerTTC = warn["ttc"]["danger_threshold"].as<float>(3.0f);
            config_.riskConfig.cautionTTC = warn["ttc"]["caution_threshold"].as<float>(5.0f);
            config_.riskConfig.enableSmoothing = warn["risk"]["enable_smoothing"].as<bool>(true);
            config_.riskConfig.smoothingWindow = warn["risk"]["smoothing_window"].as<int>(5);

            config_.warningConfig.audioEnabled = warn["audio"]["enabled"].as<bool>(true);
            config_.warningConfig.soundsDir = warn["audio"]["sounds_dir"].as<std::string>(config_.warningConfig.soundsDir);
            config_.warningConfig.criticalSound = warn["audio"]["critical_sound"].as<std::string>(config_.warningConfig.criticalSound);
            config_.warningConfig.dangerSound = warn["audio"]["danger_sound"].as<std::string>(config_.warningConfig.dangerSound);
            // No caution_sound: CAUTION intentionally plays no audio (see warning_system.cpp)

            if (warn["visual"]) {
                auto vis = warn["visual"];
                config_.visConfig.overlayAlpha = vis["overlay_alpha"].as<float>(config_.visConfig.overlayAlpha);
                config_.visConfig.showDistance = vis["show_distance"].as<bool>(config_.visConfig.showDistance);
                config_.visConfig.showTTC = vis["show_ttc"].as<bool>(config_.visConfig.showTTC);
                config_.visConfig.showSpeed = vis["show_speed"].as<bool>(config_.visConfig.showSpeed);
                auto readColor = [&](const char* key, cv::Scalar fallback) -> cv::Scalar {
                    auto node = vis[key];
                    if (node && node.size() == 3) {
                        return cv::Scalar(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
                    }
                    return fallback;
                };
                config_.visConfig.colorSafe = readColor("color_safe", config_.visConfig.colorSafe);
                config_.visConfig.colorCaution = readColor("color_caution", config_.visConfig.colorCaution);
                config_.visConfig.colorDanger = readColor("color_danger", config_.visConfig.colorDanger);
                config_.visConfig.colorCritical = readColor("color_critical", config_.visConfig.colorCritical);
            }
        }

        // Traffic sign/light secondary detector config
        if (sysConfig["traffic_sign"]) {
            auto ts = sysConfig["traffic_sign"];
            config_.trafficSignConfig.enabled = ts["enabled"].as<bool>(true);
            config_.trafficSignConfig.modelPath = ts["model_path"].as<std::string>("./models/yolov8n_traffic.onnx");
            config_.trafficSignConfig.labelsPath = ts["labels_path"].as<std::string>("./models/traffic_labels.txt");
            config_.trafficSignConfig.inputSize = ts["input_size"].as<int>(320);
            config_.trafficSignConfig.confThreshold = ts["conf_threshold"].as<float>(0.40f);
            config_.trafficSignConfig.processInterval = ts["process_interval"].as<int>(3);
        }

        LOG_INFO("Pipeline", "Configuration loaded successfully");
        return true;

    } catch (const YAML::Exception& e) {
        LOG_ERROR("Pipeline", "Config loading failed: " + std::string(e.what()));
        return false;
    }
#else
    // Fallback when yaml-cpp is not available - use default configuration
    LOG_WARNING("Pipeline", "yaml-cpp not available - using default configuration");
    LOG_WARNING("Pipeline", "Ignoring config files: " + systemConfigPath);
    return true;  // Continue with default config
#endif
}

// ==============================================================================
// Initialization
// ==============================================================================
bool Pipeline::initFromLoadedConfig() {
    return init(config_);
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;
    LOG_INFO("Pipeline", "Initializing FCW Pipeline...");

    // Create output directories
    fs::create_directories("./results/logs");
    fs::create_directories("./results/videos");

    // Initialize logger
    utils::Logger::getInstance().init(config_.logPath);

    // ---- Open input source ----
    if (config_.inputType == "video") {
        if (!camera_.openVideo(config_.inputSource)) {
            LOG_ERROR("Pipeline", "Failed to open video source");
            return false;
        }
    } else if (config_.inputType == "camera") {
        bool cameraOk = false;
        if (config_.cameraType == "usb") {
            int devId = 0;
            if (!config_.inputSource.empty()) {
                try { devId = std::stoi(config_.inputSource); } catch (...) {}
            }
            LOG_INFO("Pipeline", "Opening USB camera device " + std::to_string(devId));
            cameraOk = camera_.openUSB(devId, config_.inputWidth, config_.inputHeight);
        } else {
            LOG_INFO("Pipeline", "Opening CSI camera sensor " + std::to_string(config_.csiSensorId));
            cameraOk = camera_.openCSI(config_.csiSensorId,
                                        config_.csiCaptureWidth,
                                        config_.csiCaptureHeight,
                                        config_.csiFps,
                                        config_.csiFlipMethod);
        }
        if (!cameraOk) {
            LOG_ERROR("Pipeline", "Failed to open camera");
            return false;
        }
    }

    // ---- Initialize detector ----
    if (config_.enableDetection) {
        if (!detector_.init(config_.detectorConfig)) {
            LOG_ERROR("Pipeline", "Failed to initialize detector");
            return false;
        }
    }

    // ---- Initialize tracker ----
    if (config_.enableTracking) {
        tracker_.setConfig(config_.trackerConfig);
    }

    // ---- Initialize distance estimator ----
    if (config_.enableDistance) {
        distanceEstimator_.setConfig(config_.distanceConfig);
        distanceEstimator_.setCameraModel(config_.cameraModel);
        distanceEstimator_.setImageSize(camera_.getWidth(), camera_.getHeight());
    }

    // ---- Initialize speed estimator ----
    SpeedConfig speedCfg;
    speedEstimator_.setConfig(speedCfg);

    // ---- Initialize KITTI OXTS reader (ego speed ground truth) ----
    if (!config_.oxtsDataFolder.empty()) {
        // Direct path specified via --oxts
        oxtsReader_.setFolder(config_.oxtsDataFolder);
        LOG_INFO("Pipeline", "OXTS direct path: " + config_.oxtsDataFolder);
    } else {
        // Auto-detect: match video to correct KITTI drive by frame count
        int totalFrames = camera_.getFrameCount();
        std::string kittiRoot = config_.kittiRoot.empty() ? "../KITTI" : config_.kittiRoot;

        if (fs::exists(kittiRoot)) {
            if (oxtsReader_.autoDetectFromVideo(config_.inputSource, kittiRoot, totalFrames)) {
                LOG_INFO("Pipeline", "OXTS matched drive: " + oxtsReader_.getDriveName() +
                         " → " + oxtsReader_.getFolder());
            } else {
                LOG_WARNING("Pipeline", "OXTS auto-detect failed, ego speed will be 0");
            }
        } else {
            LOG_WARNING("Pipeline", "KITTI root not found: " + kittiRoot);
        }
    }

    // ---- Initialize TTC ----
    if (config_.enableTTC) {
        ttcCalculator_.setConfig(config_.ttcConfig);
    }

    // ---- Initialize risk ----
    riskAssessor_.setConfig(config_.riskConfig);

    // ---- Initialize warning ----
    if (config_.enableWarning) {
        warningSystem_.setConfig(config_.warningConfig);
        warningSystem_.startThread();  // Start dedicated warning thread
    }

    // ---- Initialize visualization ----
    if (config_.enableVisualization) {
        visualization_.setConfig(config_.visConfig);
    }

    // ---- Initialize traffic sign/light processor ----
    trafficSignProcessor_.init(config_.trafficSignConfig);

    // ---- Video writer ----
    if (config_.saveVideo) {
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = camera_.getFPS() > 0 ? camera_.getFPS() : 15.0;
        videoWriter_.open(config_.videoOutputPath, fourcc, fps,
                          cv::Size(camera_.getWidth(), camera_.getHeight()));
        if (!videoWriter_.isOpened()) {
            LOG_WARNING("Pipeline", "Failed to open video writer");
        }
    }

    initialized_ = true;
    LOG_INFO("Pipeline", "FCW Pipeline initialized successfully!");
    return true;
}

// ==============================================================================
// Main Run Loop
// ==============================================================================
void Pipeline::run() {
    if (!initialized_) {
        LOG_ERROR("Pipeline", "Pipeline not initialized!");
        return;
    }

    running_ = true;
    LOG_INFO("Pipeline", "Starting FCW Pipeline...");

    while (running_) {
        if (!processFrame()) break;

        // Check for quit key
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {  // ESC or 'q'
            LOG_INFO("Pipeline", "User requested quit");
            break;
        }
    }

    // Print timing summary
    timer_.printSummary();

    // Cleanup
    stop();
}

// ==============================================================================
// Process Single Frame
// ==============================================================================
bool Pipeline::processFrame() {
    timer_.frameTick();
    frameCount_++;

    // ---- 1. Capture ----
    cv::Mat frame;
    {
        utils::ScopedTimer st(timer_, "capture");
        if (!camera_.read(frame) || frame.empty()) {
            LOG_INFO("Pipeline", "End of input (frame " + std::to_string(frameCount_) + ")");
            return false;
        }
    }

    // ---- 2. Detection (every frame by default; every Nth if detectInterval
    //         is explicitly configured > 1) ----
    DetectionResult detections;
    bool detectionRanThisFrame = false;
    if (config_.enableDetection) {
        bool shouldRunDetector = ((frameCount_ - 1) % config_.detectInterval == 0);
        if (shouldRunDetector) {
            utils::ScopedTimer st(timer_, "detection");
            try {
                detections = detector_.detect(frame);
            } catch (const std::exception& e) {
                LOG_ERROR("Pipeline", std::string("Detector threw, skipping this frame's detections: ") + e.what());
                detections = DetectionResult();
            }
            detections.frameId = frameCount_;
            detectionRanThisFrame = true;
        }
    }

    // ---- 3. Tracking ----
    std::vector<Track*> activeTracks;
    if (config_.enableTracking) {
        utils::ScopedTimer st(timer_, "tracking");
        if (config_.enableDetection && !detectionRanThisFrame) {
            // detect_interval > 1 and this frame's detector run was skipped:
            // advance tracks via Kalman-only prediction, not penalized as missed.
            activeTracks = tracker_.predictOnly();
        } else {
            // Detection fully disabled (existing behavior: tracks age out via
            // markMissed on empty detections) or the detector did run this frame.
            activeTracks = tracker_.update(detections);
        }
    }

    // ---- 4. Distance Estimation ----
    std::unordered_map<int, DistanceInfo> distances;
    if (config_.enableDistance && !activeTracks.empty()) {
        utils::ScopedTimer st(timer_, "distance");
        distances = distanceEstimator_.estimate(activeTracks);
    }

    // ---- 5. Speed Estimation (timestamp-based) ----
    float timestampMs = camera_.getPositionMs();
    std::unordered_map<int, SpeedInfo> speeds;

    // Set ego speed: fixed config → KITTI OXTS → fallback 50 km/h
    if (config_.fixedEgoSpeedKmh >= 0.0f) {
        speedEstimator_.setEgoSpeed(config_.fixedEgoSpeedKmh);
    } else {
        bool gotOxts = false;
        if (oxtsReader_.isEnabled()) {
            OxtsData oxtsData = oxtsReader_.readFrame(frameCount_ - 1);  // 0-indexed
            if (oxtsData.valid) {
                speedEstimator_.setEgoSpeed(oxtsData.vf * 3.6f);  // m/s → km/h
                speedEstimator_.setOxtsData(oxtsData);
                gotOxts = true;
            }
        }
        if (!gotOxts) {
            speedEstimator_.setEgoSpeed(50.0f);
        }
    }

    if (config_.enableDistance && !distances.empty()) {
        utils::ScopedTimer st(timer_, "speed");
        speeds = speedEstimator_.estimate(distances, timestampMs);
    }

    // ---- 6. TTC Calculation ----
    std::unordered_map<int, TTCInfo> ttcs;
    if (config_.enableTTC && !activeTracks.empty()) {
        utils::ScopedTimer st(timer_, "ttc");
        ttcs = ttcCalculator_.calculate(activeTracks, distances, speeds);
    }

    // ---- 7. Risk Assessment ----
    std::unordered_map<int, RiskAssessment> risks;
    if (config_.enableTTC && !ttcs.empty()) {
        utils::ScopedTimer st(timer_, "risk");
        risks = riskAssessor_.assess(ttcs, distances);
    }

    // ---- 8. Warning ----
    if (config_.enableWarning) {
        RiskAssessment highest = riskAssessor_.getHighestRisk();
        warningSystem_.trigger(highest);
    }

    // ---- Eval CSV export (per-frame per-track data) ----
    if (evalEnabled_ && evalLog_.is_open()) {
        static const char* classNames[] = {
            "person", "bike", "car", "motor", "rider",
            "bus", "train", "truck", "traffic_sign", "traffic_light"
        };
        float egoKmh = speedEstimator_.getEgoSpeedKmh();
        
        if (activeTracks.empty()) {
            evalLog_ << frameCount_ << ",-1,,,,,,,,,,,,," << egoKmh << ",," << std::endl;
        } else {
            for (const Track* track : activeTracks) {
                int id = track->getId();
                int cls = track->getClassId();
                const char* clsName = (cls >= 0 && cls < 10) ? classNames[cls] : "unknown";
                auto bbox = track->getBBox();
                
                float dist = 0.0f, latOff = 0.0f;
                bool inEgo = false;
                auto distIt = distances.find(id);
                if (distIt != distances.end()) {
                    dist = distIt->second.smoothedDistance;
                    latOff = distIt->second.lateralOffset;
                    inEgo = distIt->second.inEgoPath;
                }
                
                float closingMs = 0.0f;
                std::string vState = "UNKNOWN";
                auto speedIt = speeds.find(id);
                if (speedIt != speeds.end()) {
                    closingMs = speedIt->second.closingSpeedMs;
                    switch (speedIt->second.vehicleState) {
                        case VehicleState::SAME_DIRECTION: vState = "SAME_DIR"; break;
                        case VehicleState::STATIONARY: vState = "STATIONARY"; break;
                        case VehicleState::ONCOMING: vState = "ONCOMING"; break;
                        default: break;
                    }
                }
                
                float ttc = -1.0f;
                auto ttcIt = ttcs.find(id);
                if (ttcIt != ttcs.end() && ttcIt->second.valid) {
                    ttc = ttcIt->second.ttcSmoothed;
                }
                
                std::string riskStr = "SAFE";
                auto riskIt = risks.find(id);
                if (riskIt != risks.end()) {
                    switch (riskIt->second.level) {
                        case RiskLevel::CAUTION: riskStr = "CAUTION"; break;
                        case RiskLevel::DANGER: riskStr = "DANGER"; break;
                        case RiskLevel::CRITICAL: riskStr = "CRITICAL"; break;
                        default: break;
                    }
                }
                
                evalLog_ << frameCount_ << "," << id << "," << cls << "," << clsName << ","
                         << bbox.x1 << "," << bbox.y1 << "," << bbox.width() << "," << bbox.height() << ","
                         << dist << "," << closingMs << "," << ttc << "," << latOff << ","
                         << (inEgo ? 1 : 0) << "," << egoKmh << "," << vState << "," << riskStr
                         << std::endl;
            }
        }
    }

    // ---- 8b. Traffic Sign/Light Classification ----
    TrafficSignResult trafficResult;
    if (trafficSignProcessor_.isReady()) {
        trafficResult = trafficSignProcessor_.process(frame, detections, frameCount_);
    }

    // ---- 9. Visualization ----
    if (config_.enableVisualization) {
        utils::ScopedTimer st(timer_, "visualization");
        float egoSpeedKmh = speedEstimator_.getEgoSpeedKmh();
        visualization_.draw(frame, activeTracks, distances, speeds,
                            ttcs, risks, timer_.getFPS(), detections, egoSpeedKmh,
                            trafficResult);

        // Display: native aspect for video, 1024x600 for camera
        if (config_.inputType == "camera") {
            cv::Mat display;
            cv::resize(frame, display, cv::Size(1024, 600));
            cv::imshow("FCW System", display);
        } else {
            cv::imshow("FCW System", frame);
        }
    }

    // ---- 10. Save output ----
    if (config_.saveVideo && videoWriter_.isOpened()) {
        videoWriter_.write(frame);
    }

    // Log periodic stats
    if (frameCount_ % 100 == 0) {
        LOG_INFO("Pipeline", "Frame " + std::to_string(frameCount_) +
                 " | FPS: " + std::to_string(static_cast<int>(timer_.getFPS())) +
                 " | Tracks: " + std::to_string(activeTracks.size()) +
                 " | Det: " + std::to_string(detections.count()));
    }

    return true;
}

// ==============================================================================
// Stop
// ==============================================================================
void Pipeline::stop() {
    running_ = false;
    warningSystem_.stopThread();  // Stop warning thread
    camera_.release();
    detector_.cleanup();
    if (videoWriter_.isOpened()) {
        videoWriter_.release();
    }
    if (evalLog_.is_open()) {
        evalLog_.close();
    }
    cv::destroyAllWindows();
    LOG_INFO("Pipeline", "Pipeline stopped. Total frames: " + std::to_string(frameCount_));
}

// ==============================================================================
// Eval CSV Export
// ==============================================================================
void Pipeline::enableEvalLog(const std::string& csvPath) {
    fs::path p(csvPath);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    evalLog_.open(csvPath, std::ios::out | std::ios::trunc);
    if (evalLog_.is_open()) {
        evalEnabled_ = true;
        evalLog_ << "frame,track_id,class_id,class_name,bbox_x,bbox_y,bbox_w,bbox_h,"
                 << "distance_m,closing_speed_ms,ttc_s,lateral_offset_m,in_ego_path,"
                 << "ego_speed_kmh,vehicle_state,risk_level" << std::endl;
        LOG_INFO("Pipeline", "Eval CSV enabled: " + csvPath);
    } else {
        LOG_ERROR("Pipeline", "Failed to open eval CSV: " + csvPath);
    }
}

} // namespace fcw
