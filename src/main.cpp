// ==============================================================================
// FCW System - Forward Collision Warning System
// ==============================================================================
// Vision-Based FCW Using YOLOv8 Detection & TTC Estimation on Jetson Nano
//
// Usage:
//   ./fcw_system --config config/system_config.yaml
//   ./fcw_system --video path/to/video.mp4
//   ./fcw_system --camera 0
//
// Author: KLTN Project
// ==============================================================================

#include "pipeline.h"
#include "threaded_pipeline.h"
#include "logger.h"

#include <iostream>
#include <string>
#include <fstream>

// Auto-detect best model: prefer .engine (TensorRT) over .onnx
static std::string findModelPath() {
    const char* candidates[] = {
        "./models/yolov8s.engine",
        "./models/yolov8n.engine",
        "./models/yolov8s.onnx",
        "./models/yolov8n.onnx"
    };
    for (const auto& path : candidates) {
        std::ifstream f(path);
        if (f.good()) return path;
    }
    return "./models/yolov8s.onnx";
}

void printUsage(const char* progName) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Forward Collision Warning System\n";
    std::cout << "  YOLOv8 + TTC on Jetson Nano\n";
    std::cout << "========================================\n";
    std::cout << "\n";
    std::cout << "Usage:\n";
    std::cout << "  " << progName << " --config <system_config.yaml> [--camera-config <camera_config.yaml>] [--warning-config <warning_config.yaml>]\n";
    std::cout << "  " << progName << " --video <path_to_video>\n";
    std::cout << "  " << progName << " --camera <device_id> [--usb]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --config, -c       Path to system_config.yaml\n";
    std::cout << "  --camera-config    Path to camera_config.yaml\n";
    std::cout << "  --warning-config   Path to warning_config.yaml\n";
    std::cout << "  --video, -v        Direct path to input video\n";
    std::cout << "  --camera           Camera device ID (default: 0, CSI by default)\n";
    std::cout << "  --usb              Use USB camera instead of CSI\n";
    std::cout << "  --threaded         Enable multi-threaded pipeline\n";
    std::cout << "  --help, -h         Show this help message\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " --camera 0                  # CSI camera on Jetson Nano\n";
    std::cout << "  " << progName << " --camera 0 --usb            # USB webcam\n";
    std::cout << "  " << progName << " --camera 0 --threaded       # CSI + multi-threaded\n";
    std::cout << "  " << progName << " --video test.mp4            # Video file\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    // Debug trace file to verify execution
    {
        std::ofstream dbg("debug_trace.txt");
        dbg << "FCW main() started. argc=" << argc << std::endl;
        for (int i = 0; i < argc; i++) dbg << "  argv[" << i << "]=" << argv[i] << std::endl;
        dbg.flush();
    }

    // Parse command line arguments
    std::string configPath;
    std::string cameraConfigPath;
    std::string warningConfigPath;
    std::string videoPath;
    int cameraId = -1;
    bool useThreaded = false;
    bool useUSB = false;
    std::string oxtsFolder;
    std::string kittiRoot;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            configPath = argv[++i];
        } else if (arg == "--camera-config" && i + 1 < argc) {
            cameraConfigPath = argv[++i];
        } else if (arg == "--warning-config" && i + 1 < argc) {
            warningConfigPath = argv[++i];
        } else if ((arg == "--video" || arg == "-v") && i + 1 < argc) {
            videoPath = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            cameraId = std::stoi(argv[++i]);
        } else if (arg == "--usb") {
            useUSB = true;
        } else if (arg == "--threaded") {
            useThreaded = true;
        } else if (arg == "--oxts" && i + 1 < argc) {
            oxtsFolder = argv[++i];
        } else if (arg == "--kitti-root" && i + 1 < argc) {
            kittiRoot = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Initialize logger
    std::cout << "[DEBUG] Logger init..." << std::endl;
    fcw::utils::Logger::getInstance().init("./results/logs/system.log",
                                            fcw::utils::LogLevel::INFO);
    LOG_INFO("Main", "FCW System starting...");
    std::cout << "[DEBUG] Creating pipeline..." << std::endl;
    fcw::Pipeline pipeline;

    if (!configPath.empty()) {
        // Load from YAML config files
        if (!pipeline.loadConfig(configPath, cameraConfigPath, warningConfigPath)) {
            LOG_FATAL("Main", "Failed to load configuration");
            return 1;
        }

        // Initialize pipeline with the loaded config
        if (!pipeline.initFromLoadedConfig()) {
            LOG_FATAL("Main", "Failed to initialize pipeline from config");
            return 1;
        }
    } else if (!videoPath.empty()) {
        // Direct video mode
        std::cout << "[DEBUG] Video mode: " << videoPath << std::endl;
        fcw::PipelineConfig config;
        config.inputType = "video";
        config.inputSource = videoPath;
        config.detectorConfig.modelPath = findModelPath();
        config.detectorConfig.labelsPath = "./models/labels.txt";
        config.oxtsDataFolder = oxtsFolder;
        config.kittiRoot = kittiRoot;

        {
            std::ofstream dbg("debug_trace.txt", std::ios::app);
            dbg << "Before pipeline.init() video=" << videoPath << std::endl;
            dbg << "Model=" << config.detectorConfig.modelPath << std::endl;
            dbg.flush();
        }

        if (!pipeline.init(config)) {
            std::ofstream dbg("debug_trace.txt", std::ios::app);
            dbg << "pipeline.init() FAILED" << std::endl;
            LOG_FATAL("Main", "Failed to initialize pipeline");
            return 1;
        }

        {
            std::ofstream dbg("debug_trace.txt", std::ios::app);
            dbg << "pipeline.init() SUCCESS" << std::endl;
            dbg.flush();
        }
    } else if (cameraId >= 0) {
        // Camera mode
        fcw::PipelineConfig config;
        config.inputType = "camera";
        config.inputSource = std::to_string(cameraId);
        config.cameraType = useUSB ? "usb" : "csi";
        config.detectorConfig.modelPath = findModelPath();
        config.detectorConfig.labelsPath = "./models/labels.txt";
        LOG_INFO("Main", "Camera mode - model: " + config.detectorConfig.modelPath);

        if (!pipeline.init(config)) {
            LOG_FATAL("Main", "Failed to initialize pipeline");
            return 1;
        }
    } else {
        // Default: use config files
        configPath = "./config/system_config.yaml";
        cameraConfigPath = "./config/camera_config.yaml";
        warningConfigPath = "./config/warning_config.yaml";

        if (!pipeline.loadConfig(configPath, cameraConfigPath, warningConfigPath)) {
            printUsage(argv[0]);
            LOG_FATAL("Main", "No valid input specified and default config not found");
            return 1;
        }
    }

    // Run the pipeline
    if (useThreaded) {
        // Multi-threaded mode
        LOG_INFO("Main", "Starting multi-threaded FCW pipeline...");
        fcw::ThreadedPipeline threadedPipeline;

        if (!configPath.empty()) {
            if (!threadedPipeline.loadAndInit(configPath, cameraConfigPath, warningConfigPath)) {
                LOG_FATAL("Main", "Failed to initialize threaded pipeline");
                return 1;
            }
        } else if (!videoPath.empty()) {
            fcw::ThreadedPipelineConfig tConfig;
            tConfig.baseConfig.inputType = "video";
            tConfig.baseConfig.inputSource = videoPath;
            tConfig.baseConfig.detectorConfig.modelPath = findModelPath();
            tConfig.baseConfig.detectorConfig.labelsPath = "./models/labels.txt";
            if (!threadedPipeline.init(tConfig)) {
                LOG_FATAL("Main", "Failed to initialize threaded pipeline");
                return 1;
            }
        } else if (cameraId >= 0) {
            fcw::ThreadedPipelineConfig tConfig;
            tConfig.baseConfig.inputType = "camera";
            tConfig.baseConfig.inputSource = std::to_string(cameraId);
            tConfig.baseConfig.cameraType = useUSB ? "usb" : "csi";
            tConfig.baseConfig.detectorConfig.modelPath = findModelPath();
            tConfig.baseConfig.detectorConfig.labelsPath = "./models/labels.txt";
            LOG_INFO("Main", "Threaded camera mode - model: " + tConfig.baseConfig.detectorConfig.modelPath);
            if (!threadedPipeline.init(tConfig)) {
                LOG_FATAL("Main", "Failed to initialize threaded pipeline");
                return 1;
            }
        } else {
            if (!threadedPipeline.loadAndInit(
                    "./config/system_config.yaml",
                    "./config/camera_config.yaml",
                    "./config/warning_config.yaml")) {
                LOG_FATAL("Main", "Failed to initialize threaded pipeline");
                return 1;
            }
        }
        threadedPipeline.run();
    } else {
        // Single-threaded mode (original pipeline)
        LOG_INFO("Main", "Starting single-threaded FCW pipeline...");
        pipeline.run();
    }

    LOG_INFO("Main", "FCW System terminated normally");
    return 0;
}
