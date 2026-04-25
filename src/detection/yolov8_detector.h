#pragma once
// ==============================================================================
// YOLOv8 Detector - Vehicle detection with dual backend
// ==============================================================================
// 
// This is a CORE module of the FCW system.
// Supports two inference backends:
//   1. TensorRT (USE_TENSORRT) - optimized for Jetson Nano / NVIDIA GPU
//   2. OpenCV DNN (default)    - portable, works on any platform
//
// Pipeline: Load ONNX/Engine → Preprocess → Inference → Postprocess → NMS
//
// YOLOv8 output format:
//   For detection: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
//   bbox format: cx, cy, w, h (center-based)
// ==============================================================================

#include <string>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#ifdef USE_TENSORRT
// TensorRT headers
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#elif defined(USE_ONNXRUNTIME)
// MinGW GCC doesn't define _stdcall (uses __stdcall instead)
#if defined(__MINGW32__) && !defined(_stdcall)
#define _stdcall __stdcall
#endif
#include <onnxruntime_cxx_api.h>
#endif

#include "detection_result.h"
#include "image_preprocess.h"

namespace fcw {

struct DetectorConfig {
    std::string modelPath;          // Path to .engine or .onnx file
    std::string labelsPath;         // Path to labels.txt
    int inputWidth = 640;           // Model input width
    int inputHeight = 640;          // Model input height
    float confThreshold = 0.45f;    // Confidence threshold
    float nmsThreshold = 0.50f;     // NMS IoU threshold
    int maxDetections = 100;        // Max detections per frame
    std::vector<int> targetClasses = {0, 1, 2, 3, 5, 6, 7, 9};  // COCO: person, bicycle, car, motorcycle, bus, train, truck, traffic light
    bool useFP16 = true;            // Half precision
};

#ifdef USE_TENSORRT
/**
 * TensorRT Logger for YOLO engine.
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};
#endif

/**
 * YOLOv8 Detector with dual backend support.
 * 
 * Backend selection:
 *   - USE_TENSORRT defined: TensorRT inference (Jetson Nano / GPU)
 *   - Otherwise: OpenCV DNN inference (portable / CPU)
 *
 * Workflow:
 *   1. Load model (TensorRT engine or ONNX via OpenCV DNN)
 *   2. For each frame:
 *      a. Preprocess (letterbox, normalize, HWC→CHW)
 *      b. Run inference
 *      c. Postprocess (decode boxes, apply NMS)
 */
class YOLOv8Detector {
public:
    YOLOv8Detector();
    ~YOLOv8Detector();

    /**
     * Initialize detector with config.
     * Loads TensorRT engine and allocates GPU memory.
     */
    bool init(const DetectorConfig& config);

    /**
     * Run detection on a single frame.
     * 
     * @param frame  Input image (BGR, original size)
     * @return       Detection results with bounding boxes in original image coords
     */
    DetectionResult detect(const cv::Mat& frame);

    /** Check if detector is initialized */
    bool isInitialized() const { return initialized_; }

    /** Get model input size */
    cv::Size getInputSize() const { return cv::Size(config_.inputWidth, config_.inputHeight); }

    /** Get class labels */
    const std::vector<std::string>& getLabels() const { return labels_; }

    /** Release resources */
    void cleanup();

private:
    // ---- Engine Loading ----
    
#ifdef USE_TENSORRT
    /** Load a serialized TensorRT engine from file. */
    bool loadEngine(const std::string& enginePath);

    /** Build TensorRT engine from ONNX model. */
    bool buildEngineFromONNX(const std::string& onnxPath);

    /** Serialize engine to file for caching. */
    bool saveEngine(const std::string& enginePath);
#elif defined(USE_ONNXRUNTIME)
    /** Load ONNX model via ONNX Runtime. */
    bool loadONNX(const std::string& onnxPath);
#else
    /** Load ONNX model via OpenCV DNN. */
    bool loadONNX(const std::string& onnxPath);
#endif

    // ---- Memory Management ----
    
#ifdef USE_TENSORRT
    /** Allocate CUDA device memory for input/output buffers. */
    bool allocateBuffers();

    /** Free all CUDA buffers. */
    void freeBuffers();
#endif

    // ---- Inference Pipeline ----

    /**
     * Preprocess frame for YOLOv8:
     *   1. Letterbox resize to [inputW x inputH]
     *   2. BGR → RGB
     *   3. Normalize to [0, 1]
     *   4. HWC → CHW layout (for TensorRT)
     */
    void preprocessFrame(const cv::Mat& frame, float* inputBuffer, LetterboxInfo& lbInfo);

#ifdef USE_TENSORRT
    /**
     * Execute TensorRT inference.
     */
    bool executeInference(float* inputBuffer, float* outputBuffer);
#endif

    /**
     * Postprocess YOLOv8 output:
     *   1. Decode raw output tensor → candidate detections
     *   2. Filter by confidence threshold
     *   3. Filter by target classes
     *   4. Apply NMS to remove overlapping boxes
     *   5. Remap coordinates from letterbox space → original image space
     */
    DetectionResult postprocess(float* outputBuffer, const LetterboxInfo& lbInfo,
                                 int originalWidth, int originalHeight);

    /**
     * Decode YOLOv8 output tensor.
     * YOLOv8 output: [1, 84, 8400] → transpose to [8400, 84]
     * Each row: [cx, cy, w, h, class_scores...]
     */
    std::vector<Detection> decodeOutput(float* output, int numAnchors, int numClasses);

    /** Load class labels from file */
    bool loadLabels(const std::string& labelsPath);

    // ---- Member Variables ----
    DetectorConfig config_;
    bool initialized_ = false;

#ifdef USE_TENSORRT
    // TensorRT objects
    TRTLogger trtLogger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA buffers
    void* buffers_[2] = {nullptr, nullptr};  // [0]=input, [1]=output
    int inputIndex_ = 0;
    int outputIndex_ = 1;

    // CUDA stream for async operations
    cudaStream_t stream_ = nullptr;
#elif defined(USE_ONNXRUNTIME)
    // ONNX Runtime objects
    Ort::Env ortEnv_{ORT_LOGGING_LEVEL_WARNING, "fcw"};
    std::unique_ptr<Ort::Session> ortSession_;
    Ort::AllocatorWithDefaultOptions ortAllocator_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<std::string> inputNamesOwned_;
    std::vector<std::string> outputNamesOwned_;
#else
    // OpenCV DNN network
    cv::dnn::Net net_;
#endif

    // Model info
    int batchSize_ = 1;
    int numClasses_ = 80;       // COCO classes
    int numAnchors_ = 8400;     // YOLOv8 anchor points
    size_t inputSize_ = 0;      // Input buffer size in bytes
    size_t outputSize_ = 0;     // Output buffer size in bytes

    // Host buffers
    std::vector<float> hostInput_;
    std::vector<float> hostOutput_;

    // Class labels
    std::vector<std::string> labels_;

    // Preprocessor
    ImagePreprocessor preprocessor_;
};

} // namespace fcw
