// ==============================================================================
// YOLOv8 Detector Implementation - Dual Backend (TensorRT / OpenCV DNN)
// ==============================================================================
// CORE MODULE: Vehicle detection using YOLOv8
//
// Two backends:
//   USE_TENSORRT: TensorRT inference on NVIDIA GPU (Jetson Nano)
//   Default:      OpenCV DNN inference (portable, CPU/GPU via OpenCV)
//
// This implementation handles the full detection pipeline:
//   1. Load model (TensorRT engine or ONNX via OpenCV DNN)
//   2. Run YOLOv8 inference
//   3. Postprocess outputs (decode, filter, NMS)
// ==============================================================================

#include "yolov8_detector.h"
#include "nms.h"
#include "logger.h"
#include "timer.h"

#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>

#ifdef USE_TENSORRT
#include <NvOnnxParser.h>
#endif

#include <opencv2/imgproc.hpp>

namespace fcw {

// ==============================================================================
// TRT Logger (TensorRT only)
// ==============================================================================
#ifdef USE_TENSORRT
void TRTLogger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG_FATAL("TensorRT", msg);
            break;
        case Severity::kERROR:
            LOG_ERROR("TensorRT", msg);
            break;
        case Severity::kWARNING:
            LOG_WARNING("TensorRT", msg);
            break;
        case Severity::kINFO:
            LOG_DEBUG("TensorRT", msg);
            break;
        default:
            break;
    }
}
#endif

// ==============================================================================
// Constructor / Destructor
// ==============================================================================
YOLOv8Detector::YOLOv8Detector() {}

YOLOv8Detector::~YOLOv8Detector() {
    cleanup();
}

// ==============================================================================
// Initialization
// ==============================================================================
bool YOLOv8Detector::init(const DetectorConfig& config) {
    config_ = config;
    LOG_INFO("Detector", "Initializing YOLOv8 detector...");
    LOG_INFO("Detector", "Model: " + config.modelPath);
    LOG_INFO("Detector", "Input size: " + std::to_string(config.inputWidth) + "x" +
             std::to_string(config.inputHeight));
    LOG_INFO("Detector", "Confidence threshold: " + std::to_string(config.confThreshold));
    LOG_INFO("Detector", "NMS threshold: " + std::to_string(config.nmsThreshold));

#ifdef USE_TENSORRT
    LOG_INFO("Detector", "Backend: TensorRT");

    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        LOG_ERROR("Detector", "Failed to create CUDA stream: " +
                  std::string(cudaGetErrorString(err)));
        return false;
    }

    // Load TensorRT engine
    std::string path = config.modelPath;
    bool loaded = false;

    if (path.substr(path.length() - 7) == ".engine" ||
        path.substr(path.length() - 5) == ".plan") {
        loaded = loadEngine(path);
    } else if (path.substr(path.length() - 5) == ".onnx") {
        loaded = buildEngineFromONNX(path);
        if (loaded) {
            std::string enginePath = path.substr(0, path.length() - 5) + ".engine";
            saveEngine(enginePath);
        }
    } else {
        LOG_ERROR("Detector", "Unsupported model format. Use .engine or .onnx");
        return false;
    }

    if (!loaded) {
        LOG_ERROR("Detector", "Failed to load model");
        return false;
    }

    // Allocate GPU buffers
    if (!allocateBuffers()) {
        LOG_ERROR("Detector", "Failed to allocate GPU buffers");
        return false;
    }
#elif defined(USE_ONNXRUNTIME)
    LOG_INFO("Detector", "Backend: ONNX Runtime");

    std::string path = config.modelPath;
    if (path.length() >= 5 && path.substr(path.length() - 5) == ".onnx") {
        if (!loadONNX(path)) {
            LOG_ERROR("Detector", "Failed to load ONNX model via ONNX Runtime");
            return false;
        }
    } else {
        LOG_ERROR("Detector", "ONNX Runtime backend requires .onnx model file");
        return false;
    }
#else
    LOG_INFO("Detector", "Backend: OpenCV DNN");

    std::string path = config.modelPath;
    if (path.length() >= 5 && path.substr(path.length() - 5) == ".onnx") {
        if (!loadONNX(path)) {
            LOG_ERROR("Detector", "Failed to load ONNX model via OpenCV DNN");
            return false;
        }
    } else {
        LOG_ERROR("Detector", "OpenCV DNN backend requires .onnx model file");
        return false;
    }
#endif

    // Setup preprocessor
    PreprocessConfig ppConfig;
    ppConfig.targetWidth = config.inputWidth;
    ppConfig.targetHeight = config.inputHeight;
    ppConfig.normalize = true;
    ppConfig.swapRB = true;
    ppConfig.letterbox = true;
    preprocessor_.setConfig(ppConfig);

    // Load class labels
    if (!config.labelsPath.empty()) {
        loadLabels(config.labelsPath);
    }

    initialized_ = true;
    LOG_INFO("Detector", "YOLOv8 detector initialized successfully!");
    LOG_INFO("Detector", "  Num classes: " + std::to_string(numClasses_));
    LOG_INFO("Detector", "  Num anchors: " + std::to_string(numAnchors_));

    return true;
}

// ==============================================================================
// Detection - Main entry point
// ==============================================================================
DetectionResult YOLOv8Detector::detect(const cv::Mat& frame) {
    DetectionResult result;
    if (!initialized_ || frame.empty()) return result;

    auto startTime = std::chrono::high_resolution_clock::now();

#ifdef USE_TENSORRT
    // TensorRT path
    LetterboxInfo lbInfo;
    preprocessFrame(frame, hostInput_.data(), lbInfo);

    cudaMemcpyAsync(buffers_[inputIndex_], hostInput_.data(),
                    inputSize_, cudaMemcpyHostToDevice, stream_);

    if (!executeInference(static_cast<float*>(buffers_[inputIndex_]),
                          static_cast<float*>(buffers_[outputIndex_]))) {
        LOG_ERROR("Detector", "Inference failed");
        return result;
    }

    cudaMemcpyAsync(hostOutput_.data(), buffers_[outputIndex_],
                    outputSize_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    result = postprocess(hostOutput_.data(), lbInfo, frame.cols, frame.rows);
#elif defined(USE_ONNXRUNTIME)
    // ONNX Runtime path
    LetterboxInfo lbInfo;

    // Preprocess
    cv::Mat preprocessed;
    lbInfo = preprocessor_.preprocess(frame, preprocessed);

    // Convert HWC -> CHW planar float
    int channels = 3, h = config_.inputHeight, w = config_.inputWidth;
    std::vector<float> inputTensor(channels * h * w);
    std::vector<cv::Mat> chw(channels);
    for (int c = 0; c < channels; c++) {
        chw[c] = cv::Mat(h, w, CV_32FC1, inputTensor.data() + c * h * w);
    }
    cv::split(preprocessed, chw);

    // Create ONNX Runtime tensor
    std::array<int64_t, 4> inputShape = {1, channels, h, w};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputOrt = Ort::Value::CreateTensor<float>(
        memInfo, inputTensor.data(), inputTensor.size(),
        inputShape.data(), inputShape.size());

    // Run inference
    auto outputTensors = ortSession_->Run(
        Ort::RunOptions{nullptr},
        inputNames_.data(), &inputOrt, 1,
        outputNames_.data(), outputNames_.size());

    // Get output [1, 84, 8400]
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    numClasses_ = static_cast<int>(outputShape[1]) - 4;
    numAnchors_ = static_cast<int>(outputShape[2]);

    result = postprocess(outputData, lbInfo, frame.cols, frame.rows);
#else
    // OpenCV DNN path
    LetterboxInfo lbInfo;

    // Letterbox resize
    cv::Mat preprocessed;
    lbInfo = preprocessor_.preprocess(frame, preprocessed);

    // Create blob from preprocessed image (already normalized, RGB)
    cv::Mat blob = cv::dnn::blobFromImage(preprocessed, 1.0, cv::Size(), cv::Scalar(), false, false);

    net_.setInput(blob);
    cv::Mat output = net_.forward();

    // Output shape: [1, 84, 8400] for YOLOv8
    int dims1 = output.size[1];  // 84
    int dims2 = output.size[2];  // 8400
    numClasses_ = dims1 - 4;
    numAnchors_ = dims2;

    result = postprocess((float*)output.data, lbInfo, frame.cols, frame.rows);
#endif

    auto endTime = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(
        endTime - startTime).count();

    return result;
}

// ==============================================================================
// ONNX Runtime Backend
// ==============================================================================
#ifdef USE_ONNXRUNTIME
bool YOLOv8Detector::loadONNX(const std::string& onnxPath) {
    LOG_INFO("Detector", "Loading ONNX model via ONNX Runtime: " + onnxPath);

    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(4);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Convert path to wide string for Windows
        ortSession_ = std::make_unique<Ort::Session>(ortEnv_, onnxPath.c_str(), sessionOptions);

        // Get input info
        size_t numInputs = ortSession_->GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            auto name = ortSession_->GetInputNameAllocated(i, ortAllocator_);
            inputNamesOwned_.push_back(name.get());
        }
        for (auto& s : inputNamesOwned_) inputNames_.push_back(s.c_str());

        // Get output info
        size_t numOutputs = ortSession_->GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            auto name = ortSession_->GetOutputNameAllocated(i, ortAllocator_);
            outputNamesOwned_.push_back(name.get());
        }
        for (auto& s : outputNamesOwned_) outputNames_.push_back(s.c_str());

        // Probe output shape
        auto outputShape = ortSession_->GetOutputTypeInfo(0)
            .GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.size() >= 3) {
            numClasses_ = static_cast<int>(outputShape[1]) - 4;
            numAnchors_ = static_cast<int>(outputShape[2]);
        }

        LOG_INFO("Detector", "ONNX Runtime model loaded: " +
                 std::to_string(numClasses_) + " classes, " +
                 std::to_string(numAnchors_) + " anchors");
        return true;

    } catch (const Ort::Exception& e) {
        LOG_ERROR("Detector", "ONNX Runtime error: " + std::string(e.what()));
        return false;
    }
}
#endif

// ==============================================================================
// OpenCV DNN Backend
// ==============================================================================
#ifndef USE_TENSORRT
#ifndef USE_ONNXRUNTIME
bool YOLOv8Detector::loadONNX(const std::string& onnxPath) {
    LOG_INFO("Detector", "Loading ONNX model via OpenCV DNN: " + onnxPath);

    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
    } catch (const cv::Exception& e) {
        LOG_ERROR("Detector", "Failed to read ONNX: " + std::string(e.what()));
        return false;
    }

    if (net_.empty()) {
        LOG_ERROR("Detector", "Network is empty after loading");
        return false;
    }

    // Use CPU backend
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    LOG_INFO("Detector", "Using OpenCV DNN backend (CPU)");

    // Probe the network to determine output dimensions
    cv::Mat dummy = cv::Mat::zeros(config_.inputHeight, config_.inputWidth, CV_32FC3);
    cv::Mat blob = cv::dnn::blobFromImage(dummy, 1.0, cv::Size(), cv::Scalar(), false, false);
    net_.setInput(blob);
    cv::Mat testOutput = net_.forward();

    numClasses_ = testOutput.size[1] - 4;
    numAnchors_ = testOutput.size[2];

    LOG_INFO("Detector", "ONNX model loaded: " + std::to_string(numClasses_) +
             " classes, " + std::to_string(numAnchors_) + " anchors");
    return true;
}
#endif  // USE_ONNXRUNTIME
#endif  // USE_TENSORRT

// ==============================================================================
// TensorRT Backend
// ==============================================================================
#ifdef USE_TENSORRT
bool YOLOv8Detector::loadEngine(const std::string& enginePath) {
    LOG_INFO("Detector", "Loading TensorRT engine: " + enginePath);

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR("Detector", "Engine file not found: " + enginePath);
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(trtLogger_));
    if (!runtime_) {
        LOG_ERROR("Detector", "Failed to create TensorRT runtime");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), size));
    if (!engine_) {
        LOG_ERROR("Detector", "Failed to deserialize engine");
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR("Detector", "Failed to create execution context");
        return false;
    }

    LOG_INFO("Detector", "Engine loaded successfully. Bindings: " +
#if NV_TENSORRT_MAJOR >= 10
             std::to_string(engine_->getNbIOTensors()));
#else
             std::to_string(engine_->getNbBindings()));
#endif
    return true;
}

bool YOLOv8Detector::buildEngineFromONNX(const std::string& onnxPath) {
    LOG_INFO("Detector", "Building TensorRT engine from ONNX: " + onnxPath);
    LOG_INFO("Detector", "This may take several minutes on Jetson Nano...");

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(trtLogger_));
    if (!builder) {
        LOG_ERROR("Detector", "Failed to create IBuilder");
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(explicitBatch));
    if (!network) {
        LOG_ERROR("Detector", "Failed to create network");
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, trtLogger_));
    if (!parser->parseFromFile(onnxPath.c_str(),
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        LOG_ERROR("Detector", "Failed to parse ONNX model");
        return false;
    }

    auto builderConfig = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());

    if (config_.useFP16) {
        builderConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        LOG_INFO("Detector", "FP16 inference enabled");
    }

    engine_.reset(builder->buildEngineWithConfig(*network, *builderConfig));
    if (!engine_) {
        LOG_ERROR("Detector", "Failed to build TensorRT engine");
        return false;
    }

    runtime_.reset(nvinfer1::createInferRuntime(trtLogger_));
    context_.reset(engine_->createExecutionContext());

    LOG_INFO("Detector", "Engine built successfully!");
    return true;
}

bool YOLOv8Detector::saveEngine(const std::string& enginePath) {
    if (!engine_) return false;

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(engine_->serialize());
    std::ofstream file(enginePath, std::ios::binary);
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    file.close();

    LOG_INFO("Detector", "Engine saved to: " + enginePath);
    return true;
}

// ==============================================================================
// Buffer Management (TensorRT)
// ==============================================================================
bool YOLOv8Detector::allocateBuffers() {
#if NV_TENSORRT_MAJOR >= 10
    // TensorRT 10+ API
    nvinfer1::Dims inputDims = engine_->getTensorShape(engine_->getIOTensorName(0));
    batchSize_ = inputDims.d[0];
    int channels = inputDims.d[1];
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];

    inputSize_ = batchSize_ * channels * inputH * inputW * sizeof(float);
    hostInput_.resize(batchSize_ * channels * inputH * inputW);

    nvinfer1::Dims outputDims = engine_->getTensorShape(engine_->getIOTensorName(1));
    int outputDim1 = outputDims.d[1];
    int outputDim2 = outputDims.d[2];
#else
    // TensorRT 7/8 API (Jetson Nano JetPack 4.x)
    inputIndex_ = engine_->getBindingIndex("images");
    outputIndex_ = engine_->getBindingIndex("output0");
    // Fallback: if named bindings not found, use index 0 and 1
    if (inputIndex_ < 0) inputIndex_ = 0;
    if (outputIndex_ < 0) outputIndex_ = 1;

    nvinfer1::Dims inputDims = engine_->getBindingDimensions(inputIndex_);
    batchSize_ = (inputDims.d[0] > 0) ? inputDims.d[0] : 1;
    int channels = inputDims.d[1];
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];

    inputSize_ = batchSize_ * channels * inputH * inputW * sizeof(float);
    hostInput_.resize(batchSize_ * channels * inputH * inputW);

    nvinfer1::Dims outputDims = engine_->getBindingDimensions(outputIndex_);
    int outputDim1 = outputDims.d[1];
    int outputDim2 = outputDims.d[2];
#endif

    numClasses_ = outputDim1 - 4;
    numAnchors_ = outputDim2;

    outputSize_ = batchSize_ * outputDim1 * outputDim2 * sizeof(float);
    hostOutput_.resize(batchSize_ * outputDim1 * outputDim2);

    cudaError_t err;
    err = cudaMalloc(&buffers_[0], inputSize_);
    if (err != cudaSuccess) {
        LOG_ERROR("Detector", "Failed to allocate input GPU buffer");
        return false;
    }

    err = cudaMalloc(&buffers_[1], outputSize_);
    if (err != cudaSuccess) {
        LOG_ERROR("Detector", "Failed to allocate output GPU buffer");
        cudaFree(buffers_[0]);
        return false;
    }

    LOG_INFO("Detector", "GPU buffers allocated: input=" +
             std::to_string(inputSize_ / 1024) + "KB, output=" +
             std::to_string(outputSize_ / 1024) + "KB");
    return true;
}

void YOLOv8Detector::freeBuffers() {
    if (buffers_[0]) { cudaFree(buffers_[0]); buffers_[0] = nullptr; }
    if (buffers_[1]) { cudaFree(buffers_[1]); buffers_[1] = nullptr; }
}

// ==============================================================================
// Inference (TensorRT)
// ==============================================================================
bool YOLOv8Detector::executeInference(float* inputBuffer, float* outputBuffer) {
    (void)inputBuffer;
    (void)outputBuffer;

#if NV_TENSORRT_MAJOR >= 10
    // TensorRT 10+ API: must set tensor addresses explicitly
    const char* inputName = engine_->getIOTensorName(0);
    const char* outputName = engine_->getIOTensorName(1);
    context_->setTensorAddress(inputName, buffers_[inputIndex_]);
    context_->setTensorAddress(outputName, buffers_[outputIndex_]);
    bool success = context_->enqueueV3(stream_);
#else
    // TensorRT 7/8 API (Jetson Nano)
    bool success = context_->enqueueV2(buffers_, stream_, nullptr);
#endif
    if (!success) {
        LOG_ERROR("Detector", "TensorRT inference failed");
        return false;
    }
    return true;
}
#endif  // USE_TENSORRT

// ==============================================================================
// Preprocessing (shared)
// ==============================================================================
void YOLOv8Detector::preprocessFrame(const cv::Mat& frame, float* inputBuffer,
                                      LetterboxInfo& lbInfo) {
    cv::Mat preprocessed;
    lbInfo = preprocessor_.preprocess(frame, preprocessed);

    int channels = 3;
    int height = config_.inputHeight;
    int width = config_.inputWidth;

    std::vector<cv::Mat> chw(channels);
    for (int c = 0; c < channels; c++) {
        chw[c] = cv::Mat(height, width, CV_32FC1,
                         inputBuffer + c * height * width);
    }
    cv::split(preprocessed, chw);
}

// ==============================================================================
// Postprocessing (shared)
// ==============================================================================
DetectionResult YOLOv8Detector::postprocess(float* outputBuffer,
                                             const LetterboxInfo& lbInfo,
                                             int originalWidth, int originalHeight) {
    DetectionResult result;

    std::vector<Detection> candidates = decodeOutput(outputBuffer, numAnchors_, numClasses_);

    std::vector<Detection> nmsResult = applyNMS(candidates, config_.nmsThreshold);

    for (auto& det : nmsResult) {
        preprocessor_.remapCoordinates(det.bbox.x1, det.bbox.y1,
                                        det.bbox.x2, det.bbox.y2, lbInfo);

        det.bbox.x1 = utils::clamp(det.bbox.x1, 0.0f, static_cast<float>(originalWidth));
        det.bbox.y1 = utils::clamp(det.bbox.y1, 0.0f, static_cast<float>(originalHeight));
        det.bbox.x2 = utils::clamp(det.bbox.x2, 0.0f, static_cast<float>(originalWidth));
        det.bbox.y2 = utils::clamp(det.bbox.y2, 0.0f, static_cast<float>(originalHeight));

        if (det.classId < static_cast<int>(labels_.size())) {
            det.className = labels_[det.classId];
        } else {
            det.className = "class_" + std::to_string(det.classId);
        }

        result.detections.push_back(det);
    }

    if (static_cast<int>(result.detections.size()) > config_.maxDetections) {
        std::sort(result.detections.begin(), result.detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });
        result.detections.resize(config_.maxDetections);
    }

    return result;
}

std::vector<Detection> YOLOv8Detector::decodeOutput(float* output,
                                                      int numAnchors, int numClasses) {
    std::vector<Detection> detections;
    detections.reserve(numAnchors);

    // YOLOv8 output shape: [1, (4 + numClasses), numAnchors]
    int stride = numAnchors;

    for (int i = 0; i < numAnchors; i++) {
        float cx = output[0 * stride + i];
        float cy = output[1 * stride + i];
        float w  = output[2 * stride + i];
        float h  = output[3 * stride + i];

        float maxScore = -1.0f;
        int bestClass = -1;

        for (int c = 0; c < numClasses; c++) {
            float score = output[(4 + c) * stride + i];
            if (score > maxScore) {
                maxScore = score;
                bestClass = c;
            }
        }

        if (maxScore < config_.confThreshold) continue;

        bool isTarget = false;
        for (int tc : config_.targetClasses) {
            if (bestClass == tc) { isTarget = true; break; }
        }
        if (!isTarget) continue;

        Detection det;
        det.bbox.x1 = cx - w / 2.0f;
        det.bbox.y1 = cy - h / 2.0f;
        det.bbox.x2 = cx + w / 2.0f;
        det.bbox.y2 = cy + h / 2.0f;
        det.confidence = maxScore;
        det.classId = bestClass;

        detections.push_back(det);
    }

    return detections;
}

// ==============================================================================
// Labels (shared)
// ==============================================================================
bool YOLOv8Detector::loadLabels(const std::string& labelsPath) {
    std::ifstream file(labelsPath);
    if (!file.is_open()) {
        LOG_WARNING("Detector", "Labels file not found: " + labelsPath);
        return false;
    }

    labels_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }

    LOG_INFO("Detector", "Loaded " + std::to_string(labels_.size()) + " class labels");
    return true;
}

// ==============================================================================
// Cleanup
// ==============================================================================
void YOLOv8Detector::cleanup() {
#ifdef USE_TENSORRT
    freeBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    context_.reset();
    engine_.reset();
    runtime_.reset();
#elif defined(USE_ONNXRUNTIME)
    ortSession_.reset();
    inputNames_.clear();
    outputNames_.clear();
    inputNamesOwned_.clear();
    outputNamesOwned_.clear();
#else
    net_ = cv::dnn::Net();
#endif
    initialized_ = false;
    LOG_INFO("Detector", "Detector resources released");
}

} // namespace fcw
