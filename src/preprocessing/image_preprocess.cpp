// ==============================================================================
// Image Preprocessing Implementation
// ==============================================================================

#include "image_preprocess.h"
#include <opencv2/imgproc.hpp>

namespace fcw {

ImagePreprocessor::ImagePreprocessor() {}

ImagePreprocessor::ImagePreprocessor(const PreprocessConfig& config)
    : config_(config) {}

LetterboxInfo ImagePreprocessor::preprocess(const cv::Mat& input, cv::Mat& output) {
    cv::Mat resized;
    LetterboxInfo info;

    if (config_.letterbox) {
        info = letterboxResize(input, resized, config_.targetWidth, config_.targetHeight);
    } else {
        cv::resize(input, resized, cv::Size(config_.targetWidth, config_.targetHeight));
        info.scale = static_cast<float>(config_.targetWidth) / input.cols;
        info.padLeft = 0;
        info.padTop = 0;
        info.newWidth = config_.targetWidth;
        info.newHeight = config_.targetHeight;
    }

    // BGR to RGB if needed
    if (config_.swapRB) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    // Convert to float32 and normalize to [0, 1]
    if (config_.normalize) {
        resized.convertTo(output, CV_32FC3, 1.0 / 255.0);
    } else {
        resized.convertTo(output, CV_32FC3);
    }

    return info;
}

void ImagePreprocessor::remapCoordinates(float& x1, float& y1, float& x2, float& y2,
                                          const LetterboxInfo& info) const {
    // Remove padding offset and scale back to original coordinates
    x1 = (x1 - info.padLeft) / info.scale;
    y1 = (y1 - info.padTop) / info.scale;
    x2 = (x2 - info.padLeft) / info.scale;
    y2 = (y2 - info.padTop) / info.scale;
}

void ImagePreprocessor::setConfig(const PreprocessConfig& config) {
    config_ = config;
}

LetterboxInfo ImagePreprocessor::letterboxResize(const cv::Mat& input, cv::Mat& output,
                                                  int targetW, int targetH) const {
    LetterboxInfo info;
    int imgW = input.cols;
    int imgH = input.rows;

    // Compute scale to fit within target while maintaining aspect ratio
    float scaleW = static_cast<float>(targetW) / imgW;
    float scaleH = static_cast<float>(targetH) / imgH;
    info.scale = std::min(scaleW, scaleH);

    info.newWidth = static_cast<int>(imgW * info.scale);
    info.newHeight = static_cast<int>(imgH * info.scale);

    // Compute padding
    info.padLeft = (targetW - info.newWidth) / 2;
    info.padTop = (targetH - info.newHeight) / 2;
    int padRight = targetW - info.newWidth - info.padLeft;
    int padBottom = targetH - info.newHeight - info.padTop;

    // Resize
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(info.newWidth, info.newHeight));

    // Add padding (gray border = 114)
    cv::copyMakeBorder(resized, output, info.padTop, padBottom,
                       info.padLeft, padRight,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return info;
}

} // namespace fcw
