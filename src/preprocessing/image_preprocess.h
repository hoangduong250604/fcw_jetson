#pragma once
// ==============================================================================
// Image Preprocessing - Prepare frames for detection
// ==============================================================================

#include <opencv2/core.hpp>

namespace fcw {

struct PreprocessConfig {
    int targetWidth = 640;
    int targetHeight = 640;
    bool normalize = true;        // Normalize to [0,1]
    bool swapRB = true;           // BGR to RGB
    bool letterbox = true;        // Maintain aspect ratio with padding
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3] = {1.0f, 1.0f, 1.0f};
};

struct LetterboxInfo {
    float scale;
    int padLeft;
    int padTop;
    int newWidth;
    int newHeight;
};

class ImagePreprocessor {
public:
    ImagePreprocessor();
    explicit ImagePreprocessor(const PreprocessConfig& config);

    /**
     * Preprocess image for YOLOv8 inference.
     * Applies letterbox resize, color conversion, and normalization.
     * 
     * @param input   Raw camera frame (BGR)
     * @param output  Preprocessed image ready for inference
     * @return        LetterboxInfo for coordinate remapping
     */
    LetterboxInfo preprocess(const cv::Mat& input, cv::Mat& output);

    /**
     * Remap detection coordinates from model space back to original image space.
     */
    void remapCoordinates(float& x1, float& y1, float& x2, float& y2,
                          const LetterboxInfo& info) const;

    /** Set config */
    void setConfig(const PreprocessConfig& config);

private:
    /**
     * Letterbox resize: resize maintaining aspect ratio, pad with gray.
     */
    LetterboxInfo letterboxResize(const cv::Mat& input, cv::Mat& output,
                                   int targetW, int targetH) const;

    PreprocessConfig config_;
};

} // namespace fcw
