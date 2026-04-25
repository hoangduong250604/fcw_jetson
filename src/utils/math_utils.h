#pragma once
// ==============================================================================
// Math Utilities for FCW System
// ==============================================================================

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

// Define M_PI if not already defined (for cross-platform compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace fcw {
namespace utils {

// ------------------------------------------------
// Bounding Box Utilities
// ------------------------------------------------
struct BBox {
    float x1, y1, x2, y2;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float centerX() const { return (x1 + x2) / 2.0f; }
    float centerY() const { return (y1 + y2) / 2.0f; }
    float area() const { return width() * height(); }
    cv::Rect toRect() const {
        return cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                        static_cast<int>(width()), static_cast<int>(height()));
    }
};

/**
 * Compute Intersection over Union (IoU) between two bounding boxes.
 * Critical for NMS and tracker association.
 */
inline float computeIoU(const BBox& a, const BBox& b) {
    float interX1 = std::max(a.x1, b.x1);
    float interY1 = std::max(a.y1, b.y1);
    float interX2 = std::min(a.x2, b.x2);
    float interY2 = std::min(a.y2, b.y2);

    float interArea = std::max(0.0f, interX2 - interX1) *
                      std::max(0.0f, interY2 - interY1);

    float unionArea = a.area() + b.area() - interArea;

    return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
}

// ------------------------------------------------
// Distance/Geometry Utilities
// ------------------------------------------------

/** Euclidean distance between two 2D points */
inline float euclideanDist(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

/** Clamp value between min and max */
template <typename T>
inline T clamp(T value, T minVal, T maxVal) {
    return std::max(minVal, std::min(value, maxVal));
}

// ------------------------------------------------
// Signal Processing / Smoothing
// ------------------------------------------------

/**
 * Exponential Moving Average (EMA) filter.
 * Used for smoothing distance/speed/TTC estimates.
 *   smoothed = alpha * new_value + (1 - alpha) * prev_smoothed
 */
inline float ema(float newVal, float prevSmoothed, float alpha) {
    return alpha * newVal + (1.0f - alpha) * prevSmoothed;
}

/**
 * Simple Moving Average over a window.
 */
inline float movingAverage(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : values) sum += v;
    return sum / static_cast<float>(values.size());
}

// ------------------------------------------------
// Conversion
// ------------------------------------------------

/** Degrees to radians */
inline float degToRad(float degrees) {
    return degrees * static_cast<float>(M_PI) / 180.0f;
}

/** Radians to degrees */
inline float radToDeg(float radians) {
    return radians * 180.0f / static_cast<float>(M_PI);
}

/** m/s to km/h */
inline float msToKmh(float ms) {
    return ms * 3.6f;
}

/** km/h to m/s */
inline float kmhToMs(float kmh) {
    return kmh / 3.6f;
}

} // namespace utils
} // namespace fcw
