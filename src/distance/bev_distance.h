#pragma once
// ==============================================================================
// BEV Distance Estimator - Bird's Eye View perspective transform
// ==============================================================================
// Inspired by open-adas BirdViewModel.
// Uses perspective transform to map image points to real-world coordinates.
//
// Calibration uses 4 known ground points (e.g., from a carpet/marker):
//   Image points → BEV (top-down) → Real-world meters
//
// This provides an alternative distance estimation method that doesn't
// require object height assumptions, making it more robust for various
// vehicle types (trucks, motorcycles, etc).
// ==============================================================================

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <mutex>
#include <atomic>

namespace fcw {

struct BEVCalibration {
    // 4 calibration points in image space (normalized 0-1)
    cv::Point2f imageTL, imageTR, imageBR, imageBL;

    // Real-world dimensions (meters)
    float carWidth = 1.8f;          // Width of ego vehicle
    float carpetWidth = 3.0f;       // Width of calibration area
    float carToCarpetDist = 3.0f;   // Distance from car to near edge
    float carpetLength = 5.0f;      // Length of calibration area
};

class BEVDistanceEstimator {
public:
    BEVDistanceEstimator();
    
    // Disable copy and move - use by reference or pointer
    BEVDistanceEstimator(const BEVDistanceEstimator&) = delete;
    BEVDistanceEstimator& operator=(const BEVDistanceEstimator&) = delete;

    /**
     * Calibrate using 4 ground points.
     *
     * @param calib  Calibration data with image points and real dimensions
     */
    void calibrate(const BEVCalibration& calib);

    /**
     * Load calibration from file (open-adas format).
     */
    bool loadCalibration(const std::string& filePath);

    /**
     * Estimate distance to an object at a given bottom-center point (image coords).
     *
     * @param bottomCenter  Bottom-center of bounding box in image pixels
     * @param imageWidth    Image width in pixels
     * @param imageHeight   Image height in pixels
     * @return Distance in meters, -1 if not calibrated
     */
    float estimateDistance(const cv::Point2f& bottomCenter,
                          int imageWidth, int imageHeight) const;

    /**
     * Transform image points to bird's eye view coordinates.
     */
    void transformPoints(const std::vector<cv::Point2f>& imagePoints,
                         std::vector<cv::Point2f>& bevPoints,
                         int imageWidth, int imageHeight) const;

    /**
     * Get danger zone mask in image coordinates.
     * The danger zone extends forward from the ego vehicle by dangerDistance meters.
     *
     * @param imageSize      Output image size
     * @param dangerDistance  Distance in meters for danger zone
     * @return Binary mask of danger zone in image coordinates
     */
    cv::Mat getDangerZoneMask(const cv::Size& imageSize, float dangerDistance) const;

    bool isCalibrated() const { return calibrated_.load(); }

private:
    // BEV image dimensions (virtual top-down view)
    static constexpr int BEV_WIDTH = 1000;
    static constexpr int BEV_HEIGHT = 10000;

    // Calibration data
    BEVCalibration calib_;

    // 4 points in BEV space corresponding to calibration area
    cv::Point2f bevTL_, bevTR_, bevBR_, bevBL_;

    // Pixel-to-meter ratios
    float widthPixelToMeter_ = 0.1f;
    float heightPixelToMeter_ = 0.1f;

    // Car position in BEV pixels
    float carYPixel_ = 1000.0f;
    float carWidthPixel_ = 750.0f;

    // Transform matrices
    cv::Mat bev2imgMatrix_;
    cv::Mat img2bevMatrix_;

    std::atomic<bool> calibrated_{false};
    mutable std::mutex mutex_;
};

} // namespace fcw
