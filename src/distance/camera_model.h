#pragma once
// ==============================================================================
// Camera Model - Pinhole camera geometry for distance estimation
// ==============================================================================

#include <opencv2/core.hpp>
#include <cmath>

namespace fcw {

/**
 * Pinhole camera model for monocular distance estimation.
 * 
 * The key relationship:
 *   distance = (focal_length * real_height) / pixel_height
 * 
 * This is derived from similar triangles in the pinhole camera model:
 *   pixel_height / focal_length = real_height / distance
 */
struct CameraModel {
    // Focal lengths (pixels)
    float fx = 721.5377f;
    float fy = 721.5377f;

    // Principal point (pixels)
    float cx = 609.5593f;
    float cy = 172.854f;

    // Camera mounting height (meters above ground)
    float mountHeight = 1.65f;

    // Camera pitch angle (radians, positive = looking down)
    float pitchAngle = 0.0f;

    /**
     * Estimate distance using pinhole model and known object height.
     * 
     * Formula: D = (f_y * H_real) / h_pixel
     * 
     * Where:
     *   D = distance to object (meters)
     *   f_y = focal length in y direction (pixels)
     *   H_real = known real-world height of object (meters)
     *   h_pixel = height of object in image (pixels)
     *
     * @param bboxHeightPixels  Height of bounding box in pixels
     * @param realHeightMeters  Known real height of the object (typically ~1.5m for cars)
     * @return Estimated distance in meters
     */
    float estimateDistance(float bboxHeightPixels, float realHeightMeters) const {
        if (bboxHeightPixels <= 0.0f) return -1.0f;
        return (fy * realHeightMeters) / bboxHeightPixels;
    }

    /**
     * Estimate distance using ground plane geometry.
     * 
     * Uses the bottom of the bounding box (contact point with ground)
     * and the camera's height + pitch to estimate distance.
     *
     * Formula:
     *   D = H_cam / tan(alpha + pitch)
     *   where alpha = atan((bottom_y - cy) / fy)
     *
     * @param bottomY  Y coordinate of bbox bottom edge (pixels)
     * @return Estimated distance in meters
     */
    float estimateDistanceGroundPlane(float bottomY) const {
        float alpha = std::atan2(bottomY - cy, fy);
        float totalAngle = alpha + pitchAngle;
        if (totalAngle <= 0.0f) return -1.0f;  // Object above horizon
        return mountHeight / std::tan(totalAngle);
    }

    /** Get 3x3 intrinsic matrix */
    cv::Mat getK() const {
        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        return K;
    }
};

} // namespace fcw
