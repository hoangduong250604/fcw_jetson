// ==============================================================================
// BEV Distance Estimator Implementation
// ==============================================================================
// Bird's Eye View distance estimation using perspective transform.
// Based on open-adas BirdViewModel approach.
//
// Method:
//   1. Calibrate with 4 ground points + known dimensions
//   2. Compute perspective transform: image → BEV top-down view
//   3. For each object's bottom-center point, transform to BEV
//   4. Calculate distance from car position to object in BEV meters
//
// This avoids the need for object height assumptions (pinhole model).
// ==============================================================================

#include "bev_distance.h"
#include "logger.h"

#include <fstream>
#include <sstream>

namespace fcw {

BEVDistanceEstimator::BEVDistanceEstimator() {
    // Default BEV calibration area (centered in BEV image)
    bevTL_ = cv::Point2f(250.0f, 8000.0f);
    bevTR_ = cv::Point2f(750.0f, 8000.0f);
    bevBR_ = cv::Point2f(750.0f, 8500.0f);
    bevBL_ = cv::Point2f(250.0f, 8500.0f);
}

void BEVDistanceEstimator::calibrate(const BEVCalibration& calib) {
    std::lock_guard<std::mutex> lock(mutex_);
    calib_ = calib;

    // Compute pixel-to-meter ratios
    float bevCarpetWidth = bevTR_.x - bevTL_.x;   // 500 pixels
    float bevCarpetHeight = bevBR_.y - bevTR_.y;   // 500 pixels

    widthPixelToMeter_ = calib.carpetWidth / bevCarpetWidth;
    heightPixelToMeter_ = calib.carpetLength / bevCarpetHeight;

    // Car position in BEV (behind the calibration carpet)
    carYPixel_ = bevBR_.y + calib.carToCarpetDist / heightPixelToMeter_;
    carWidthPixel_ = calib.carWidth / widthPixelToMeter_;

    calibrated_.store(true);

    LOG_INFO("BEV", "Calibrated: pixel/m ratio W=" +
             std::to_string(widthPixelToMeter_) + " H=" +
             std::to_string(heightPixelToMeter_) +
             " CarY=" + std::to_string(carYPixel_));
}

bool BEVDistanceEstimator::loadCalibration(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        LOG_WARNING("BEV", "Calibration file not found: " + filePath);
        return false;
    }

    BEVCalibration calib;
    std::string label;
    file >> label >> calib.carWidth;
    file >> label >> calib.carpetWidth;
    file >> label >> calib.carToCarpetDist;
    file >> label >> calib.carpetLength;
    file >> label >> calib.imageTL.x >> label >> calib.imageTL.y;
    file >> label >> calib.imageTR.x >> label >> calib.imageTR.y;
    file >> label >> calib.imageBR.x >> label >> calib.imageBR.y;
    file >> label >> calib.imageBL.x >> label >> calib.imageBL.y;

    calibrate(calib);
    LOG_INFO("BEV", "Loaded calibration from: " + filePath);
    return true;
}

float BEVDistanceEstimator::estimateDistance(const cv::Point2f& bottomCenter,
                                              int imageWidth, int imageHeight) const {
    if (!calibrated_.load()) return -1.0f;
    std::lock_guard<std::mutex> lock(mutex_);

    // Build image-to-BEV transform
    std::vector<cv::Point2f> imgPts = {
        cv::Point2f(calib_.imageTL.x * imageWidth, calib_.imageTL.y * imageHeight),
        cv::Point2f(calib_.imageTR.x * imageWidth, calib_.imageTR.y * imageHeight),
        cv::Point2f(calib_.imageBR.x * imageWidth, calib_.imageBR.y * imageHeight),
        cv::Point2f(calib_.imageBL.x * imageWidth, calib_.imageBL.y * imageHeight)
    };
    std::vector<cv::Point2f> bevPts = {bevTL_, bevTR_, bevBR_, bevBL_};

    cv::Mat transform = cv::getPerspectiveTransform(imgPts, bevPts);

    // Transform the bottom-center point to BEV
    std::vector<cv::Point2f> srcPts = {bottomCenter};
    std::vector<cv::Point2f> dstPts;
    cv::perspectiveTransform(srcPts, dstPts, transform);

    // Distance from car to object in BEV
    float objY = dstPts[0].y;
    float distance = (carYPixel_ - objY) * heightPixelToMeter_;

    return (distance > 0.0f) ? distance : -1.0f;
}

void BEVDistanceEstimator::transformPoints(const std::vector<cv::Point2f>& imagePoints,
                                            std::vector<cv::Point2f>& bevPoints,
                                            int imageWidth, int imageHeight) const {
    if (!calibrated_.load()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<cv::Point2f> imgPts = {
        cv::Point2f(calib_.imageTL.x * imageWidth, calib_.imageTL.y * imageHeight),
        cv::Point2f(calib_.imageTR.x * imageWidth, calib_.imageTR.y * imageHeight),
        cv::Point2f(calib_.imageBR.x * imageWidth, calib_.imageBR.y * imageHeight),
        cv::Point2f(calib_.imageBL.x * imageWidth, calib_.imageBL.y * imageHeight)
    };
    std::vector<cv::Point2f> bevPts = {bevTL_, bevTR_, bevBR_, bevBL_};

    cv::Mat transform = cv::getPerspectiveTransform(imgPts, bevPts);
    cv::perspectiveTransform(imagePoints, bevPoints, transform);
}

cv::Mat BEVDistanceEstimator::getDangerZoneMask(const cv::Size& imageSize,
                                                  float dangerDistance) const {
    if (!calibrated_.load()) return cv::Mat();
    std::lock_guard<std::mutex> lock(mutex_);

    // Create danger zone in BEV space
    float dangerY = carYPixel_ - dangerDistance / heightPixelToMeter_;
    float halfCarW = carWidthPixel_ / 2.0f;
    float centerX = static_cast<float>(BEV_WIDTH) / 2.0f;

    cv::Mat bevMask(cv::Size(BEV_WIDTH, BEV_HEIGHT), CV_8UC1, cv::Scalar(0));
    cv::rectangle(bevMask,
                  cv::Point(static_cast<int>(centerX - halfCarW), static_cast<int>(dangerY)),
                  cv::Point(static_cast<int>(centerX + halfCarW), static_cast<int>(carYPixel_) - 1),
                  cv::Scalar(255), -1);

    // Transform danger zone back to image space
    std::vector<cv::Point2f> imgPts = {
        cv::Point2f(calib_.imageTL.x * imageSize.width, calib_.imageTL.y * imageSize.height),
        cv::Point2f(calib_.imageTR.x * imageSize.width, calib_.imageTR.y * imageSize.height),
        cv::Point2f(calib_.imageBR.x * imageSize.width, calib_.imageBR.y * imageSize.height),
        cv::Point2f(calib_.imageBL.x * imageSize.width, calib_.imageBL.y * imageSize.height)
    };
    std::vector<cv::Point2f> bevPts = {bevTL_, bevTR_, bevBR_, bevBL_};

    cv::Mat bev2img = cv::getPerspectiveTransform(bevPts, imgPts);

    cv::Mat imageMask;
    cv::warpPerspective(bevMask, imageMask, bev2img, imageSize);
    cv::threshold(imageMask, imageMask, 1, 255, cv::THRESH_BINARY);

    return imageMask;
}

} // namespace fcw
