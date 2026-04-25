#pragma once
// ==============================================================================
// Distance Estimator - Estimate distance to tracked vehicles
// ==============================================================================
// Uses monocular camera geometry to estimate distance from bounding box.
//
// Methods supported:
//   1. BBox Height method: D = (f * H_real) / h_pixel
//      Most reliable for vehicle detection. Uses known average car height.
//
//   2. Ground Plane method: Uses bbox bottom + camera geometry
//      Requires accurate camera mounting calibration.
//
//   3. Combined: Average both methods for robustness
//
// EMA smoothing is applied per-track to reduce distance estimation noise.
// ==============================================================================

#include <unordered_map>
#include <vector>
#include "camera_model.h"
#include "bev_distance.h"
#include "track.h"

namespace fcw {

enum class DistanceMethod {
    BBOX_HEIGHT,     // Use bounding box height
    GROUND_PLANE,    // Use ground plane geometry
    COMBINED,        // Average of bbox_height and ground_plane
    BEV              // Bird's Eye View perspective transform
};

struct DistanceConfig {
    DistanceMethod method = DistanceMethod::BBOX_HEIGHT;
    float referenceHeight = 1.5f;    // Average vehicle height (meters)
    float maxDistance = 100.0f;       // Max reliable distance
    float minDistance = 2.0f;         // Min reliable distance
    float smoothingAlpha = 0.3f;     // EMA smoothing factor (lower = heavier smoothing)
    float corridorHalfWidth = 1.5f;  // Ego-lane corridor half-width (meters)
};

/**
 * Per-track distance estimation data.
 */
struct DistanceInfo {
    int trackId = -1;
    float rawDistance = 0.0f;         // Unsmoothed estimate (meters)
    float smoothedDistance = 0.0f;    // EMA-smoothed distance (meters)
    float previousDistance = 0.0f;    // Distance at previous frame
    
    // Ego-Lane Corridor (lateral offset analysis)
    float lateralOffset = 0.0f;       // X-axis offset from camera center (meters, negative=left)
    float longitudinalDist = 0.0f;    // Z-axis depth (meters, straight ahead)
    bool inEgoPath = false;           // true = vehicle is in ego lane corridor
    
    bool valid = false;               // Whether estimate is reliable
    cv::Rect bbox;                    // Bounding box for this detection
};

class DistanceEstimator {
public:
    DistanceEstimator();
    explicit DistanceEstimator(const DistanceConfig& config);

    /**
     * Estimate distances for all active tracks.
     * @param tracks  Active confirmed tracks from tracker
     * @return Map of trackId → DistanceInfo
     */
    std::unordered_map<int, DistanceInfo> estimate(const std::vector<Track*>& tracks);

    /**
     * Estimate distance for a single detection bbox.
     */
    float estimateSingle(const utils::BBox& bbox) const;

    /** Set camera model */
    void setCameraModel(const CameraModel& model);

    /** Set BEV distance estimator (load calibration from file) */
    void setBEVEstimator(const std::string& calibFilePath);

    /** Set config */
    void setConfig(const DistanceConfig& config);

    /** Set image resolution (must match video) */
    void setImageSize(int width, int height) { imageWidth_ = width; imageHeight_ = height; }

    /** Get distance for specific track */
    DistanceInfo getDistance(int trackId) const;

private:
    float computeDistance(const utils::BBox& bbox) const;

    DistanceConfig config_;
    CameraModel cameraModel_;
    std::unique_ptr<BEVDistanceEstimator> bevEstimator_;
    int imageWidth_ = 1280;
    int imageHeight_ = 720;

    // Per-track smoothed distances
    std::unordered_map<int, DistanceInfo> trackDistances_;
};

} // namespace fcw
