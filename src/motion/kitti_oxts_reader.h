#pragma once
// ==============================================================================
// KITTI OXTS Reader - Read GPS/IMU ground truth data
// ==============================================================================
// Reads OXTS .txt files from KITTI dataset to get ground-truth ego speed.
// Each frame N has a corresponding file: 0000000N.txt (10-digit zero-padded)
// Index 8 (vf) = Forward Velocity in m/s
// ==============================================================================

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace fcw {

struct OxtsData {
    float lat = 0.0f;        // [0] Latitude (deg)
    float lon = 0.0f;        // [1] Longitude (deg)
    float alt = 0.0f;        // [2] Altitude (m)
    float roll = 0.0f;       // [3] Roll (rad)
    float pitch = 0.0f;      // [4] Pitch (rad)
    float yaw = 0.0f;        // [5] Yaw / Heading (rad)
    float vn = 0.0f;         // [6] Velocity North (m/s)
    float ve = 0.0f;         // [7] Velocity East (m/s)
    float vf = 0.0f;         // [8] Forward Velocity (m/s) ← EGO SPEED
    float vl = 0.0f;         // [9] Leftward Velocity (m/s)
    float vu = 0.0f;         // [10] Upward Velocity (m/s)
    float ax = 0.0f;         // [11] Forward Acceleration (m/s²) - body frame
    float ay = 0.0f;         // [12] Lateral Acceleration (m/s²) - body frame
    float az = 0.0f;         // [13] Vertical Acceleration (m/s²) - body frame
    float af = 0.0f;         // [14] Forward Acceleration (m/s²) - navigation frame
    float al = 0.0f;         // [15] Leftward Acceleration (m/s²) - navigation frame
    float au = 0.0f;         // [16] Upward Acceleration (m/s²) - navigation frame
    float wx = 0.0f;         // [17] Angular rate X - roll rate (rad/s)
    float wy = 0.0f;         // [18] Angular rate Y - pitch rate (rad/s)
    float wz = 0.0f;         // [19] Angular rate Z - yaw rate (rad/s)
    bool valid = false;
};

class KittiOxtsReader {
public:
    KittiOxtsReader() = default;

    /// Set the OXTS data folder path (e.g., ".../oxts/data/")
    void setFolder(const std::string& oxtsDataFolder) {
        oxtsFolder_ = oxtsDataFolder;
        enabled_ = !oxtsFolder_.empty();
    }

    /// Auto-detect OXTS folder by matching video frame count to KITTI drive
    /// videoPath: path to .avi/.mp4 (used to extract date from filename)
    /// kittiRoot: path to KITTI/ folder containing drive folders
    /// videoFrameCount: total frames in the video (must match OXTS file count)
    bool autoDetectFromVideo(const std::string& videoPath, const std::string& kittiRoot,
                              int videoFrameCount);

    /// Check if OXTS reader is enabled and has a valid folder
    bool isEnabled() const { return enabled_; }

    /// Read OXTS data for a specific frame index
    OxtsData readFrame(int frameIndex) const {
        if (!enabled_) return {};

        std::ostringstream filename;
        filename << oxtsFolder_ << "/"
                 << std::setfill('0') << std::setw(10) << frameIndex << ".txt";

        std::ifstream file(filename.str());
        if (!file.is_open()) return {};

        std::string line;
        if (!std::getline(file, line)) return {};

        std::istringstream iss(line);
        std::vector<float> values;
        float val;
        while (iss >> val) {
            values.push_back(val);
        }

        if (values.size() < 20) return {};

        OxtsData data;
        data.lat = values[0];
        data.lon = values[1];
        data.alt = values[2];
        data.roll = values[3];
        data.pitch = values[4];
        data.yaw = values[5];
        data.vn = values[6];
        data.ve = values[7];
        data.vf = values[8];    // Forward velocity = Ego Speed
        data.vl = values[9];
        data.vu = values[10];
        data.ax = values[11];   // Forward accel (body frame) → brake detection
        data.ay = values[12];
        data.az = values[13];
        data.af = values[14];
        data.al = values[15];
        data.au = values[16];
        data.wx = values[17];
        data.wy = values[18];
        data.wz = values[19];   // Yaw rate → turn detection
        data.valid = true;
        return data;
    }

    /// Convenience: get ego speed in km/h for a frame
    float getEgoSpeedKmh(int frameIndex) const {
        OxtsData data = readFrame(frameIndex);
        if (!data.valid) return -1.0f;
        return data.vf * 3.6f;
    }

    /// Get the configured folder path
    const std::string& getFolder() const { return oxtsFolder_; }

    /// Get the matched drive name (e.g., "2011_09_26_drive_0001_sync")
    const std::string& getDriveName() const { return driveName_; }

private:
    std::string oxtsFolder_;
    std::string driveName_;
    bool enabled_ = false;
};

} // namespace fcw
