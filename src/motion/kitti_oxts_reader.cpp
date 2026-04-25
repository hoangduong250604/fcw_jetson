// ==============================================================================
// KITTI OXTS Reader Implementation
// ==============================================================================
// Video naming convention: <drive_name>.avi (e.g., 2011_09_26_drive_0001_sync.avi)
// This makes OXTS auto-detection trivial: video stem = KITTI drive folder name.
//
// Primary:  video name → direct folder lookup (fastest, no scanning)
// Fallback: scan all drives, match by frame count (backward compatibility)
// ==============================================================================

#include "kitti_oxts_reader.h"
#include "logger.h"

#include <experimental/filesystem>
#include <algorithm>

namespace fs = std::experimental::filesystem;

namespace fcw {

// ---------------------------------------------------------------------------
// Helper: find the oxts/data/ folder inside a KITTI drive directory
// Structure: drive_folder/date/drive_name/oxts/data/
// ---------------------------------------------------------------------------
static std::string findOxtsDataDir(const fs::path& driveDir) {
    for (auto& entry : fs::recursive_directory_iterator(driveDir)) {
        if (std::experimental::filesystem::is_directory(entry.path()) && entry.path().filename() == "data") {
            fs::path parent = entry.path().parent_path();
            if (parent.filename() == "oxts") {
                fs::path testFile = entry.path() / "0000000000.txt";
                if (fs::exists(testFile)) {
                    return entry.path().string();
                }
            }
        }
    }
    return "";
}

// ---------------------------------------------------------------------------
// Helper: count .txt files inside an oxts/data/ directory
// ---------------------------------------------------------------------------
static int countOxtsFiles(const std::string& oxtsDataDir) {
    int count = 0;
    for (auto& f : fs::directory_iterator(oxtsDataDir)) {
        if (std::experimental::filesystem::is_regular_file(f.path()) && f.path().extension() == ".txt") {
            count++;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// autoDetectFromVideo — match video to its KITTI drive
// Primary:  direct name match (video stem = drive folder name)
// Fallback: frame count match
// ---------------------------------------------------------------------------
bool KittiOxtsReader::autoDetectFromVideo(const std::string& videoPath,
                                           const std::string& kittiRoot,
                                           int videoFrameCount) {
    if (kittiRoot.empty()) {
        LOG_WARNING("OxtsReader", "KITTI root path not specified");
        return false;
    }

    fs::path kittiPath(kittiRoot);
    if (!fs::exists(kittiPath)) {
        LOG_WARNING("OxtsReader", "KITTI root not found: " + kittiRoot);
        return false;
    }

    // Extract video stem: e.g., "2011_09_26_drive_0001_sync"
    fs::path vidPath(videoPath);
    std::string vidName = vidPath.stem().string();

    LOG_INFO("OxtsReader", "Auto-detecting OXTS for: " + vidName);

    // ===== PRIMARY: Direct name match (video name = drive folder name) =====
    fs::path directDrive = kittiPath / vidName;
    if (fs::exists(directDrive) && fs::is_directory(directDrive)) {
        std::string oxtsPath = findOxtsDataDir(directDrive);
        if (!oxtsPath.empty()) {
            int oxtsCount = countOxtsFiles(oxtsPath);
            oxtsFolder_ = oxtsPath;
            driveName_ = vidName;
            enabled_ = true;
            LOG_INFO("OxtsReader", "Direct match: " + vidName +
                     " → " + oxtsPath + " (" + std::to_string(oxtsCount) + " OXTS files)");
            return true;
        }
    }

    // ===== FALLBACK: Scan drives, match by frame count =====
    if (videoFrameCount <= 0) {
        LOG_WARNING("OxtsReader", "No direct match and invalid frame count");
        return false;
    }

    LOG_INFO("OxtsReader", "No direct match, scanning by frame count (" +
             std::to_string(videoFrameCount) + ")...");

    for (auto& drive : fs::directory_iterator(kittiPath)) {
        if (!std::experimental::filesystem::is_directory(drive.path())) continue;
        std::string dName = drive.path().filename().string();

        std::string oxtsPath = findOxtsDataDir(drive.path());
        if (oxtsPath.empty()) continue;

        int oxtsCount = countOxtsFiles(oxtsPath);
        if (oxtsCount == videoFrameCount) {
            oxtsFolder_ = oxtsPath;
            driveName_ = dName;
            enabled_ = true;
            LOG_INFO("OxtsReader", "Frame-count match: " + dName +
                     " → " + oxtsPath + " (" + std::to_string(oxtsCount) + " files)");
            return true;
        }
    }

    LOG_WARNING("OxtsReader", "Could not find OXTS for: " + vidName);
    return false;
}

} // namespace fcw
