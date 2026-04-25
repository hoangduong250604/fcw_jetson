#!/bin/bash
# ==============================================================================
# Run FCW System Demo
# ==============================================================================
# Usage:
#   ./scripts/run_demo.sh                                           # Default video
#   ./scripts/run_demo.sh --video /path/to/video.avi                # Specific video
#   ./scripts/run_demo.sh --camera 0                                # Live camera
#   ./scripts/run_demo.sh --video /path/to/video.avi --threaded     # Multi-threaded
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
ROOT_DIR="$(dirname "${PROJECT_DIR}")"
BUILD_DIR="${PROJECT_DIR}/build"
VIDEO_DIR="${ROOT_DIR}/video_data"

# Platform detection
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* || "$OSTYPE" == "cygwin" ]]; then
    BIN="${BUILD_DIR}/fcw_system.exe"
    export PATH="C:/opencv-mingw/x64/mingw/bin:C:/mingw64/bin:${ROOT_DIR}/onnxruntime/lib:${PATH}"
else
    BIN="${BUILD_DIR}/fcw_system"
fi

# Check if built
if [ ! -f "${BIN}" ]; then
    echo "[INFO] Binary not found. Building..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "mingw"* ]]; then
        cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
              -DUSE_ONNXRUNTIME=ON -DUSE_TENSORRT=OFF ..
        mingw32-make -j${NUMBER_OF_PROCESSORS:-4}
    else
        cmake -DCMAKE_BUILD_TYPE=Release ..
        make -j$(nproc)
    fi
    cd "${PROJECT_DIR}"
fi

# Create output directories
mkdir -p "${PROJECT_DIR}/results/logs" "${PROJECT_DIR}/results/videos"

# Find default video if no args specify input
HAS_INPUT=false
for arg in "$@"; do
    if [[ "${arg}" == "--video" || "${arg}" == "--camera" ]]; then
        HAS_INPUT=true
        break
    fi
done

echo "=========================================="
echo "  Forward Collision Warning System Demo"
echo "=========================================="
echo "[INFO] Press 'q' or ESC to quit"
echo ""

# Run
cd "${PROJECT_DIR}"
if [ "${HAS_INPUT}" = true ]; then
    ${BIN} "$@"
else
    # Use default video
    DEFAULT_VIDEO=$(ls "${VIDEO_DIR}"/2011_09_26_drive_0009_sync.avi \
                       "${VIDEO_DIR}"/2011_09_26_drive_0001_sync.avi \
                       "${VIDEO_DIR}"/2011_*.avi 2>/dev/null | head -1)
    if [ -n "${DEFAULT_VIDEO}" ]; then
        echo "[INFO] Using default video: ${DEFAULT_VIDEO}"
        ${BIN} --video "${DEFAULT_VIDEO}" "$@"
    else
        echo "[ERROR] No video found. Specify --video or --camera"
        exit 1
    fi
fi
