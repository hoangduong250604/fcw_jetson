#!/bin/bash
# ==============================================================================
# Build FCW System on Jetson Nano with TensorRT
# ==============================================================================
# Prerequisites:
#   - JetPack SDK installed (includes CUDA, TensorRT, OpenCV)
#   - cmake >= 3.10
#
# Usage:
#   ./scripts/build_jetson.sh              # Build with TensorRT (default)
#   ./scripts/build_jetson.sh --clean      # Clean build
#   ./scripts/build_jetson.sh --onnx       # Build with OpenCV DNN instead
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_DIR}/build"

# Parse args
CLEAN_BUILD=false
USE_TRT=ON
USE_ORT=OFF

for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            ;;
        --onnx)
            USE_TRT=OFF
            USE_ORT=OFF
            ;;
    esac
done

echo "=========================================="
echo "  FCW System - Jetson Nano Build"
echo "=========================================="
echo "  TensorRT: ${USE_TRT}"
echo "  Build dir: ${BUILD_DIR}"
echo "=========================================="

# Check we're on Jetson
if [ -f "/etc/nv_tegra_release" ]; then
    echo "[INFO] Jetson platform detected"
    cat /etc/nv_tegra_release
else
    echo "[WARN] Not running on Jetson - build may still work if CUDA/TensorRT are installed"
fi

# Check dependencies
echo ""
echo "[CHECK] Checking dependencies..."

# CUDA
if [ -d "/usr/local/cuda" ]; then
    CUDA_VER=$(cat /usr/local/cuda/version.txt 2>/dev/null || nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',' || echo "unknown")
    echo "  CUDA: ${CUDA_VER}"
else
    echo "  [ERROR] CUDA not found at /usr/local/cuda"
    exit 1
fi

# TensorRT
if [ "${USE_TRT}" = "ON" ]; then
    if dpkg -l 2>/dev/null | grep -q tensorrt || [ -f "/usr/include/aarch64-linux-gnu/NvInfer.h" ] || [ -f "/usr/include/NvInfer.h" ]; then
        echo "  TensorRT: installed"
    else
        echo "  [ERROR] TensorRT not found. Install via: sudo apt install tensorrt"
        exit 1
    fi
fi

# OpenCV
if pkg-config --exists opencv4 2>/dev/null; then
    echo "  OpenCV: $(pkg-config --modversion opencv4)"
elif pkg-config --exists opencv 2>/dev/null; then
    echo "  OpenCV: $(pkg-config --modversion opencv)"
else
    echo "  [WARN] OpenCV pkg-config not found, cmake will try to locate it"
fi

# cmake
if command -v cmake &> /dev/null; then
    echo "  cmake: $(cmake --version | head -1)"
else
    echo "  [ERROR] cmake not found. Install: sudo apt install cmake"
    exit 1
fi

# Clean if requested
if [ "${CLEAN_BUILD}" = true ]; then
    echo ""
    echo "[INFO] Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build dir
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure
echo ""
echo "[BUILD] Configuring with cmake..."
cmake "${PROJECT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_TENSORRT=${USE_TRT} \
    -DUSE_ONNXRUNTIME=${USE_ORT} \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Build (limit to 2 on Nano to avoid OOM)
NPROC=$(nproc)
if [ "$NPROC" -gt 2 ]; then
    NPROC=2
fi

echo ""
echo "[BUILD] Compiling with ${NPROC} parallel jobs..."
make -j${NPROC}

echo ""
echo "=========================================="
echo "  Build complete!"
echo "  Binary: ${BUILD_DIR}/fcw_system"
echo "=========================================="

# Check if model exists
if [ ! -f "${PROJECT_DIR}/models/yolov8s.engine" ] && [ "${USE_TRT}" = "ON" ]; then
    echo ""
    echo "[NEXT STEP] Convert ONNX model to TensorRT engine:"
    echo "  ./scripts/convert_model.sh ./models/yolov8s.onnx ./models/yolov8s.engine"
    echo ""
    echo "Then run:"
    echo "  ./build/fcw_system --camera 0"
fi
