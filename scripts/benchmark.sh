#!/bin/bash
# ==============================================================================
# Benchmark FCW System Performance
# ==============================================================================
# Usage:
#   ./scripts/benchmark.sh                              # Default video
#   ./scripts/benchmark.sh video_data/2011_*.avi        # Specific video
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
    PLATFORM="Windows"
else
    BIN="${BUILD_DIR}/fcw_system"
    PLATFORM="Linux"
fi

echo "=========================================="
echo "  FCW System Benchmark"
echo "=========================================="

# System info
echo ""
echo "--- System Info ---"
echo "Platform: ${PLATFORM}"
if [ "${PLATFORM}" = "Windows" ]; then
    echo "OS: $(cmd /c ver 2>/dev/null || echo 'Windows')"
    echo "CPU: $(wmic cpu get name 2>/dev/null | sed -n '2p' || echo 'Unknown')"
else
    uname -a
    if [ -f /etc/nv_tegra_release ]; then
        echo "Jetson: $(cat /etc/nv_tegra_release | head -1)"
    fi
fi

# GPU info
echo ""
echo "--- GPU Info ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
elif [ -f /etc/nv_tegra_release ]; then
    echo "Jetson Nano (Maxwell, 128 CUDA cores)"
else
    echo "No NVIDIA GPU detected (CPU inference)"
fi

# Inference backend
echo ""
echo "--- Inference Backend ---"
if [ -f "${PROJECT_DIR}/models/yolov8s.engine" ]; then
    echo "Backend: TensorRT"
    echo "Engine: ${PROJECT_DIR}/models/yolov8s.engine"
elif [ -f "${PROJECT_DIR}/models/yolov8s.onnx" ]; then
    echo "Backend: ONNX Runtime"
    echo "Model: ${PROJECT_DIR}/models/yolov8s.onnx"
    ls -lh "${PROJECT_DIR}/models/yolov8s.onnx" | awk '{print "Size: " $5}'
else
    echo "No model found!"
fi

# TensorRT engine benchmark (if available)
if [ -f "${PROJECT_DIR}/models/yolov8s.engine" ]; then
    echo ""
    echo "--- TensorRT Engine Benchmark ---"
    TRTEXEC=$(command -v trtexec || echo "/usr/src/tensorrt/bin/trtexec")
    
    if [ -f "${TRTEXEC}" ] || command -v trtexec &>/dev/null; then
        ${TRTEXEC} \
            --loadEngine="${PROJECT_DIR}/models/yolov8s.engine" \
            --warmUp=500 \
            --duration=10 \
            --fp16 2>&1 | grep -E "mean|median|GPU Compute"
    else
        echo "trtexec not found, skipping engine benchmark"
    fi
fi

# FCW pipeline benchmark
echo ""
echo "--- FCW Pipeline Benchmark ---"
if [ ! -f "${BIN}" ]; then
    echo "Binary not found. Build first: ./build.sh"
else
    VIDEO="${1}"
    if [ -z "${VIDEO}" ]; then
        VIDEO=$(ls "${VIDEO_DIR}"/2011_09_26_drive_0001_sync.avi 2>/dev/null | head -1)
    fi
    
    if [ -n "${VIDEO}" ] && [ -f "${VIDEO}" ]; then
        echo "Video: $(basename ${VIDEO})"
        echo "Running FCW system (check FPS in output)..."
        echo ""
        cd "${PROJECT_DIR}"
        ${BIN} --video "${VIDEO}" 2>&1 | tail -20
    else
        echo "No video available for benchmark"
    fi
fi
