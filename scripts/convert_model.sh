#!/bin/bash
# ==============================================================================
# Convert YOLOv8 ONNX model to TensorRT engine
# ==============================================================================
# This script converts a YOLOv8 ONNX model to a TensorRT engine
# optimized for the target platform (Jetson Nano or desktop GPU).
#
# Prerequisites:
#   - TensorRT installed
#   - trtexec available in PATH
#   - YOLOv8 ONNX model exported from ultralytics
#
# Usage:
#   ./scripts/convert_model.sh [onnx_path] [engine_path]
# ==============================================================================

set -e

# Default paths
ONNX_PATH="${1:-./models/yolov8n.onnx}"
ENGINE_PATH="${2:-./models/yolov8n.engine}"

# Configuration
INPUT_SHAPE="1x3x640x640"
PRECISION="fp16"  # fp16 for Jetson Nano, fp32 for accuracy
WORKSPACE_MB=256  # 256 MB for Jetson Nano (limited memory)

echo "=========================================="
echo "  YOLOv8 ONNX → TensorRT Conversion"
echo "=========================================="
echo "  ONNX model:    ${ONNX_PATH}"
echo "  Engine output:  ${ENGINE_PATH}"
echo "  Input shape:    ${INPUT_SHAPE}"
echo "  Precision:      ${PRECISION}"
echo "  Workspace:      ${WORKSPACE_MB} MB"
echo "=========================================="

# Check ONNX file exists
if [ ! -f "${ONNX_PATH}" ]; then
    echo "[ERROR] ONNX model not found: ${ONNX_PATH}"
    echo ""
    echo "To export YOLOv8 to ONNX, run:"
    echo "  pip install ultralytics"
    echo "  python3 -c \"from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx', imgsz=640, opset=12)\""
    echo ""
    exit 1
fi

# Check trtexec is available
if ! command -v trtexec &> /dev/null; then
    # Try common Jetson paths
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    if [ ! -f "${TRTEXEC}" ]; then
        echo "[ERROR] trtexec not found. Ensure TensorRT is installed."
        exit 1
    fi
else
    TRTEXEC="trtexec"
fi

echo ""
echo "Starting conversion (this may take 10-30 minutes on Jetson Nano)..."
echo ""

# Run conversion
${TRTEXEC} \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --${PRECISION} \
    --workspace=${WORKSPACE_MB} \
    --verbose

echo ""
echo "=========================================="
echo "  Conversion complete!"
echo "  Engine saved to: ${ENGINE_PATH}"
echo "=========================================="

# Verify engine
if [ -f "${ENGINE_PATH}" ]; then
    ENGINE_SIZE=$(du -h "${ENGINE_PATH}" | cut -f1)
    echo "  Engine size: ${ENGINE_SIZE}"
fi
