# BDD100K Dataset Pipeline
# ========================
# Complete pipeline for preparing BDD100K dataset for YOLOv8 training
# and deploying the model to Jetson Nano.
#
# Prerequisites:
#   pip install ultralytics bdd100k

# ============================================================================
# STEP 1: Download BDD100K Dataset
# ============================================================================
# Register at https://bdd-data.berkeley.edu/
# Download:
#   - 100K Images (train/val/test) → images/100k/
#   - Detection 2020 Labels        → labels/det_20/
#
# Expected structure:
#   bdd100k/
#   ├── images/
#   │   └── 100k/
#   │       ├── train/   (70,000 images)
#   │       └── val/     (10,000 images)
#   └── labels/
#       └── det_20/
#           ├── det_train.json
#           └── det_val.json

# ============================================================================
# STEP 2: Convert BDD100K JSON to YOLO Format
# ============================================================================
# This converts BDD100K's native JSON annotations to YOLO TXT format.
# Each image gets a .txt file with: class_id cx cy w h (normalized 0-1)

python scripts/dataset/bdd100k_to_yolo.py \
    --labels_dir /path/to/bdd100k/labels/det_20/ \
    --images_dir /path/to/bdd100k/images/100k/ \
    --output_dir /path/to/bdd100k_yolo/ \
    --splits train val

# ============================================================================
# STEP 3: Train YOLOv8 on BDD100K
# ============================================================================
# Use yolov8n (nano) or yolov8s (small) for Jetson Nano real-time inference.
# Training on a GPU workstation (RTX 3060/3090/4090 recommended).

# Option A: Using our training script
python scripts/dataset/train_yolov8_bdd100k.py \
    --data /path/to/bdd100k_yolo/bdd100k.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --export

# Option B: Using ultralytics CLI directly
yolo train \
    data=/path/to/bdd100k_yolo/bdd100k.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16

# ============================================================================
# STEP 4: Export Model to ONNX
# ============================================================================
# Export the best trained model to ONNX format (portable).

python scripts/dataset/export_model.py \
    --model runs/bdd100k/yolov8_bdd100k/weights/best.pt \
    --format onnx \
    --imgsz 640

# Or using ultralytics CLI:
yolo export model=best.pt format=onnx imgsz=640 simplify=True

# ============================================================================
# STEP 5: Convert ONNX to TensorRT on Jetson Nano
# ============================================================================
# On the Jetson Nano, convert ONNX to TensorRT .engine for fastest inference.
# FP16 gives ~2x speedup with minimal accuracy loss.

/usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=yolov8n_bdd100k.engine \
    --fp16 \
    --workspace=1024

# ============================================================================
# STEP 6: Deploy in FCW System
# ============================================================================
# Copy the .engine file to the fcw-system models directory:
#   cp yolov8n_bdd100k.engine fcw-system/models/yolov8n_bdd100k.engine
#
# Update fcw-system/config/system_config.yaml:
#   detection:
#     model_path: "models/yolov8n_bdd100k.engine"
#     target_classes: [0, 1, 2, 3, 4]  # pedestrian, rider, car, truck, bus
#     confidence_threshold: 0.45
#     nms_threshold: 0.50
#
# 10 BDD100K classes:
#   0: pedestrian    5: train
#   1: rider         6: motorcycle
#   2: car           7: bicycle
#   3: truck         8: traffic light
#   4: bus           9: traffic sign
