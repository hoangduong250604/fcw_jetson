#!/usr/bin/env python3
"""
YOLOv8 Training Script for BDD100K
====================================
Train YOLOv8 on BDD100K dataset for the FCW system.

Usage:
    python train_yolov8_bdd100k.py \
        --data /path/to/bdd100k_yolo/bdd100k.yaml \
        --model yolov8n.pt \
        --epochs 100 \
        --imgsz 640 \
        --batch 16

For Jetson Nano deployment, we recommend:
  - yolov8n (nano) or yolov8s (small) for real-time inference
  - Export to ONNX then TensorRT FP16 for maximum speed
"""

import argparse
import os


def train(args):
    """Train YOLOv8 on BDD100K."""
    from ultralytics import YOLO

    print("=" * 60)
    print("YOLOv8 Training on BDD100K")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print()

    # Load model (pretrained on COCO)
    model = YOLO(args.model)

    # Train on BDD100K
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Optimization
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\nTraining complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")

    return results


def export_model(model_path, args):
    """Export trained model to ONNX and TensorRT."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Export to ONNX
    print("\n--- Exporting to ONNX ---")
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    print(f"ONNX model: {onnx_path}")

    # Export to TensorRT (if available)
    if args.export_trt:
        print("\n--- Exporting to TensorRT ---")
        try:
            engine_path = model.export(
                format="engine",
                imgsz=args.imgsz,
                half=True,  # FP16 for Jetson
                device=args.device,
            )
            print(f"TensorRT engine: {engine_path}")
        except Exception as e:
            print(f"TensorRT export failed: {e}")
            print("You can convert ONNX to TensorRT on Jetson Nano using:")
            print(f"  /usr/src/tensorrt/bin/trtexec --onnx={onnx_path} --saveEngine=model.engine --fp16")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K")
    parser.add_argument("--data", required=True, help="Path to bdd100k.yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Pretrained model (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device (0, 1, cpu)")
    parser.add_argument("--project", default="runs/bdd100k")
    parser.add_argument("--name", default="yolov8_bdd100k")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--export", action="store_true",
                        help="Export model after training")
    parser.add_argument("--export_trt", action="store_true",
                        help="Also export to TensorRT")
    parser.add_argument("--export_only", default=None,
                        help="Skip training, just export this model")
    args = parser.parse_args()

    if args.export_only:
        export_model(args.export_only, args)
    else:
        results = train(args)
        if args.export:
            best_model = os.path.join(results.save_dir, "weights", "best.pt")
            export_model(best_model, args)


if __name__ == "__main__":
    main()
