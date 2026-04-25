#!/usr/bin/env python3
"""
Export YOLOv8 Model: PyTorch -> ONNX -> TensorRT
==================================================
Pipeline for preparing YOLOv8 model for Jetson Nano deployment.

Steps:
  1. PyTorch (.pt) -> ONNX (.onnx) via ultralytics export
  2. ONNX -> TensorRT (.engine) via trtexec on Jetson Nano

For Jetson Nano:
  - Use FP16 precision (half=True) for 2x speedup
  - Use static input shape (640x640) for optimal TensorRT optimization
  - Compute capability: sm_53

Usage:
    # On training machine (export to ONNX):
    python export_model.py --model best.pt --format onnx --imgsz 640

    # On Jetson Nano (convert ONNX to TensorRT):
    python export_model.py --model best.onnx --format engine --imgsz 640

    # Or using trtexec directly on Jetson:
    /usr/src/tensorrt/bin/trtexec \
        --onnx=best.onnx \
        --saveEngine=best.engine \
        --fp16 \
        --workspace=1024
"""

import argparse
import os


def export_to_onnx(model_path, imgsz):
    """Export PyTorch model to ONNX."""
    from ultralytics import YOLO

    print(f"Exporting {model_path} to ONNX...")
    model = YOLO(model_path)

    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )
    print(f"ONNX model saved: {onnx_path}")
    return onnx_path


def export_to_engine(model_path, imgsz, device="0"):
    """Export model to TensorRT engine (requires TensorRT installed)."""
    from ultralytics import YOLO

    print(f"Exporting {model_path} to TensorRT engine (FP16)...")
    model = YOLO(model_path)

    try:
        engine_path = model.export(
            format="engine",
            imgsz=imgsz,
            half=True,  # FP16 for Jetson Nano
            device=device,
        )
        print(f"TensorRT engine saved: {engine_path}")
        return engine_path
    except Exception as e:
        print(f"Ultralytics TensorRT export failed: {e}")
        print("\nFallback: Use trtexec on Jetson Nano:")
        onnx = model_path.replace(".pt", ".onnx")
        engine = model_path.replace(".pt", ".engine").replace(".onnx", ".engine")
        print(f"  /usr/src/tensorrt/bin/trtexec \\")
        print(f"      --onnx={onnx} \\")
        print(f"      --saveEngine={engine} \\")
        print(f"      --fp16 \\")
        print(f"      --workspace=1024")
        return None


def validate_model(model_path, data_yaml, imgsz):
    """Validate exported model on BDD100K val set."""
    from ultralytics import YOLO

    print(f"\nValidating {model_path}...")
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=1,
        device="0",
    )

    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for Jetson Nano")
    parser.add_argument("--model", required=True, help="Path to model (.pt or .onnx)")
    parser.add_argument("--format", choices=["onnx", "engine", "all"], default="onnx",
                        help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument("--validate", default=None,
                        help="Path to dataset YAML for validation after export")
    args = parser.parse_args()

    if args.format == "onnx" or args.format == "all":
        onnx_path = export_to_onnx(args.model, args.imgsz)

    if args.format == "engine" or args.format == "all":
        export_to_engine(args.model, args.imgsz, args.device)

    if args.validate:
        output = args.model.replace(".pt", ".onnx") if args.format == "onnx" else args.model
        validate_model(output, args.validate, args.imgsz)


if __name__ == "__main__":
    main()
