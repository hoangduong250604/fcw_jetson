#!/usr/bin/env python3
"""
BDD100K to YOLO Format Converter
=================================
Converts BDD100K native JSON annotations to YOLO TXT format.

Input:  BDD100K JSON (det_train.json, det_val.json)
Output: YOLO TXT files (one per image, class_id cx cy w h - normalized)

Usage:
    python bdd100k_to_yolo.py \
        --labels_dir /path/to/bdd100k/labels/det_20/ \
        --images_dir /path/to/bdd100k/images/100k/ \
        --output_dir /path/to/bdd100k_yolo/ \
        --splits train val
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

# Official BDD100K 10 detection classes (0-indexed for YOLO)
BDD100K_CLASSES = [
    "pedestrian",     # 0
    "rider",          # 1
    "car",            # 2
    "truck",          # 3
    "bus",            # 4
    "train",          # 5
    "motorcycle",     # 6
    "bicycle",        # 7
    "traffic light",  # 8
    "traffic sign",   # 9
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(BDD100K_CLASSES)}

# BDD100K image dimensions
IMG_WIDTH = 1280
IMG_HEIGHT = 720


def convert_box(box2d, img_w=IMG_WIDTH, img_h=IMG_HEIGHT):
    """Convert BDD100K box2d (x1,y1,x2,y2) to YOLO format (cx,cy,w,h) normalized."""
    x1 = box2d["x1"]
    y1 = box2d["y1"]
    x2 = box2d["x2"]
    y2 = box2d["y2"]

    # Clamp to image boundaries
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))

    # Convert to center format and normalize
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # Skip degenerate boxes
    if w <= 0 or h <= 0:
        return None

    return cx, cy, w, h


def convert_split(json_path, output_labels_dir, images_dir=None):
    """Convert one split (train/val) from BDD100K JSON to YOLO TXT."""
    print(f"Loading {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    os.makedirs(output_labels_dir, exist_ok=True)

    stats = defaultdict(int)
    skipped_no_labels = 0
    skipped_no_box2d = 0
    skipped_unknown_class = 0
    total_boxes = 0
    total_images = 0

    for frame in data:
        image_name = frame["name"]
        stem = Path(image_name).stem
        label_path = os.path.join(output_labels_dir, f"{stem}.txt")

        labels = frame.get("labels", [])
        if not labels:
            skipped_no_labels += 1
            # Create empty label file (image with no objects)
            with open(label_path, "w") as f:
                pass
            total_images += 1
            continue

        lines = []
        for label in labels:
            category = label.get("category", "")
            if category not in CLASS_TO_ID:
                skipped_unknown_class += 1
                continue

            box2d = label.get("box2d")
            if box2d is None:
                skipped_no_box2d += 1
                continue

            result = convert_box(box2d)
            if result is None:
                continue

            cx, cy, w, h = result
            class_id = CLASS_TO_ID[category]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[category] += 1
            total_boxes += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        total_images += 1

    print(f"  Converted: {total_images} images, {total_boxes} boxes")
    print(f"  Skipped: {skipped_no_labels} images without labels, "
          f"{skipped_no_box2d} labels without box2d, "
          f"{skipped_unknown_class} unknown classes")
    print("  Class distribution:")
    for cls_name in BDD100K_CLASSES:
        count = stats.get(cls_name, 0)
        print(f"    {cls_name}: {count}")

    return total_images, total_boxes


def create_symlinks(images_src, images_dst):
    """Create symlinks from source images to YOLO dataset structure."""
    if os.path.exists(images_dst):
        print(f"  Images dir already exists: {images_dst}")
        return

    if os.path.exists(images_src):
        # Create symlink (or copy on Windows)
        os.makedirs(os.path.dirname(images_dst), exist_ok=True)
        try:
            os.symlink(images_src, images_dst)
            print(f"  Symlinked: {images_src} -> {images_dst}")
        except OSError:
            print(f"  NOTE: Symlink failed. Please manually copy or link:")
            print(f"    {images_src} -> {images_dst}")
    else:
        print(f"  WARNING: Source images not found: {images_src}")
        print(f"  Please download BDD100K images and place them at: {images_src}")


def main():
    parser = argparse.ArgumentParser(description="Convert BDD100K JSON to YOLO format")
    parser.add_argument("--labels_dir", required=True,
                        help="Directory containing BDD100K JSON files (det_train.json, det_val.json)")
    parser.add_argument("--images_dir", default=None,
                        help="Directory containing BDD100K images (images/100k/)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to convert (default: train val)")
    args = parser.parse_args()

    print("=" * 60)
    print("BDD100K to YOLO Format Converter")
    print("=" * 60)
    print(f"Labels dir: {args.labels_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Classes ({len(BDD100K_CLASSES)}): {BDD100K_CLASSES}")
    print()

    for split in args.splits:
        print(f"\n--- Converting {split} split ---")

        # Find JSON file
        json_candidates = [
            os.path.join(args.labels_dir, f"det_{split}.json"),
            os.path.join(args.labels_dir, f"bdd100k_labels_images_{split}.json"),
            os.path.join(args.labels_dir, f"{split}.json"),
        ]
        json_path = None
        for candidate in json_candidates:
            if os.path.exists(candidate):
                json_path = candidate
                break

        if json_path is None:
            print(f"  ERROR: No JSON file found for split '{split}'. Tried:")
            for c in json_candidates:
                print(f"    {c}")
            continue

        # Output labels directory
        output_labels = os.path.join(args.output_dir, "labels", split)
        convert_split(json_path, output_labels)

        # Handle images
        if args.images_dir:
            images_src = os.path.join(args.images_dir, split)
            images_dst = os.path.join(args.output_dir, "images", split)
            create_symlinks(images_src, images_dst)

    # Generate dataset YAML
    yaml_path = os.path.join(args.output_dir, "bdd100k.yaml")
    yaml_content = generate_yaml(args.output_dir)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nDataset YAML written to: {yaml_path}")
    print("\nDone!")


def generate_yaml(output_dir):
    """Generate Ultralytics-compatible dataset YAML."""
    lines = [
        "# BDD100K Dataset Configuration for YOLOv8",
        f"# Auto-generated by bdd100k_to_yolo.py",
        "",
        f"path: {os.path.abspath(output_dir)}",
        "train: images/train",
        "val: images/val",
        "",
        "# 10 BDD100K detection classes",
        f"nc: {len(BDD100K_CLASSES)}",
        "names:",
    ]
    for idx, name in enumerate(BDD100K_CLASSES):
        lines.append(f"  {idx}: {name}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
