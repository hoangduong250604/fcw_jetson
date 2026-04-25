"""Quick scan to find all unique class IDs in the BDD100K YOLO labels."""
import os
import glob
from collections import Counter

label_dir = r"c:\VScode\KLTN\BDD100K\VOCdevkit\labels\train"
files = glob.glob(os.path.join(label_dir, "*.txt"))

class_counts = Counter()
total_boxes = 0

for f in files[:2000]:  # Sample 2000 files
    with open(f) as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_counts[int(parts[0])] += 1
                total_boxes += 1

print(f"Scanned {min(len(files), 2000)} files, {total_boxes} boxes")
print(f"Unique class IDs: {sorted(class_counts.keys())}")
print(f"Total classes: {len(class_counts)}")
print("\nClass distribution:")
for cid in sorted(class_counts.keys()):
    print(f"  Class {cid}: {class_counts[cid]} boxes")
