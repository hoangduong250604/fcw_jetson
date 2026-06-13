"""
Batch Runner: Generate eval CSVs for all KITTI drives with tracklets.
Run this first, then use evaluate_fcw.py --skip-run for analysis.

Usage:
  python run_eval_batch.py           # Run all drives
  python run_eval_batch.py --max 5   # Run first 5 drives only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
FCW_DIR = ROOT_DIR / "fcw-system"
BUILD_DIR = FCW_DIR / "build"
EXE_PATH = BUILD_DIR / "fcw_system.exe"
VIDEO_DIR = ROOT_DIR / "video_data"
KITTI_DIR = ROOT_DIR / "KITTI"
TRACKLETS_DIR = ROOT_DIR / "KITTI_tracklets"
RESULTS_DIR = ROOT_DIR / "results" / "evaluation"


def find_drives():
    """Find drives with both video and tracklets."""
    drives = []
    for tracklet_folder in sorted(TRACKLETS_DIR.iterdir()):
        if not tracklet_folder.is_dir() or not tracklet_folder.name.endswith('_tracklets'):
            continue
        drive_name = tracklet_folder.name.replace('_tracklets', '')
        video_path = VIDEO_DIR / f"{drive_name}_sync.avi"
        if video_path.exists():
            drives.append((drive_name, str(video_path)))
    return drives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=0, help="Max drives to process (0=all)")
    args = parser.parse_args()

    if not EXE_PATH.exists():
        print(f"[ERROR] Build FCW first: {EXE_PATH}")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    drives = find_drives()
    if args.max > 0:
        drives = drives[:args.max]

    print(f"[INFO] Processing {len(drives)} drives...")

    # Set up environment
    env = os.environ.copy()
    opencv_bin = r"C:\opencv-mingw\x64\mingw\bin"
    onnxrt_lib = str(ROOT_DIR / "onnxruntime" / "lib")
    extra_paths = []
    if os.path.isdir(opencv_bin):
        extra_paths.append(opencv_bin)
    if os.path.isdir(onnxrt_lib):
        extra_paths.append(onnxrt_lib)
    if extra_paths:
        env["PATH"] = ";".join(extra_paths) + ";" + env.get("PATH", "")

    success = 0
    failed = 0

    for i, (drive_name, video_path) in enumerate(drives):
        csv_path = RESULTS_DIR / f"{drive_name}_eval.csv"
        
        # Skip if already exists
        if csv_path.exists() and csv_path.stat().st_size > 100:
            print(f"  [{i+1}/{len(drives)}] {drive_name} - SKIP (CSV exists)")
            success += 1
            continue

        print(f"  [{i+1}/{len(drives)}] {drive_name} - Running...", end=" ", flush=True)

        cmd = [
            str(EXE_PATH),
            "--video", video_path,
            "--kitti-root", str(KITTI_DIR),
            "--eval-log", str(csv_path),
        ]

        try:
            result = subprocess.run(
                cmd, cwd=str(FCW_DIR), env=env,
                timeout=300, capture_output=True, text=True
            )
            if csv_path.exists() and csv_path.stat().st_size > 100:
                print("OK")
                success += 1
            else:
                print(f"FAILED (no CSV, exit={result.returncode})")
                failed += 1
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\n[DONE] Success: {success}, Failed: {failed}, Total: {len(drives)}")
    print(f"[NEXT] Run: python evaluate_fcw.py --skip-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
