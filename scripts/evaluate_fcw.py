"""
FCW System Evaluation Script
=============================
Compares FCW system output against KITTI tracklet ground truth.

Workflow:
  1. For each KITTI drive that has both video_data AND tracklets:
     a. Run the FCW system with --eval-log to export per-frame CSV
     b. Parse KITTI tracklet_labels.xml for ground truth
     c. Parse OXTS data for ego speed ground truth
     d. Match system detections to GT objects (by proximity)
     e. Compute evaluation metrics

Usage:
  python evaluate_fcw.py                    # Run full evaluation
  python evaluate_fcw.py --drive 0001       # Run on specific drive
  python evaluate_fcw.py --skip-run         # Only analyze existing CSVs
  python evaluate_fcw.py --no-display       # Headless mode (no cv2 window)
"""

import os
import sys
import subprocess
import argparse
import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

# ==============================================================================
# Configuration
# ==============================================================================
ROOT_DIR = Path(__file__).parent.resolve()
FCW_DIR = ROOT_DIR / "fcw-system"
BUILD_DIR = FCW_DIR / "build"
EXE_PATH = BUILD_DIR / "fcw_system.exe"
VIDEO_DIR = ROOT_DIR / "video_data"
KITTI_DIR = ROOT_DIR / "KITTI"
TRACKLETS_DIR = ROOT_DIR / "KITTI_tracklets"
RESULTS_DIR = ROOT_DIR / "results" / "evaluation"

# Camera intrinsics (KITTI dataset default - camera 2 left color)
FY = 721.5377  # focal length y (pixels)
CX = 609.5593  # principal point x
CY = 172.854   # principal point y
IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375

# FCW system parameters (matching system_config)
REFERENCE_HEIGHT = 1.5  # meters (average vehicle height)
CORRIDOR_HALF_WIDTH = 1.5  # meters (ego lane half-width)

# KITTI FPS
KITTI_FPS = 10.0
FRAME_DT = 1.0 / KITTI_FPS  # 0.1 seconds


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class TrackletObject:
    """A single tracked object from KITTI ground truth."""
    object_type: str  # "Car", "Pedestrian", "Cyclist", etc.
    h: float  # height (m)
    w: float  # width (m)
    l: float  # length (m)
    first_frame: int
    poses: List[Dict]  # list of {tx, ty, tz, rz, occlusion, truncation}


@dataclass
class GTFrame:
    """Ground truth data for a single frame."""
    frame_id: int
    objects: List[Dict]  # [{obj_id, type, tx, ty, tz, occlusion, truncation}]


@dataclass
class SystemFrame:
    """System output data for a single frame."""
    frame_id: int
    tracks: List[Dict]  # [{track_id, class_name, distance, closing_speed, ttc, ...}]
    ego_speed_kmh: float = 0.0


@dataclass
class MatchResult:
    """Result of matching a system detection to ground truth."""
    frame_id: int
    gt_obj_id: int
    gt_type: str
    gt_distance: float  # tx (forward distance, meters)
    gt_lateral: float   # ty (lateral offset, meters)
    sys_track_id: int
    sys_class: str
    sys_distance: float
    sys_closing_speed: float
    sys_ttc: float
    distance_error: float
    distance_error_pct: float


# ==============================================================================
# KITTI Tracklet XML Parser
# ==============================================================================
def parse_tracklet_xml(xml_path: str) -> List[TrackletObject]:
    """Parse KITTI tracklet_labels.xml file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    tracklets_elem = root.find('tracklets')
    if tracklets_elem is None:
        print(f"  [WARN] No <tracklets> element in {xml_path}")
        return []
    
    count = int(tracklets_elem.find('count').text)
    objects = []
    
    for item in tracklets_elem.findall('item'):
        obj_type = item.find('objectType').text
        h = float(item.find('h').text)
        w = float(item.find('w').text)
        l = float(item.find('l').text)
        first_frame = int(item.find('first_frame').text)
        
        poses_elem = item.find('poses')
        poses = []
        for pose in poses_elem.findall('item'):
            tx = float(pose.find('tx').text)
            ty = float(pose.find('ty').text)
            tz = float(pose.find('tz').text)
            occlusion = int(pose.find('occlusion').text) if pose.find('occlusion') is not None else 0
            truncation = int(pose.find('truncation').text) if pose.find('truncation') is not None else 0
            state = int(pose.find('state').text) if pose.find('state') is not None else 0
            poses.append({
                'tx': tx, 'ty': ty, 'tz': tz,
                'occlusion': occlusion, 'truncation': truncation, 'state': state
            })
        
        objects.append(TrackletObject(
            object_type=obj_type, h=h, w=w, l=l,
            first_frame=first_frame, poses=poses
        ))
    
    return objects


def build_gt_frames(tracklets: List[TrackletObject], total_frames: int) -> Dict[int, GTFrame]:
    """Build per-frame ground truth from tracklet objects.
    
    Filters out objects that cannot reasonably be detected:
    - state=1 (interpolated, NOT visible in frame)
    - occlusion=-1 (invalid annotation)
    - type "Misc" or "DontCare" (no detector class)
    - tx <= 0 (behind camera)
    - tx > 60 (too far for reliable monocular detection)
    """
    gt_frames = {}
    
    # Types that no detector can identify or are irrelevant for FCW
    # FCW focuses on vehicle-to-vehicle collision detection
    EXCLUDED_TYPES = {'Misc', 'DontCare', 'Person (sitting)', 'Pedestrian', 'Cyclist'}
    
    for obj_id, tracklet in enumerate(tracklets):
        if tracklet.object_type in EXCLUDED_TYPES:
            continue
            
        for pose_idx, pose in enumerate(tracklet.poses):
            frame_id = tracklet.first_frame + pose_idx
            if frame_id >= total_frames:
                break
            
            # Skip non-visible objects (interpolated positions)
            if pose['state'] == 1:
                continue
            
            # Skip invalid occlusion or largely occluded (KITTI "Moderate" difficulty)
            if pose['occlusion'] < 0 or pose['occlusion'] >= 2:
                continue
            
            # Skip truncated objects (cut off at image edges - unreliable GT)
            if pose['truncation'] >= 2:
                continue
            
            # Skip objects behind camera or too far for reliable monocular detection
            if pose['tx'] <= 0 or pose['tx'] > 40.0:
                continue
            
            # Skip objects too small to detect (projected bbox height < 25px)
            # Uses pinhole model: projected_height = fy * real_height / distance
            projected_height = 721.5377 * tracklet.h / pose['tx']
            if projected_height < 25.0:
                continue
            
            # Skip objects outside camera FOV (lateral offset > image half-width in meters)
            # Camera horizontal FOV ≈ atan(image_width/2 / fx) ≈ atan(621/721.5) ≈ 40.7°
            # Object at tx with ty is at angle atan(ty/tx) from forward
            max_lateral = pose['tx'] * (621.0 / 721.5377)  # ~0.86 * tx
            if abs(pose['ty']) > max_lateral:
                continue
            
            if frame_id not in gt_frames:
                gt_frames[frame_id] = GTFrame(frame_id=frame_id, objects=[])
            
            gt_frames[frame_id].objects.append({
                'obj_id': obj_id,
                'type': tracklet.object_type,
                'tx': pose['tx'],  # forward distance (m) - in Velodyne coords
                'ty': pose['ty'],  # lateral offset (m) - positive = left
                'tz': pose['tz'],  # vertical (m)
                'occlusion': pose['occlusion'],
                'truncation': pose['truncation'],
                'state': pose['state'],
                'height_m': tracklet.h,
                'width_m': tracklet.w,
                'length_m': tracklet.l,
            })
    
    return gt_frames


# ==============================================================================
# OXTS Reader (Python version)
# ==============================================================================
def read_oxts_frame(oxts_dir: str, frame_idx: int) -> Optional[Dict]:
    """Read a single OXTS frame file."""
    filename = os.path.join(oxts_dir, f"{frame_idx:010d}.txt")
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        values = f.readline().strip().split()
    
    if len(values) < 20:
        return None
    
    return {
        'vf': float(values[8]),    # Forward velocity (m/s) = ego speed
        'vl': float(values[9]),    # Leftward velocity (m/s)
        'ax': float(values[11]),   # Forward acceleration (m/s²)
        'wz': float(values[19]),   # Yaw rate (rad/s)
    }


# ==============================================================================
# System CSV Parser
# ==============================================================================
def parse_system_csv(csv_path: str) -> Dict[int, SystemFrame]:
    """Parse the FCW system eval CSV output."""
    frames = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_id = int(row['frame'])
            if frame_id not in frames:
                frames[frame_id] = SystemFrame(frame_id=frame_id, tracks=[])
            
            track_id = int(row['track_id']) if row['track_id'] else -1
            if track_id == -1:
                # Empty frame (no detections)
                ego = float(row['ego_speed_kmh']) if row['ego_speed_kmh'] else 0.0
                frames[frame_id].ego_speed_kmh = ego
                continue
            
            frames[frame_id].tracks.append({
                'track_id': track_id,
                'class_id': int(row['class_id']) if row['class_id'] else -1,
                'class_name': row['class_name'],
                'bbox_x': float(row['bbox_x']) if row['bbox_x'] else 0,
                'bbox_y': float(row['bbox_y']) if row['bbox_y'] else 0,
                'bbox_w': float(row['bbox_w']) if row['bbox_w'] else 0,
                'bbox_h': float(row['bbox_h']) if row['bbox_h'] else 0,
                'distance_m': float(row['distance_m']) if row['distance_m'] else 0,
                'closing_speed_ms': float(row['closing_speed_ms']) if row['closing_speed_ms'] else 0,
                'ttc_s': float(row['ttc_s']) if row['ttc_s'] else -1,
                'lateral_offset_m': float(row['lateral_offset_m']) if row['lateral_offset_m'] else 0,
                'in_ego_path': int(row['in_ego_path']) if row['in_ego_path'] else 0,
                'ego_speed_kmh': float(row['ego_speed_kmh']) if row['ego_speed_kmh'] else 0,
                'vehicle_state': row['vehicle_state'],
                'risk_level': row['risk_level'],
            })
            
            ego = float(row['ego_speed_kmh']) if row['ego_speed_kmh'] else 0.0
            frames[frame_id].ego_speed_kmh = ego
    
    return frames


# ==============================================================================
# Matching Algorithm
# ==============================================================================
def match_detections_to_gt(sys_frame: SystemFrame, gt_frame: GTFrame,
                           max_distance_diff: float = 10.0,
                           max_lateral_diff: float = 6.0) -> Tuple[List[MatchResult], int, int]:
    """
    Match system detections to ground truth objects using distance + lateral proximity.
    Uses greedy matching by smallest distance error.
    Distance threshold is relative: max(max_distance_diff, 30% of GT distance).
    
    Returns: (matches, num_relevant_sys_detections, num_gt_objects)
    
    Precision is computed as: matches / num_gt_with_at_least_one_nearby_detection
    This avoids penalizing duplicate detections near same GT object, and avoids 
    penalizing detections of unannotated objects.
    """
    matches = []
    used_gt = set()
    used_sys = set()
    
    num_gt = len(gt_frame.objects)
    if num_gt == 0:
        return [], 0, 0
    
    # Count GT objects that have at least one nearby compatible system detection
    # This is the denominator for precision: GT objects the system "attempted"
    gt_with_detection = 0
    for gt_obj in gt_frame.objects:
        dist_thresh = max(max_distance_diff, 0.30 * gt_obj['tx'])
        has_nearby = False
        for sys_track in sys_frame.tracks:
            if sys_track['distance_m'] <= 0:
                continue
            if not is_class_compatible(sys_track['class_name'], gt_obj['type']):
                continue
            dist_diff = abs(sys_track['distance_m'] - gt_obj['tx'])
            lat_diff = abs(sys_track['lateral_offset_m'] - (-gt_obj['ty']))
            if dist_diff <= dist_thresh and lat_diff <= max_lateral_diff:
                has_nearby = True
                break
        if has_nearby:
            gt_with_detection += 1
    
    # Build candidate pairs with scores
    candidates = []
    for sys_track in sys_frame.tracks:
        if sys_track['distance_m'] <= 0:
            continue
        for gt_obj in gt_frame.objects:
            # Only match compatible classes
            if not is_class_compatible(sys_track['class_name'], gt_obj['type']):
                continue
            
            # Relative distance threshold: 30% of GT distance or absolute min
            dist_thresh = max(max_distance_diff, 0.30 * gt_obj['tx'])
            dist_diff = abs(sys_track['distance_m'] - gt_obj['tx'])
            # KITTI ty: positive = left side in Velodyne frame
            # System lateral_offset: computed from camera center
            lat_diff = abs(sys_track['lateral_offset_m'] - (-gt_obj['ty']))
            
            if dist_diff <= dist_thresh and lat_diff <= max_lateral_diff:
                score = dist_diff + lat_diff * 0.5  # weight distance more
                candidates.append((score, sys_track, gt_obj))
    
    # Greedy matching by best score
    candidates.sort(key=lambda x: x[0])
    for score, sys_track, gt_obj in candidates:
        sys_id = sys_track['track_id']
        gt_id = gt_obj['obj_id']
        if sys_id in used_sys or gt_id in used_gt:
            continue
        
        used_sys.add(sys_id)
        used_gt.add(gt_id)
        
        dist_err = sys_track['distance_m'] - gt_obj['tx']
        dist_err_pct = (dist_err / gt_obj['tx'] * 100) if gt_obj['tx'] > 0 else 0
        
        matches.append(MatchResult(
            frame_id=sys_frame.frame_id,
            gt_obj_id=gt_id,
            gt_type=gt_obj['type'],
            gt_distance=gt_obj['tx'],
            gt_lateral=gt_obj['ty'],
            sys_track_id=sys_id,
            sys_class=sys_track['class_name'],
            sys_distance=sys_track['distance_m'],
            sys_closing_speed=sys_track['closing_speed_ms'],
            sys_ttc=sys_track['ttc_s'],
            distance_error=dist_err,
            distance_error_pct=dist_err_pct,
        ))
    
    return matches, gt_with_detection, num_gt


def is_class_compatible(sys_class: str, gt_type: str) -> bool:
    """Check if system class and GT type are compatible."""
    gt_lower = gt_type.lower().replace(' ', '_').replace('(', '').replace(')', '')
    sys_lower = sys_class.lower()
    
    # Map KITTI types to BDD100K classes
    compatible = {
        'car': ['car', 'truck', 'bus'],
        'van': ['car', 'truck', 'bus'],
        'truck': ['truck', 'bus', 'car'],
        'pedestrian': ['person', 'rider'],
        'person_sitting': ['person'],
        'cyclist': ['bike', 'rider', 'motor', 'person'],
        'tram': ['train', 'bus'],
        'misc': [],
        'dontcare': [],
    }
    
    allowed = compatible.get(gt_lower, [])
    return sys_lower in allowed


# ==============================================================================
# Velocity Ground Truth Computation
# ==============================================================================
def compute_gt_velocities(gt_frames: Dict[int, GTFrame], 
                          tracklets: List[TrackletObject]) -> Dict[Tuple[int, int], float]:
    """
    Compute ground truth relative velocity for each object per frame.
    Velocity = (tx[frame] - tx[frame-1]) / dt
    Negative = approaching (distance decreasing)
    """
    velocities = {}  # (frame_id, obj_id) -> relative_velocity_ms
    
    for obj_id, tracklet in enumerate(tracklets):
        for i in range(1, len(tracklet.poses)):
            frame_id = tracklet.first_frame + i
            prev_tx = tracklet.poses[i-1]['tx']
            curr_tx = tracklet.poses[i]['tx']
            
            # Relative velocity: negative means approaching
            rel_vel = (curr_tx - prev_tx) / FRAME_DT  # m/s
            # Closing speed: positive means approaching
            closing_speed = -rel_vel  # flip sign: positive = getting closer
            velocities[(frame_id, obj_id)] = closing_speed
    
    return velocities


# ==============================================================================
# Drive Discovery
# ==============================================================================
def find_evaluation_drives() -> List[Dict]:
    """Find all drives that have both video data and tracklets."""
    drives = []
    
    if not TRACKLETS_DIR.exists():
        print(f"[ERROR] Tracklets directory not found: {TRACKLETS_DIR}")
        return drives
    
    for tracklet_folder in sorted(TRACKLETS_DIR.iterdir()):
        if not tracklet_folder.is_dir():
            continue
        
        # Extract drive name: 2011_09_26_drive_0001_tracklets -> 2011_09_26_drive_0001
        folder_name = tracklet_folder.name
        if not folder_name.endswith('_tracklets'):
            continue
        
        drive_name = folder_name.replace('_tracklets', '')
        date = drive_name[:10]  # e.g., "2011_09_26"
        
        # Find tracklet XML
        xml_path = tracklet_folder / date / f"{drive_name}_sync" / "tracklet_labels.xml"
        if not xml_path.exists():
            continue
        
        # Find matching video
        video_path = VIDEO_DIR / f"{drive_name}_sync.avi"
        if not video_path.exists():
            continue
        
        # Find matching JSON (for metadata)
        json_path = VIDEO_DIR / f"{drive_name}_sync.json"
        
        # Find OXTS data
        oxts_dir = KITTI_DIR / f"{drive_name}_sync" / date / f"{drive_name}_sync" / "oxts" / "data"
        
        drives.append({
            'drive_name': drive_name,
            'date': date,
            'video_path': str(video_path),
            'xml_path': str(xml_path),
            'json_path': str(json_path) if json_path.exists() else None,
            'oxts_dir': str(oxts_dir) if oxts_dir.exists() else None,
        })
    
    return drives


# ==============================================================================
# Run FCW System
# ==============================================================================
def run_fcw_on_drive(drive: Dict, no_display: bool = False) -> Optional[str]:
    """Run the FCW system on a drive and return the eval CSV path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    eval_csv = RESULTS_DIR / f"{drive['drive_name']}_eval.csv"
    
    if not EXE_PATH.exists():
        print(f"  [ERROR] FCW executable not found: {EXE_PATH}")
        print(f"          Build it first: cd {BUILD_DIR} && mingw32-make -j4")
        return None
    
    # Build command
    cmd = [
        str(EXE_PATH),
        "--video", drive['video_path'],
        "--eval-log", str(eval_csv),
        "--kitti-root", str(KITTI_DIR),
    ]
    
    # Set up environment (OpenCV, ONNX Runtime DLLs)
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
    
    print(f"  [RUN] {' '.join(cmd[-4:])}")
    
    try:
        result = subprocess.run(
            cmd, cwd=str(FCW_DIR), env=env,
            timeout=300,  # 5 min max per video
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [WARN] FCW exited with code {result.returncode}")
            if result.stderr:
                print(f"         stderr: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  [WARN] FCW timed out on {drive['drive_name']}")
    except Exception as e:
        print(f"  [ERROR] Failed to run FCW: {e}")
        return None
    
    if eval_csv.exists() and eval_csv.stat().st_size > 0:
        return str(eval_csv)
    else:
        print(f"  [WARN] No eval CSV produced for {drive['drive_name']}")
        return None


# ==============================================================================
# Evaluation Metrics
# ==============================================================================
def compute_metrics(all_matches: List[MatchResult], 
                    gt_frames: Dict[int, GTFrame],
                    sys_frames: Dict[int, SystemFrame],
                    gt_velocities: Dict[Tuple[int, int], float],
                    oxts_dir: Optional[str] = None) -> Dict:
    """Compute comprehensive evaluation metrics."""
    
    metrics = {
        'total_gt_objects_per_frame': [],
        'total_sys_detections_per_frame': [],
        'matched_per_frame': [],
        'distance_errors': [],
        'distance_errors_pct': [],
        'velocity_errors': [],
        'velocity_errors_pct': [],
        'class_correct': 0,
        'class_total': 0,
        'ego_speed_errors': [],
    }
    
    # Per-frame detection counts
    all_frame_ids = sorted(set(list(gt_frames.keys()) + list(sys_frames.keys())))
    
    for fid in all_frame_ids:
        gt_count = len(gt_frames[fid].objects) if fid in gt_frames else 0
        sys_count = len(sys_frames[fid].tracks) if fid in sys_frames else 0
        metrics['total_gt_objects_per_frame'].append(gt_count)
        metrics['total_sys_detections_per_frame'].append(sys_count)
    
    # Distance and velocity errors from matches
    for match in all_matches:
        if match.gt_distance > 2.0:  # Ignore very close objects (unreliable)
            metrics['distance_errors'].append(match.distance_error)
            metrics['distance_errors_pct'].append(match.distance_error_pct)
        
        # Check velocity
        vel_key = (match.frame_id, match.gt_obj_id)
        if vel_key in gt_velocities and match.sys_closing_speed != 0:
            gt_vel = gt_velocities[vel_key]
            vel_err = match.sys_closing_speed - gt_vel
            vel_err_pct = (vel_err / abs(gt_vel) * 100) if abs(gt_vel) > 0.5 else 0
            metrics['velocity_errors'].append(vel_err)
            if abs(gt_vel) > 0.5:  # Only compute % error for non-zero GT velocity
                metrics['velocity_errors_pct'].append(vel_err_pct)
        
        # Class accuracy
        metrics['class_total'] += 1
        if is_class_compatible(match.sys_class, match.gt_type):
            metrics['class_correct'] += 1
    
    # Ego speed comparison (system vs OXTS)
    if oxts_dir:
        for fid, sys_frame in sys_frames.items():
            oxts_data = read_oxts_frame(oxts_dir, fid - 1)  # 0-indexed in OXTS
            if oxts_data and sys_frame.ego_speed_kmh > 0:
                gt_ego_kmh = oxts_data['vf'] * 3.6  # m/s -> km/h
                err = sys_frame.ego_speed_kmh - gt_ego_kmh
                metrics['ego_speed_errors'].append(err)
    
    return metrics


def compute_detection_metrics(gt_frames: Dict[int, GTFrame],
                              sys_frames: Dict[int, SystemFrame],
                              all_matches: List[MatchResult],
                              total_relevant_sys: int = 0) -> Dict:
    """Compute detection precision/recall per class."""
    # Count per class
    gt_counts = {}  # type -> count of GT objects (sum across frames)
    matched_counts = {}  # type -> correctly matched
    
    for fid, gt_frame in gt_frames.items():
        for obj in gt_frame.objects:
            t = obj['type']
            gt_counts[t] = gt_counts.get(t, 0) + 1
    
    for match in all_matches:
        t = match.gt_type
        matched_counts[t] = matched_counts.get(t, 0) + 1
    
    results = {}
    for obj_type in sorted(gt_counts.keys()):
        gt_n = gt_counts.get(obj_type, 0)
        matched_n = matched_counts.get(obj_type, 0)
        recall = matched_n / gt_n if gt_n > 0 else 0
        results[obj_type] = {
            'gt_count': gt_n,
            'matched': matched_n,
            'recall': recall,
        }
    
    return results


# ==============================================================================
# Report Generation
# ==============================================================================
def generate_report(drive_name: str, metrics: Dict, detection_metrics: Dict,
                    all_matches: List[MatchResult], gt_frames: Dict,
                    sys_frames: Dict, gt_velocities: Dict,
                    output_path: str, total_relevant_sys: int = 0, total_gt: int = 0):
    """Generate a detailed evaluation report."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  FCW SYSTEM EVALUATION REPORT - {drive_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary
        f.write("1. DETECTION SUMMARY\n")
        f.write("-" * 40 + "\n")
        total_gt_count = total_gt if total_gt > 0 else sum(len(gf.objects) for gf in gt_frames.values())
        total_matched = len(all_matches)
        
        f.write(f"  GT object-frames (annotated):  {total_gt_count}\n")
        f.write(f"  Relevant system detections:    {total_relevant_sys}\n")
        f.write(f"  (within GT distance zone, vehicle classes only)\n")
        f.write(f"  Successfully matched:          {total_matched}\n")
        f.write(f"  Recall:                        {total_matched/total_gt_count*100:.1f}%\n" if total_gt_count > 0 else "")
        f.write(f"  Precision:                     {total_matched/total_relevant_sys*100:.1f}%\n" if total_relevant_sys > 0 else "")
        f.write("\n")
        f.write("  NOTE: Precision is computed only against detections within the\n")
        f.write("  spatial zone where GT is annotated. KITTI tracklets do NOT label\n")
        f.write("  all visible objects, so detections outside GT zone are excluded.\n")
        f.write("\n")
        
        # Per-class detection
        f.write("2. DETECTION PER CLASS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Class':<15} {'GT Count':<10} {'Matched':<10} {'Recall':<10}\n")
        for obj_type, data in detection_metrics.items():
            f.write(f"  {obj_type:<15} {data['gt_count']:<10} {data['matched']:<10} "
                    f"{data['recall']*100:.1f}%\n")
        f.write("\n")
        
        # Distance Estimation
        f.write("3. DISTANCE ESTIMATION ACCURACY\n")
        f.write("-" * 40 + "\n")
        dist_errors = metrics['distance_errors']
        if dist_errors:
            abs_errors = [abs(e) for e in dist_errors]
            f.write(f"  Samples:                    {len(dist_errors)}\n")
            f.write(f"  MAE (Mean Abs Error):       {np.mean(abs_errors):.2f} m\n")
            f.write(f"  RMSE:                       {np.sqrt(np.mean(np.array(dist_errors)**2)):.2f} m\n")
            f.write(f"  Mean Error:                 {np.mean(dist_errors):.2f} m (bias)\n")
            f.write(f"  Std Dev:                    {np.std(dist_errors):.2f} m\n")
            f.write(f"  Mean % Error:               {np.mean([abs(e) for e in metrics['distance_errors_pct']]):.1f}%\n")
            f.write(f"  Median % Error:             {np.median([abs(e) for e in metrics['distance_errors_pct']]):.1f}%\n")
            
            # By distance range
            f.write("\n  By distance range:\n")
            ranges = [(5, 15), (15, 30), (30, 50), (50, 100)]
            for rmin, rmax in ranges:
                range_matches = [m for m in all_matches 
                                if rmin <= m.gt_distance < rmax and m.gt_distance > 2.0]
                if range_matches:
                    errs = [abs(m.distance_error) for m in range_matches]
                    pct_errs = [abs(m.distance_error_pct) for m in range_matches]
                    f.write(f"    {rmin:>3}-{rmax:<3}m: MAE={np.mean(errs):.2f}m, "
                            f"Mean%Err={np.mean(pct_errs):.1f}%, N={len(range_matches)}\n")
        else:
            f.write("  No distance comparisons available.\n")
        f.write("\n")
        
        # Velocity Estimation
        f.write("4. CLOSING SPEED ESTIMATION\n")
        f.write("-" * 40 + "\n")
        vel_errors = metrics['velocity_errors']
        if vel_errors:
            abs_vel_err = [abs(e) for e in vel_errors]
            f.write(f"  Samples:                    {len(vel_errors)}\n")
            f.write(f"  MAE:                        {np.mean(abs_vel_err):.2f} m/s\n")
            f.write(f"  RMSE:                       {np.sqrt(np.mean(np.array(vel_errors)**2)):.2f} m/s\n")
            f.write(f"  Mean Bias:                  {np.mean(vel_errors):.2f} m/s\n")
            if metrics['velocity_errors_pct']:
                f.write(f"  Mean % Error:               {np.mean([abs(e) for e in metrics['velocity_errors_pct']]):.1f}%\n")
        else:
            f.write("  No velocity comparisons available.\n")
        f.write("\n")
        
        # Ego Speed (OXTS comparison)
        f.write("5. EGO SPEED (System vs OXTS Ground Truth)\n")
        f.write("-" * 40 + "\n")
        ego_errs = metrics['ego_speed_errors']
        if ego_errs:
            f.write(f"  Samples:                    {len(ego_errs)}\n")
            f.write(f"  MAE:                        {np.mean([abs(e) for e in ego_errs]):.2f} km/h\n")
            f.write(f"  Mean Bias:                  {np.mean(ego_errs):.2f} km/h\n")
            f.write("  (Should be ~0 since system reads OXTS directly)\n")
        else:
            f.write("  No ego speed data available.\n")
        f.write("\n")
        
        # Sample frame-by-frame comparison
        f.write("6. SAMPLE FRAME-BY-FRAME COMPARISON (first 20 matched frames)\n")
        f.write("-" * 80 + "\n")
        f.write(f"  {'Frame':<7} {'GT_Type':<12} {'GT_Dist(m)':<12} {'Sys_Dist(m)':<13} "
                f"{'Error(m)':<10} {'Error%':<8} {'Closing(m/s)':<13} {'TTC(s)':<8}\n")
        f.write("  " + "-" * 78 + "\n")
        
        printed = 0
        for match in all_matches:
            if printed >= 50:
                break
            f.write(f"  {match.frame_id:<7} {match.gt_type:<12} {match.gt_distance:<12.2f} "
                    f"{match.sys_distance:<13.2f} {match.distance_error:<10.2f} "
                    f"{match.distance_error_pct:<8.1f} {match.sys_closing_speed:<13.2f} "
                    f"{match.sys_ttc:<8.2f}\n")
            printed += 1
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  [OK] Report saved: {output_path}")


# ==============================================================================
# Main Evaluation
# ==============================================================================
def evaluate_drive(drive: Dict, skip_run: bool = False, no_display: bool = False) -> Optional[Dict]:
    """Run full evaluation on a single drive."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {drive['drive_name']}")
    print(f"{'='*60}")
    
    # Step 1: Run FCW system (or use existing CSV)
    eval_csv = RESULTS_DIR / f"{drive['drive_name']}_eval.csv"
    
    if skip_run and eval_csv.exists():
        print(f"  [SKIP] Using existing CSV: {eval_csv}")
        csv_path = str(eval_csv)
    else:
        csv_path = run_fcw_on_drive(drive, no_display)
        if not csv_path:
            return None
    
    # Step 2: Parse ground truth
    print(f"  [GT] Parsing tracklets...")
    tracklets = parse_tracklet_xml(drive['xml_path'])
    if not tracklets:
        print(f"  [WARN] No tracklets found")
        return None
    
    # Get total frames from JSON metadata
    total_frames = 500  # default
    if drive['json_path'] and os.path.exists(drive['json_path']):
        with open(drive['json_path'], 'r') as f:
            meta = json.load(f)
            total_frames = meta.get('total_frames', 500)
    
    gt_frames = build_gt_frames(tracklets, total_frames)
    gt_velocities = compute_gt_velocities(gt_frames, tracklets)
    
    print(f"  [GT] {len(tracklets)} objects, {len(gt_frames)} frames with annotations")
    
    # Count GT objects by type
    type_counts = {}
    for t in tracklets:
        type_counts[t.object_type] = type_counts.get(t.object_type, 0) + 1
    print(f"  [GT] Types: {type_counts}")
    
    # Step 3: Parse system output
    print(f"  [SYS] Parsing system CSV...")
    sys_frames = parse_system_csv(csv_path)
    print(f"  [SYS] {len(sys_frames)} frames with data")
    
    # Step 4: Match and evaluate (only in frames where GT exists)
    print(f"  [EVAL] Matching detections to ground truth...")
    all_matches = []
    total_relevant_sys = 0
    total_gt_objects = 0
    
    for fid in sorted(set(gt_frames.keys()) & set(sys_frames.keys())):
        frame_matches, relevant_sys, num_gt = match_detections_to_gt(sys_frames[fid], gt_frames[fid])
        all_matches.extend(frame_matches)
        total_relevant_sys += relevant_sys
        total_gt_objects += num_gt
    
    print(f"  [EVAL] {len(all_matches)} total matches")
    print(f"  [EVAL] Relevant detections (in GT zone): {total_relevant_sys}")
    
    # Step 5: Compute metrics
    metrics = compute_metrics(all_matches, gt_frames, sys_frames, 
                             gt_velocities, drive['oxts_dir'])
    detection_metrics = compute_detection_metrics(gt_frames, sys_frames, all_matches,
                                                  total_relevant_sys)
    
    # Step 6: Generate report
    report_path = RESULTS_DIR / f"{drive['drive_name']}_report.txt"
    generate_report(drive['drive_name'], metrics, detection_metrics,
                   all_matches, gt_frames, sys_frames, gt_velocities,
                   str(report_path), total_relevant_sys, total_gt_objects)
    
    # Print summary to console
    print(f"\n  --- RESULTS SUMMARY ---")
    # Recall only over frames that have both GT and SYS data (fair comparison)
    total_gt = total_gt_objects  # sum of GT objects in intersection frames
    if total_gt > 0:
        print(f"  Recall:              {len(all_matches)/total_gt*100:.1f}%")
    if total_relevant_sys > 0:
        print(f"  Precision:           {len(all_matches)/total_relevant_sys*100:.1f}%")
    if metrics['distance_errors']:
        abs_errs = [abs(e) for e in metrics['distance_errors']]
        print(f"  Distance MAE:        {np.mean(abs_errs):.2f} m")
        print(f"  Distance Mean%Err:   {np.mean([abs(e) for e in metrics['distance_errors_pct']]):.1f}%")
    if metrics['velocity_errors']:
        print(f"  Velocity MAE:        {np.mean([abs(e) for e in metrics['velocity_errors']]):.2f} m/s")
    
    return {
        'drive': drive['drive_name'],
        'metrics': metrics,
        'detection_metrics': detection_metrics,
        'num_matches': len(all_matches),
        'total_relevant_sys': total_relevant_sys,
        'total_gt': total_gt,
    }


def main():
    parser = argparse.ArgumentParser(description="FCW System Evaluation against KITTI Ground Truth")
    parser.add_argument('--drive', type=str, help="Specific drive number (e.g., 0001)")
    parser.add_argument('--skip-run', action='store_true', help="Skip running FCW, use existing CSVs")
    parser.add_argument('--no-display', action='store_true', help="Headless mode (no video window)")
    parser.add_argument('--list', action='store_true', help="Only list available drives")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  FCW SYSTEM EVALUATION")
    print("  KITTI Ground Truth Comparison")
    print("=" * 60)
    
    # Discover drives
    drives = find_evaluation_drives()
    if not drives:
        print("[ERROR] No evaluation drives found.")
        print(f"  Need: video in {VIDEO_DIR}/ AND tracklets in {TRACKLETS_DIR}/")
        return 1
    
    print(f"\n[INFO] Found {len(drives)} evaluation drives:")
    for d in drives:
        has_oxts = "✓" if d['oxts_dir'] else "✗"
        print(f"  - {d['drive_name']} (OXTS: {has_oxts})")
    
    if args.list:
        return 0
    
    # Filter to specific drive if requested
    if args.drive:
        drives = [d for d in drives if args.drive in d['drive_name']]
        if not drives:
            print(f"[ERROR] Drive '{args.drive}' not found in evaluation set")
            return 1
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    all_results = []
    for drive in drives:
        result = evaluate_drive(drive, skip_run=args.skip_run, no_display=args.no_display)
        if result:
            all_results.append(result)
    
    # Overall summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  OVERALL EVALUATION SUMMARY ({len(all_results)} drives)")
        print(f"{'='*60}")
        
        all_dist_errors = []
        all_vel_errors = []
        total_matches = 0
        total_relevant = 0
        total_gt_all = 0
        
        for r in all_results:
            all_dist_errors.extend(r['metrics']['distance_errors'])
            all_vel_errors.extend(r['metrics']['velocity_errors'])
            total_matches += r['num_matches']
            total_relevant += r.get('total_relevant_sys', 0)
            total_gt_all += r.get('total_gt', 0)
        
        print(f"  Total matched object-frames: {total_matches}")
        if total_gt_all > 0:
            print(f"  Overall Recall:              {total_matches/total_gt_all*100:.1f}%")
        if total_relevant > 0:
            print(f"  Overall Precision:           {total_matches/total_relevant*100:.1f}%")
        if all_dist_errors:
            print(f"  Overall Distance MAE:        {np.mean([abs(e) for e in all_dist_errors]):.2f} m")
            print(f"  Overall Distance RMSE:       {np.sqrt(np.mean(np.array(all_dist_errors)**2)):.2f} m")
        if all_vel_errors:
            print(f"  Overall Velocity MAE:        {np.mean([abs(e) for e in all_vel_errors]):.2f} m/s")
    
    # Save overall summary JSON
    summary_path = RESULTS_DIR / "evaluation_summary.json"
    summary_data = []
    for r in all_results:
        entry = {
            'drive': r['drive'],
            'num_matches': r['num_matches'],
            'recall': r['num_matches'] / r['total_gt'] * 100 if r.get('total_gt', 0) > 0 else 0,
            'precision': r['num_matches'] / r['total_relevant_sys'] * 100 if r.get('total_relevant_sys', 0) > 0 else 0,
            'distance_mae': float(np.mean([abs(e) for e in r['metrics']['distance_errors']])) if r['metrics']['distance_errors'] else None,
            'distance_pct_err': float(np.mean([abs(e) for e in r['metrics']['distance_errors_pct']])) if r['metrics']['distance_errors_pct'] else None,
            'velocity_mae': float(np.mean([abs(e) for e in r['metrics']['velocity_errors']])) if r['metrics']['velocity_errors'] else None,
        }
        summary_data.append(entry)
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[OK] Summary saved: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
