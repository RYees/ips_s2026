"""
masking.py
----------
Production Object Masking Pipeline with Integrated Cross-Modal Audit Engine.
Preserves existing pathing architecture while dynamically calculating the height 
threshold using 2D visual edge feedback and 3D point cloud linearity.

Usage:
    python3 offline_case/masking.py img0018 --base offline_case/samples
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import open3d as o3d

# ─────────────────────────────────────────────────────────────────────────────
# Core Boundary Configurations
# ─────────────────────────────────────────────────────────────────────────────
RANSAC_DISTANCE_THRESHOLD = 0.012  # m
MAX_HEIGHT_ABOVE_BELT_MM = 150.0   # Rejects high overhead structural fixtures

# Hard-Coded Global Structural Boundaries
RAIL_LEFT_X_LIMIT = -0.22          
RAIL_RIGHT_X_LIMIT = 0.22          

# Pointwise Local Mass Verification
LOCAL_RADIUS_METERS = 0.010        
MIN_NEIGHBORS_IN_RADIUS = 20       


@dataclass
class CaptureInfo:
    raw_depth_shape: Tuple[int, int] = (0, 0)
    rgb_shape: Tuple[int, int] = (0, 0)
    crop_top: int = 0
    crop_left: int = 0
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    plane_coefs: Optional[Tuple[float, float, float, float]] = None


def parse_info(path: Path) -> CaptureInfo:
    info = CaptureInfo()
    if not path.exists():
        print(f"  [ERROR] Metadata file completely missing at: {path}", flush=True)
        return info
    lines = path.read_text().splitlines()

    def _floats(s):
        return [float(x) for x in re.findall(r"-?[\d]+\.[\d]+(?:[eE][+-]?\d+)?|-?\d+", s)]

    def _ints(s):
        return [int(x) for x in re.findall(r"-\d+|\d+", s)]

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Crop:"):
            v = _ints(s)
            if len(v) >= 4:
                info.crop_top, info.crop_left = v[0], v[2]
        elif s.startswith("Cropped RGB shape:") or s.startswith("RGB shape:"):
            v = _ints(s)
            if len(v) >= 2:
                info.rgb_shape = (v[0], v[1])
        elif s.startswith("Raw depth shape:"):
            v = _ints(s)
            if len(v) >= 2:
                info.raw_depth_shape = (v[0], v[1])
        elif s.startswith("Intrinsics matrix:"):
            r0 = _floats(lines[i + 1]) if i + 1 < len(lines) else []
            r1 = _floats(lines[i + 2]) if i + 2 < len(lines) else []
            if len(r0) >= 3: info.fx, info.cx = r0[0], r0[2]
            if len(r1) >= 3: info.fy, info.cy = r1[1], r1[2]
        elif s.startswith("Plane equation:"):
            v = _floats(s)
            if len(v) >= 4:
                info.plane_coefs = (v[0], v[1], v[2], v[3])
    return info


def run_masking_from_point_cloud(
    pcd: o3d.geometry.PointCloud, info: CaptureInfo, rgb_img: Optional[np.ndarray] = None
) -> tuple[np.ndarray, tuple[float, float, float, float] | None, int, int]:
    target_H, target_W = info.rgb_shape[0], info.rgb_shape[1]
    
    print("\n================================================================================", flush=True)
    print("                STAGE 1: POINT CLOUD ARRAY INITIAL INGESTION", flush=True)
    print("================================================================================", flush=True)
    raw_pts = np.asarray(pcd.points)
    total_input_points = len(raw_pts)
    print(f"  [INGESTION REPORT] Raw spatial points loaded into memory: {total_input_points:,}", flush=True)

    print("\n================================================================================", flush=True)
    print("                STAGE 2: CROSS-MODAL AUDIT ENGINE CALIBRATION", flush=True)
    print("================================================================================", flush=True)
    
    if info.plane_coefs is not None:
        a, b, c, d = info.plane_coefs
        print(f"  [METADATA ANCHOR] Loaded system coefficients: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}", flush=True)
    else:
        ransac_model, _ = pcd.segment_plane(RANSAC_DISTANCE_THRESHOLD, 3, 2000)
        a, b, c, d = ransac_model

    if c > 0: a, b, c, d = -a, -b, -c, -d
    plane_norm = np.sqrt(a**2 + b**2 + c**2) + 1e-9
    
    # Calculate true perpendicular height map
    all_heights_mm = ((a * raw_pts[:, 0] + b * raw_pts[:, 1] + c * raw_pts[:, 2] + d) / plane_norm) * 1000.0
    
    # Fallback default baseline height
    adaptive_height_threshold_mm = 3.5

    # If RGB canvas matrix is present, execute Cross-Modal Analysis
    if rgb_img is not None:
        print("  [CROSS-MODAL] Analyzing 2D Visual Edges vs 3D Spatial Structures...", flush=True)
        
        # 1. Extract 2D Edge Profile
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        
        rgb_edge_thresh = np.quantile(grad_mag, 0.98)
        edge_mask_2d = grad_mag > rgb_edge_thresh
        
        # Project cloud points down into 2D to map visual-edge intersections
        u = np.round(raw_pts[:, 0] * info.fx / raw_pts[:, 2] + info.cx).astype(int) - info.crop_left
        v = np.round(raw_pts[:, 1] * info.fy / raw_pts[:, 2] + info.cy).astype(int) - info.crop_top
        valid_proj = (u >= 0) & (u < rgb_img.shape[1]) & (v >= 0) & (v < rgb_img.shape[0])
        
        pts_inside_edges = valid_proj.copy()
        pts_inside_edges[valid_proj] = edge_mask_2d[v[valid_proj], u[valid_proj]]
        
        # Sample heights inside visual contours
        visual_heights = all_heights_mm[pts_inside_edges & (all_heights_mm > -5.0) & (all_heights_mm < 50.0)]
        
        # 2. Extract 3D Local Structure Profiles (Linearity)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        sample_size = min(30000, len(raw_pts))
        sample_indices = np.random.choice(len(raw_pts), size=sample_size, replace=False)
        structural_heights = []
        
        for idx in sample_indices:
            [_, k_idx, _] = pcd_tree.search_radius_vector_3d(raw_pts[idx], 0.015)
            if len(k_idx) >= 6:
                neighbors = raw_pts[k_idx]
                cov = np.cov(neighbors, rowvar=False)
                eigenvalues, _ = np.linalg.eigh(cov)
                eigenvalues = eigenvalues[::-1]
                l1, l2, l3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
                
                # Check for tight thread-like structures (High Linearity)
                if l1 > 1e-8 and ((l1 - l2) / (l1 + 1e-9)) > 0.75:
                    structural_heights.append(all_heights_mm[idx])
        
        # 3. Reconcile thresholds
        rgb_floor = np.percentile(visual_heights, 5) if len(visual_heights) > 0 else 0.0
        pcd_floor = np.percentile(structural_heights, 5) if len(structural_heights) > 0 else 0.0
        
        print(f"  [CROSS-MODAL] 2D Visual Edge Base Floor: {rgb_floor:.2f}mm", flush=True)
        print(f"  [CROSS-MODAL] 3D Linearity Object Floor: {pcd_floor:.2f}mm", flush=True)
        
        if abs(rgb_floor - pcd_floor) <= 2.5 and min(rgb_floor, pcd_floor) > 1.5:
            adaptive_height_threshold_mm = min(rgb_floor, pcd_floor) - 0.5
            print(f"  [CROSS-MODAL] >> HARMONIZED THRESHOLD ESTABLISHED: {adaptive_height_threshold_mm:.2f}mm <<", flush=True)
        else:
            # Safe statistical default fallback if data streams diverge significantly
            adaptive_height_threshold_mm = 3.2
            print(f"  [CROSS-MODAL] Variance out of bounds. Applying safety baseline: {adaptive_height_threshold_mm:.2f}mm", flush=True)

    # Apply the calculated threshold gate
    in_floor_noise_zone = (all_heights_mm <= adaptive_height_threshold_mm)
    in_ceiling_structure_zone = (all_heights_mm >= MAX_HEIGHT_ABOVE_BELT_MM)
    above_belt_mask = (~in_floor_noise_zone) & (~in_ceiling_structure_zone)
    
    pts_stage2 = raw_pts[above_belt_mask]
    print(f"  [ELEVATION FILTER QUANTIZATION]")
    print(f"    ├── Points Demolished as Floor/Belt Noise (Height <= {adaptive_height_threshold_mm:.2f}mm): {np.sum(in_floor_noise_zone):,}", flush=True)
    print(f"    └── Points Surviving Elevation Gate (Within Workspace Window): {len(pts_stage2):,}", flush=True)

    if len(pts_stage2) == 0:
        return np.zeros((target_H, target_W), dtype=np.uint8), (a,b,c,d), 0, total_input_points

    print("\n================================================================================", flush=True)
    print("                STAGE 3: POINTWISE STATIC BOUNDARY DISMANTLING", flush=True)
    print("================================================================================", flush=True)
    outside_left = (pts_stage2[:, 0] <= RAIL_LEFT_X_LIMIT)
    outside_right = (pts_stage2[:, 0] >= RAIL_RIGHT_X_LIMIT)
    within_conveyor_belt = (~outside_left) & (~outside_right)
    
    pts_stage3 = pts_stage2[within_conveyor_belt]
    print(f"  [BOUNDARY FILTER QUANTIZATION]")
    print(f"    ├── Points Demolished as Left Rail Noise (X <= {RAIL_LEFT_X_LIMIT}m): {np.sum(outside_left):,}", flush=True)
    print(f"    └── Points Retained Inside Active Transport Area: {len(pts_stage3):,}", flush=True)

    if len(pts_stage3) == 0:
        return np.zeros((target_H, target_W), dtype=np.uint8), (a,b,c,d), 0, total_input_points

    print("\n================================================================================", flush=True)
    print("                STAGE 4: LOCALIZED NEIGHBORHOOD MASS VERIFICATION", flush=True)
    print("================================================================================", flush=True)
    pcd_workspace = o3d.geometry.PointCloud()
    pcd_workspace.points = o3d.utility.Vector3dVector(pts_stage3)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_workspace)
    
    valid_mass_indices = []
    sparse_noise_count = 0
    
    for i in range(len(pts_stage3)):
        [_, _, dists] = pcd_tree.search_radius_vector_3d(pts_stage3[i], LOCAL_RADIUS_METERS)
        if len(dists) >= MIN_NEIGHBORS_IN_RADIUS:
            valid_mass_indices.append(i)
        else:
            sparse_noise_count += 1
            
    pts_confirmed = pts_stage3[valid_mass_indices]
    print(f"  [MASS DENSITY QUANTIZATION]")
    print(f"    ├── Points Demolished as Sparse Phantom Reflections/Ghosts: {sparse_noise_count:,}", flush=True)
    print(f"    └── Points Confirmed Belonging to Stable Structural Masses  : {len(pts_confirmed):,}", flush=True)

    mask = np.zeros((target_H, target_W), dtype=np.uint8)
    if len(pts_confirmed) == 0:
        return mask, (a,b,c,d), 0, total_input_points

    print("\n================================================================================", flush=True)
    print("                STAGE 5: MATRIX LENS PROJECTION & ALIGNMENT MAP", flush=True)
    print("================================================================================", flush=True)
    u_raw = np.round(pts_confirmed[:, 0] * info.fx / pts_confirmed[:, 2] + info.cx).astype(int)
    v_raw = np.round(pts_confirmed[:, 1] * info.fy / pts_confirmed[:, 2] + info.cy).astype(int)

    u = u_raw - info.crop_left
    v = v_raw - info.crop_top

    in_bounds = (u >= 0) & (u < target_W) & (v >= 0) & (v < target_H)
    u_valid, v_valid = u[in_bounds], v[in_bounds]
    
    print(f"  [PROJECTION BOUNDS AUDIT]")
    print(f"    └── Mappings Successfully Committed directly to Array Layout : {len(u_valid):,}", flush=True)

    mask[v_valid, u_valid] = 255

    print("\n================================================================================", flush=True)
    print("                STAGE 6: MATRIC PATH MORPHOLOGY CLOSURE RECONSTRUCTION", flush=True)
    print("================================================================================", flush=True)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    print(f"  [MORPHOLOGICAL RECONSTRUCTION QUANTIZATION]")
    print(f"    └── Net total target output mask layout weight : {int(np.count_nonzero(mask)):,}", flush=True)

    return mask, (a,b,c,d), len(pts_confirmed), total_input_points


def run_masking(ply_path: Path, info: CaptureInfo, rgb_path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    # Load RGB image natively using the existing path context
    rgb_img = cv2.imread(str(rgb_path))
    mask, _, _, _ = run_masking_from_point_cloud(pcd, info, rgb_img)
    return mask


def run(name: str, samples: Path, out_dir: Path):
    ply_path = samples / "pointcloud" / f"{name}.ply"
    info_path = samples / "info" / f"{name}.txt"
    
    # Mirror your exact folder hierarchy to locate the partner asset safely
    rgb_path = samples / "image" / f"{name}.png"
    if not rgb_path.exists():
        rgb_path = samples / "rgb" / f"{name}.png"

    info = parse_info(info_path)
    mask = run_masking(ply_path, info, rgb_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_target_path = out_dir / f"{name}.png"
    cv2.imwrite(str(mask_target_path), mask)

    fg_px = int(np.count_nonzero(mask))

    print("\n" + "=" * 80, flush=True)
    print("                UNBUFFERED DIAGNOSTIC ENGINE VERBOSE REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"  Target Image Asset Name : {name}", flush=True)
    print(f"  Mask Active White Pixels: {fg_px:,}", flush=True)
    print(f"  Execution Pipeline Status: {'SUCCESS ✓' if fg_px > 0 else 'EMPTY BLACK OUTPUT ✗'}", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="img0018")
    parser.add_argument("--base", default="offline_case/samples")
    parser.add_argument("--out", default="offline_case/samples/mask")
    args = parser.parse_args()
    run(name=args.name, samples=Path(args.base), out_dir=Path(args.out))