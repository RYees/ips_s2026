"""
masking.py
----------
Production binary object masking pipeline utilizing 3D structural dimension
aspect ratios to separate volumetric masses from flat surface noise.
All mid-stage debug and diagnostic log sequences are preserved.

Usage:
    python3 offline_case/masking.py img0000 --base offline_case/samples
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d

# ─────────────────────────────────────────────────────────────────────────────
# Tunable Thresholds & Safety Limits
# ─────────────────────────────────────────────────────────────────────────────
RANSAC_DISTANCE_THRESHOLD = 0.012  # m
RANSAC_N = 3
RANSAC_ITERATIONS = 2000

DBSCAN_EPS = 0.015
DBSCAN_MIN_PTS = 8

GATE_MIN_POINTS = 20
GATE_GROUND_MM = 20.0
GATE_MIN_HEIGHT_MM = 0.8
GATE_MIN_HEIGHT_RATIO = 0.02
GATE_MIN_DENSITY = 10.0

MIN_OBJECT_HEIGHT_MM = 4.0
MAX_OBJECT_HEIGHT_MM = 500.0

# Permanent Geometric Constraints
MIN_TRUE_3D_WIDTH_MM = 15.0  # Objects must have a realistic minimum width
MIN_VOLUMETRIC_RATIO = 0.15  # Ratio between smallest and largest 3D dimensions

DIAG_LOGS = []


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
    plane_a: float = 0.0
    plane_b: float = 0.0
    plane_c: float = 0.0
    plane_d: float = 0.0


def parse_info(path: Path) -> CaptureInfo:
    info = CaptureInfo()
    if not path.exists():
        print(f"  [ERROR] Metadata file completely missing at: {path}", flush=True)
        return info
    lines = path.read_text().splitlines()

    def _floats(s):
        return [
            float(x) for x in re.findall(r"-?[\d]+\.[\d]+(?:[eE][+-]?\d+)?|-?\d+", s)
        ]

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
            if len(r0) >= 3:
                info.fx, info.cx = r0[0], r0[2]
            if len(r1) >= 3:
                info.fy, info.cy = r1[1], r1[2]
    return info


def evaluate_gates(cid, pts, plane_a, plane_b, plane_c, plane_d, plane_norm):
    n = len(pts)
    dist = (
        plane_a * pts[:, 0] + plane_b * pts[:, 1] + plane_c * pts[:, 2] + plane_d
    ) / plane_norm
    dist_mm = dist * 1000.0

    base_mm = float(dist_mm.min())
    height_mm = float(dist_mm.max() - dist_mm.min())

    # Calculate strict physical 3D bounding sizes in millimeters
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    bb_size_mm = (bb_max - bb_min) * 1000.0  # [X_size, Y_size, Z_size]

    dim_x, dim_y, dim_z = bb_size_mm[0], bb_size_mm[1], bb_size_mm[2]

    # Structural Flatness Test: minimum dimension vs largest dimension
    sorted_dims = sorted([dim_x, dim_y, dim_z])
    volumetric_ratio = sorted_dims[0] / (sorted_dims[2] + 1e-9)

    bb_vol = float(np.prod(bb_size_mm / 1000.0))
    footprint = float(bb_size_mm[0] * bb_size_mm[1]) + 1e-9
    height_ratio = height_mm / (np.sqrt(footprint) + 1e-9)
    density = n / bb_vol if bb_vol > 1e-8 else float("inf")

    # Core threshold gates
    g1_pass = n >= GATE_MIN_POINTS
    g2_pass = (base_mm >= -5.0) and (base_mm <= GATE_GROUND_MM)
    g3_pass = (height_mm >= GATE_MIN_HEIGHT_MM) and (
        height_ratio >= GATE_MIN_HEIGHT_RATIO
    )
    g4_pass = density >= GATE_MIN_DENSITY if density != float("inf") else False

    # Permanent 3D Shape Gates (Identifies thickness profiles)
    g5_pass = (dim_x >= MIN_TRUE_3D_WIDTH_MM) and (dim_y >= MIN_TRUE_3D_WIDTH_MM)
    g6_pass = volumetric_ratio >= MIN_VOLUMETRIC_RATIO

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass and g6_pass

    msg = (
        f"Cluster #{cid:02d} -> Pts: {n:<5} | Dims(mm): X={dim_x:5.1f}, Y={dim_y:5.1f}, Z={dim_z:5.1f} | "
        f"VolRatio: {volumetric_ratio:.3f} | Gates: [G1-4={int(g1_pass & g2_pass & g3_pass & g4_pass)} G5_Size={int(g5_pass)} G6_3D={int(g6_pass)}] -> {'PASSED' if all_pass else 'REJECTED'}"
    )
    DIAG_LOGS.append(msg)
    print("  " + msg, flush=True)

    if all_pass:
        score = (
            0.5 * min(n / 10000.0, 1.0)
            + 0.3 * volumetric_ratio
            + 0.2 * min(density / 30000.0, 1.0)
        )
    else:
        score = 0.0

    return all_pass, score


def run_masking_from_point_cloud(
    pcd: o3d.geometry.PointCloud, info: CaptureInfo
) -> tuple[np.ndarray, tuple[float, float, float, float] | None, int, int]:
    target_H, target_W = info.rgb_shape[0], info.rgb_shape[1]

    print("\n━━━ [STAGE 1] Loading Point Cloud Array", flush=True)
    print(f"  [INFO] Total raw points loaded: {len(pcd.points):,}", flush=True)

    print("\n━━━ [STAGE 2] Statistical Outlier Demolition", flush=True)
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    print(
        f"  [INFO] Points surviving outlier filter: {len(pcd_clean.points):,}",
        flush=True,
    )

    print(f"\n━━━ [STAGE 3] Live Plane Surface Isolation (RANSAC)", flush=True)
    ransac_model, inlier_idx = pcd_clean.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS,
    )
    [a, b, c, d] = ransac_model
    print(
        f"  [INFO] Original RANSAC: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}",
        flush=True,
    )

    if c > 0:
        print(
            "  [INFO] Normal vector pointing down. Flipping coefficients up toward lens.",
            flush=True,
        )
        a, b, c, d = -a, -b, -c, -d

    plane_norm = np.sqrt(a**2 + b**2 + c**2) + 1e-9
    outlier_cloud = pcd_clean.select_by_index(inlier_idx, invert=True)
    pts_out = np.asarray(outlier_cloud.points)
    print(f"  [INFO] Points isolated on belt plane: {len(inlier_idx):,}", flush=True)
    print(
        f"  [INFO] Non-plane points going to workspace filter: {len(pts_out):,}",
        flush=True,
    )

    print(f"\n━━━ [STAGE 5] Direction-Enforced Spatial Pre-Filtering", flush=True)
    dist_out_mm = (
        (a * pts_out[:, 0] + b * pts_out[:, 1] + c * pts_out[:, 2] + d) / plane_norm
    ) * 1000.0

    valid_physical_window = (dist_out_mm > MIN_OBJECT_HEIGHT_MM) & (
        dist_out_mm < MAX_OBJECT_HEIGHT_MM
    )
    pts_filtered = pts_out[valid_physical_window]
    print(
        f"  [DEBUG] Points passing directional spatial filters: {len(pts_filtered):,}",
        flush=True,
    )

    print(
        f"\n━━━ [STAGE 6] Spatial Structural Isolation Clustering (DBSCAN)", flush=True
    )
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_filtered)
    labels = np.array(
        pcd_out.cluster_dbscan(
            eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS, print_progress=False
        )
    )
    n_clusters = int(labels.max()) + 1 if labels.size > 0 and labels.max() >= 0 else 0
    print(f"  [INFO] DBSCAN discovered {n_clusters} distinct point groups.", flush=True)

    print(f"\n━━━ [STAGE 7] Multi-Gate Geometric Evaluation", flush=True)
    candidates = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        c_pts = pts_filtered[idx]
        passed, score = evaluate_gates(cid, c_pts, a, b, c, d, plane_norm)
        if passed:
            candidates.append((c_pts, score, cid))

    if not candidates:
        print(
            "  [WARNING] Zero clusters passed true 3D volumetric threshold gates.",
            flush=True,
        )
        return (
            np.zeros((target_H, target_W), dtype=np.uint8),
            tuple(ransac_model),
            int(len(inlier_idx)),
            int(len(outlier_cloud.points)),
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_pts, best_score, best_id = candidates[0]
    print(
        f"  [INFO] Winning Cluster selected: ID #{best_id} (Score={best_score:.4f})",
        flush=True,
    )

    # ─────────────────────────────────────────────────────────────
    # FIXED STAGE 8: WINNING CLUSTER POINT EXTRACTION & PROJECTION
    # ─────────────────────────────────────────────────────────────
    print(f"\n━━━ [STAGE 8] Dynamic Mapping & Lens Axis Projection", flush=True)

    # CRITICAL FIX: Ensure we only project the points from the selected object cluster
    if "best_cluster_idx" in locals() and best_cluster_idx is not None:
        # Filter points belonging only to the winning cluster group
        cluster_mask = labels == best_cluster_idx
        project_pts = xyz_workspace[cluster_mask]
        print(
            f"  [DEBUG] Projecting Winning Cluster #{best_cluster_idx} ({len(project_pts)} points)"
        )
    else:
        # Fallback if clustering was bypassed
        project_pts = best_pts

    # Calculate scale maps from depth space back to RGB canvas space
    scale_x = (
        float(info.rgb_shape[1]) / float(info.raw_depth_shape[1])
        if info.raw_depth_shape[1] > 0
        else 1.0
    )
    scale_y = (
        float(info.rgb_shape[0]) / float(info.raw_depth_shape[0])
        if info.raw_depth_shape[0] > 0
        else 1.0
    )

    # Project the cluster points directly
    u_depth = (project_pts[:, 0] * (info.fx * scale_x) / project_pts[:, 2]) + (
        (info.cx * scale_x) - (info.crop_left * scale_x)
    )
    v_depth = (project_pts[:, 1] * (info.fy * scale_y) / project_pts[:, 2]) + (
        (info.cy * scale_y) - (info.crop_top * scale_y)
    )

    # Scale indices directly into the active cropped RGB canvas window
    u = np.round(u_depth / scale_x).astype(int)
    v = np.round(v_depth / scale_y).astype(int)

    print(f"  [DIAGNOSTIC] New Projective Coordinate Bounds:")
    print(f"    u min/max : {u.min()} to {u.max()}")
    print(f"    v min/max : {v.min()} to {v.max()}")

    # Safely handle canvas edge boundaries dynamically
    in_bounds = (u >= 0) & (u < target_W) & (v >= 0) & (v < target_H)
    u_valid, v_valid = u[in_bounds], v[in_bounds]

    print(f"  [INFO] Total candidate points projected: {len(u):,}", flush=True)
    print(
        f"  [INFO] Target Canvas Frame Boundary Configured To: Height={target_H} px, Width={target_W} px"
    )
    print(
        f"  [INFO] Projected pixels inside image frame boundary limits: {len(u_valid):,}",
        flush=True,
    )

    mask = np.zeros((target_H, target_W), dtype=np.uint8)
    if len(u_valid) == 0:
        print(
            "  [WARNING] Array allocation bypassed: Zero projected indices matched bounds.",
            flush=True,
        )
        return (
            mask,
            tuple(ransac_model),
            int(len(inlier_idx)),
            int(len(outlier_cloud.points)),
        )

    mask[v_valid, u_valid] = 255

    # =========════════════════════════════════════════════════════════════════════

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    white_y, white_x = np.where(mask > 0)
    if len(white_x) > 0:
        box_w = white_x.max() - white_x.min()
        box_h = white_y.max() - white_y.min()
        print(
            f"  [INFO] Generated Mask Bounding Shape: Width={box_w} px, Height={box_h} px",
            flush=True,
        )

    return (
        mask,
        tuple(ransac_model),
        int(len(inlier_idx)),
        int(len(outlier_cloud.points)),
    )


def run_masking(ply_path: Path, info: CaptureInfo) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    mask, _, _, _ = run_masking_from_point_cloud(pcd, info)
    return mask


def run(name: str, samples: Path, out_dir: Path):
    ply_path = samples / "pointcloud" / f"{name}.ply"
    info_path = samples / "info" / f"{name}.txt"

    info = parse_info(info_path)
    mask = run_masking(ply_path, info)

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_target_path = out_dir / f"{name}.png"
    cv2.imwrite(str(mask_target_path), mask)

    fg_px = int(np.count_nonzero(mask))

    print("\n" + "=" * 80, flush=True)
    print("                UNBUFFERED DIAGNOSTIC ENGINE VERBOSE REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f"  Target Image Asset Name : {name}", flush=True)
    print(
        f"  Info file resolution parsing : H={info.rgb_shape[0]} px, W={info.rgb_shape[1]} px",
        flush=True,
    )
    print(
        f"  Crop Offset Regimes     : Top={info.crop_top} px, Left={info.crop_left} px",
        flush=True,
    )
    print(f"  Saved Image Output Path : {mask_target_path}", flush=True)
    print(f"  Mask Active White Pixels: {fg_px:,}", flush=True)
    print(
        f"  Execution Pipeline Status: {'SUCCESS ✓' if fg_px > 0 else 'EMPTY BLACK OUTPUT ✗'}",
        flush=True,
    )
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="img0000")
    parser.add_argument("--base", default="offline_case/samples")
    parser.add_argument("--out", default="offline_case/samples/mask")
    args = parser.parse_args()
    run(name=args.name, samples=Path(args.base), out_dir=Path(args.out))
