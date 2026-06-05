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

    # G5 DISABLED - was killing small/thin objects (wires, rods).
    # G2 (base_mm <= 20mm) already rejects lighting independently.
    # Kept as informational only to monitor what it would have done.
    g5_pass = (dim_x >= MIN_TRUE_3D_WIDTH_MM) and (dim_y >= MIN_TRUE_3D_WIDTH_MM)
    g6_pass = volumetric_ratio >= MIN_VOLUMETRIC_RATIO

    # G5 intentionally excluded from all_pass
    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g6_pass

    msg = (
        f"Cluster #{cid:02d} -> Pts: {n:<5} | Dims(mm): X={dim_x:5.1f}, Y={dim_y:5.1f}, Z={dim_z:5.1f} | "
        f"VolRatio: {volumetric_ratio:.3f} | Gates: [G1-4={int(g1_pass & g2_pass & g3_pass & g4_pass)} G5_Size(INFO)={int(g5_pass)} G6_3D={int(g6_pass)}] -> {'PASSED' if all_pass else 'REJECTED'}"
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


def run_masking_from_point_cloud(pcd, info):
    target_H, target_W = info.rgb_shape

    print("\n━━━ [STAGE 1] Point Cloud Loaded")
    pts = np.asarray(pcd.points)
    print(f"  Total points: {len(pts):,}")

    # ─────────────────────────────────────────
    # STAGE 2 — CLEANING
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 2] Outlier Removal")
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    pts = np.asarray(pcd_clean.points)
    print(f"  After cleaning: {len(pts):,}")

    # ─────────────────────────────────────────
    # STAGE 3 — PLANE FIT (REFERENCE ONLY)
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 3] Plane Estimation (NO REMOVAL)")

    plane_model, inliers = pcd_clean.segment_plane(
        distance_threshold=0.003,
        ransac_n=3,
        num_iterations=2000
    )

    a, b, c, d = plane_model

    if c > 0:
        a, b, c, d = -a, -b, -c, -d

    plane_norm = np.sqrt(a*a + b*b + c*c) + 1e-9

    print(f"  Plane: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
    print(f"  Inliers used: {len(inliers):,}")

    # ─────────────────────────────────────────
    # STAGE 4 — HEIGHT FIELD
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 4] Height Field")

    heights_mm = (
        (a * pts[:,0] + b * pts[:,1] + c * pts[:,2] + d)
        / plane_norm
    ) * 1000.0

    print(f"  Height stats:")
    print(f"    Min: {heights_mm.min():.2f} mm")
    print(f"    Max: {heights_mm.max():.2f} mm")
    print(f"    Mean: {heights_mm.mean():.2f} mm")
    print(f"    Std: {heights_mm.std():.2f} mm")

    # ─────────────────────────────────────────
    # STAGE 5 — NOISE MODEL
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 5] Noise Floor")

    near_plane = heights_mm[np.abs(heights_mm) < 20]

    mean = np.mean(near_plane)
    std = np.std(near_plane)

    noise_floor = mean + 2.5 * std

    print(f"  Mean: {mean:.2f} mm")
    print(f"  Std : {std:.2f} mm")
    print(f"  → Floor: {noise_floor:.2f} mm")

    # ─────────────────────────────────────────
    # STAGE 6 — INITIAL HEIGHT FILTER
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 6] Height-Based Candidate Selection")

    height_mask = heights_mm > 1.0  # remove flat belt noise

    pts_h = pts[height_mask]
    heights_h = heights_mm[height_mask]

    print(f"  Points after height filter (>1mm): {len(pts_h):,}")

    # ─────────────────────────────────────────
    # STAGE 7 — LOCAL HEIGHT CONTRAST (KEY FIX)
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 7] Local Height Contrast Filter")

    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=10).fit(pts_h)
    _, indices = nbrs.kneighbors(pts_h)

    local_diff = np.abs(
        heights_h[:, None] - heights_h[indices]
    ).mean(axis=1)

    print(f"  Local contrast stats:")
    print(f"    Min: {local_diff.min():.2f}")
    print(f"    Max: {local_diff.max():.2f}")
    print(f"    Mean: {local_diff.mean():.2f}")

    contrast_threshold = 0.5  # tune later

    contrast_mask = local_diff > contrast_threshold

    pts_final = pts_h[contrast_mask]

    print(f"  Points after contrast filter: {len(pts_final):,}")

    if len(pts_final) == 0:
        print("  [WARNING] No points left after filtering")
        return np.zeros((target_H, target_W), dtype=np.uint8), None, 0, 0

    # ─────────────────────────────────────────
    # STAGE 8 — DBSCAN
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 8] DBSCAN")

    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(pts_final)

    labels = np.array(
        pcd_final.cluster_dbscan(
            eps=0.02,
            min_points=3,
            print_progress=False
        )
    )

    n_clusters = labels.max() + 1 if labels.size > 0 else 0
    print(f"  Clusters found: {n_clusters}")

    # ─────────────────────────────────────────
    # STAGE 9 — PROJECTION
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 9] Projection")

    mask = np.zeros((target_H, target_W), dtype=np.uint8)

    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        cluster_pts = pts_final[idx]

        if len(cluster_pts) < 5:
            continue

        u = np.round(cluster_pts[:,0] * info.fx / cluster_pts[:,2] + info.cx).astype(int)
        v = np.round(cluster_pts[:,1] * info.fy / cluster_pts[:,2] + info.cy).astype(int)

        u -= info.crop_left
        v -= info.crop_top

        valid = (u >= 0) & (u < target_W) & (v >= 0) & (v < target_H)

        mask[v[valid], u[valid]] = 255

    # ─────────────────────────────────────────
    # STAGE 10 — LIGHT MORPH
    # ─────────────────────────────────────────
    print("\n━━━ [STAGE 10] Morphology")

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    print(f"  Final mask pixels: {np.count_nonzero(mask):,}")

    return mask, plane_model, len(inliers), len(pts)


def run_masking(ply_path: Path, info: CaptureInfo) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    mask, _, _, _ = run_masking_from_point_cloud(pcd, info)
    return mask


def run(name: str, samples: Path, out_dir: Path):
    ply_path = samples / "pointcloud" / f"{name}.ply"
    info_path = samples / "info" / f"{name}.txt"

    info = parse_info(info_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    mask, _, _, _ = run_masking_from_point_cloud(pcd, info)

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_target_path = out_dir / f"{name}.png"
    cv2.imwrite(str(mask_target_path), mask)

    debug_dir = out_dir / "debug_clusters"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save raw pre-DBSCAN projection (unfragmented, no morphology)
    raw_pre = getattr(run_masking_from_point_cloud, "_raw_predbscan_mask", None)
    if raw_pre is not None:
        raw_path = debug_dir / f"{name}_raw_predbscan.png"
        cv2.imwrite(str(raw_path), raw_pre)
        print(f"  [DEBUG] Saved raw pre-DBSCAN mask: {raw_path}", flush=True)

    # Save one mask per passing cluster for visual debugging
    cluster_masks = getattr(run_masking_from_point_cloud, "_cluster_masks", [])
    if cluster_masks:
        for c_id, c_mask, c_score in cluster_masks:
            c_path = debug_dir / f"{name}_cluster{c_id:02d}_score{c_score:.3f}.png"
            cv2.imwrite(str(c_path), c_mask)
            print(f"  [DEBUG] Saved cluster mask: {c_path}", flush=True)
        print(f"  [DEBUG] {len(cluster_masks)} cluster mask(s) saved to {debug_dir}", flush=True)

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