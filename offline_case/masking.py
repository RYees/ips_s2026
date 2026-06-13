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
import datetime
import re
import sys
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

# Keep thin appendages from getting clipped before clustering.
MIN_OBJECT_HEIGHT_MM = 1.5
MAX_OBJECT_HEIGHT_MM = 500.0

# Debug toggle: temporarily skip the final opening step to see whether it is
# shaving off thin tips / appendages from the projected mask.
DEBUG_SKIP_MORPH_OPEN = True

# Permanent Geometric Constraints
MIN_TRUE_3D_WIDTH_MM = 15.0  # Objects must have a realistic minimum width
MIN_VOLUMETRIC_RATIO = 0.15  # Ratio between smallest and largest 3D dimensions

DIAG_LOGS = []
LAST_PIPELINE_DIAGNOSTICS = {}
LAST_PIPELINE_MASKS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Global log file handle — set once in run() / run_masking_from_point_cloud()
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FILE = None  # open file handle, or None


def _set_log_file(path: Path):
    """Open (append) the log file and store the handle globally."""
    global _LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = open(path, "a", encoding="utf-8")
    _LOG_FILE.write(
        f"\n{'=' * 80}\n"
        f"  masking.py run started: {datetime.datetime.now()}\n"
        f"{'=' * 80}\n"
    )
    _LOG_FILE.flush()


def _close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None:
        _LOG_FILE.write(f"  masking.py run ended: {datetime.datetime.now()}\n")
        _LOG_FILE.flush()
        _LOG_FILE.close()
        _LOG_FILE = None


def log(msg: str, flush: bool = True):
    """Print to stdout AND write to the log file (if open)."""
    print(msg, flush=flush)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        if flush:
            _LOG_FILE.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
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
        log(f"  [ERROR] Metadata file completely missing at: {path}")
        return info
    lines = path.read_text().splitlines()

    def _floats(s):
        return [
            float(x) for x in re.findall(r"-?[\d]+\.[\d]+(?:[eE][+-]?\d+)?|-?\d+", s)
        ]

    def _ints(s):
        return [int(x) for x in re.findall(r"-\d+|\d+", s)]

    crop = None
    pointcloud_crop = None
    pointcloud_rgb_shape = None
    pointcloud_depth_shape = None
    cropped_rgb_shape = None
    raw_depth_shape = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Point cloud crop:"):
            pointcloud_crop = _ints(s)
            continue
        if s.startswith("Point cloud RGB shape:"):
            v = _ints(s)
            if len(v) >= 2:
                pointcloud_rgb_shape = (v[0], v[1])
            continue
        if s.startswith("Point cloud depth shape:"):
            v = _ints(s)
            if len(v) >= 2:
                pointcloud_depth_shape = (v[0], v[1])
            continue
        if s.startswith("Crop:"):
            continue
        if s.startswith("top=") and "left=" in s and "right=" in s:
            crop = _ints(s)
            continue
        if s.startswith("Cropped RGB shape:") or s.startswith("RGB shape:"):
            v = _ints(s)
            if len(v) >= 2:
                cropped_rgb_shape = (v[0], v[1])
            continue
        if s.startswith("Raw depth shape:"):
            v = _ints(s)
            if len(v) >= 2:
                raw_depth_shape = (v[0], v[1])
            continue
        if s.startswith("Intrinsics matrix:"):
            r0 = _floats(lines[i + 1]) if i + 1 < len(lines) else []
            r1 = _floats(lines[i + 2]) if i + 2 < len(lines) else []
            if len(r0) >= 3:
                info.fx, info.cx = r0[0], r0[2]
            if len(r1) >= 3:
                info.fy, info.cy = r1[1], r1[2]

    if pointcloud_crop and len(pointcloud_crop) >= 4:
        info.crop_top, info.crop_left = pointcloud_crop[0], pointcloud_crop[2]
    elif crop and len(crop) >= 4:
        info.crop_top, info.crop_left = crop[0], crop[2]

    if pointcloud_rgb_shape is not None:
        info.rgb_shape = pointcloud_rgb_shape
    elif cropped_rgb_shape is not None:
        info.rgb_shape = cropped_rgb_shape

    if pointcloud_depth_shape is not None:
        info.raw_depth_shape = pointcloud_depth_shape
    elif raw_depth_shape is not None:
        info.raw_depth_shape = raw_depth_shape

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_gates(cid, pts, plane_a, plane_b, plane_c, plane_d, plane_norm):
    n = len(pts)
    dist = (
        plane_a * pts[:, 0] + plane_b * pts[:, 1] + plane_c * pts[:, 2] + plane_d
    ) / plane_norm
    dist_mm = dist * 1000.0

    base_mm = float(dist_mm.min())
    height_mm = float(dist_mm.max() - dist_mm.min())

    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    bb_size_mm = (bb_max - bb_min) * 1000.0

    dim_x, dim_y, dim_z = bb_size_mm[0], bb_size_mm[1], bb_size_mm[2]

    sorted_dims = sorted([dim_x, dim_y, dim_z])
    volumetric_ratio = sorted_dims[0] / (sorted_dims[2] + 1e-9)

    bb_vol = float(np.prod(bb_size_mm / 1000.0))
    footprint = float(bb_size_mm[0] * bb_size_mm[1]) + 1e-9
    height_ratio = height_mm / (np.sqrt(footprint) + 1e-9)
    density = n / bb_vol if bb_vol > 1e-8 else float("inf")

    g1_pass = n >= GATE_MIN_POINTS
    g2_pass = (base_mm >= -5.0) and (base_mm <= GATE_GROUND_MM)
    g3_pass = (height_mm >= GATE_MIN_HEIGHT_MM) and (
        height_ratio >= GATE_MIN_HEIGHT_RATIO
    )
    g4_pass = density >= GATE_MIN_DENSITY if density != float("inf") else False
    g5_pass = (dim_x >= MIN_TRUE_3D_WIDTH_MM) and (dim_y >= MIN_TRUE_3D_WIDTH_MM)
    g6_pass = volumetric_ratio >= MIN_VOLUMETRIC_RATIO

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass and g6_pass

    msg = (
        f"Cluster #{cid:02d} -> Pts: {n:<5} | Dims(mm): X={dim_x:5.1f}, Y={dim_y:5.1f}, Z={dim_z:5.1f} | "
        f"VolRatio: {volumetric_ratio:.3f} | Gates: [G1-4={int(g1_pass & g2_pass & g3_pass & g4_pass)} "
        f"G5_Size={int(g5_pass)} G6_3D={int(g6_pass)}] -> {'PASSED' if all_pass else 'REJECTED'}"
    )
    DIAG_LOGS.append(msg)
    log("  " + msg)

    if all_pass:
        score = (
            0.5 * min(n / 10000.0, 1.0)
            + 0.3 * volumetric_ratio
            + 0.2 * min(density / 30000.0, 1.0)
        )
    else:
        score = 0.0

    return all_pass, score


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_masking_from_point_cloud(
    pcd: o3d.geometry.PointCloud, info: CaptureInfo
) -> tuple[np.ndarray, tuple[float, float, float, float] | None, int, int]:
    global LAST_PIPELINE_DIAGNOSTICS, LAST_PIPELINE_MASKS
    LAST_PIPELINE_DIAGNOSTICS = {}
    LAST_PIPELINE_MASKS = {}

    target_H, target_W = info.rgb_shape[0], info.rgb_shape[1]

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 1] Loading Point Cloud Array")
    log(f"  [INFO] Total raw points loaded: {len(pcd.points):,}")

    # ── STAGE 2 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 2] Statistical Outlier Demolition")
    pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    log(f"  [INFO] Points surviving outlier filter: {len(pcd_clean.points):,}")

    # ── STAGE 3 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 3] Live Plane Surface Isolation (RANSAC)")
    ransac_model, inlier_idx = pcd_clean.segment_plane(
        distance_threshold=RANSAC_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS,
    )
    [a, b, c, d] = ransac_model
    log(f"  [INFO] Original RANSAC: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")

    if c > 0:
        log(
            "  [INFO] Normal vector pointing down. Flipping coefficients up toward lens."
        )
        a, b, c, d = -a, -b, -c, -d

    plane_norm = np.sqrt(a**2 + b**2 + c**2) + 1e-9
    outlier_cloud = pcd_clean.select_by_index(inlier_idx, invert=True)
    pts_out = np.asarray(outlier_cloud.points)
    log(f"  [INFO] Points isolated on belt plane: {len(inlier_idx):,}")
    log(f"  [INFO] Non-plane points going to workspace filter: {len(pts_out):,}")

    # ── STAGE 5 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 5] Direction-Enforced Spatial Pre-Filtering")
    dist_out_mm = (
        (a * pts_out[:, 0] + b * pts_out[:, 1] + c * pts_out[:, 2] + d) / plane_norm
    ) * 1000.0
    if len(dist_out_mm) > 0:
        below_count = int((dist_out_mm <= MIN_OBJECT_HEIGHT_MM).sum())
        above_count = int((dist_out_mm >= MAX_OBJECT_HEIGHT_MM).sum())
        log(
            f"  [INFO] Height gate window : {MIN_OBJECT_HEIGHT_MM:.2f}mm "
            f"to {MAX_OBJECT_HEIGHT_MM:.2f}mm"
        )
        log(
            f"  [INFO] Dist-from-plane mm : min={dist_out_mm.min():.3f} "
            f"mean={dist_out_mm.mean():.3f} median={np.median(dist_out_mm):.3f} "
            f"max={dist_out_mm.max():.3f}"
        )
        log(
            f"  [INFO] Rejected by lower gate: {below_count:,} / {len(dist_out_mm):,}"
        )
        log(
            f"  [INFO] Rejected by upper gate: {above_count:,} / {len(dist_out_mm):,}"
        )

    valid_physical_window = (dist_out_mm > MIN_OBJECT_HEIGHT_MM) & (
        dist_out_mm < MAX_OBJECT_HEIGHT_MM
    )
    pts_filtered = pts_out[valid_physical_window]
    log(f"  [DEBUG] Points passing directional spatial filters: {len(pts_filtered):,}")

    LAST_PIPELINE_DIAGNOSTICS.update(
        {
            "height_gate_min_mm": float(MIN_OBJECT_HEIGHT_MM),
            "height_gate_max_mm": float(MAX_OBJECT_HEIGHT_MM),
            "dist_min_mm": float(dist_out_mm.min()) if len(dist_out_mm) > 0 else 0.0,
            "dist_mean_mm": float(dist_out_mm.mean()) if len(dist_out_mm) > 0 else 0.0,
            "dist_median_mm": float(np.median(dist_out_mm)) if len(dist_out_mm) > 0 else 0.0,
            "dist_max_mm": float(dist_out_mm.max()) if len(dist_out_mm) > 0 else 0.0,
            "stage5_input_points": int(len(pts_out)),
            "stage5_pass_points": int(len(pts_filtered)),
            "stage5_rejected_lower": int((dist_out_mm <= MIN_OBJECT_HEIGHT_MM).sum()),
            "stage5_rejected_upper": int((dist_out_mm >= MAX_OBJECT_HEIGHT_MM).sum()),
        }
    )

    # ── STAGE 6 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 6] Spatial Structural Isolation Clustering (DBSCAN)")
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_filtered)
    labels = np.array(
        pcd_out.cluster_dbscan(
            eps=DBSCAN_EPS, min_points=DBSCAN_MIN_PTS, print_progress=False
        )
    )
    n_clusters = int(labels.max()) + 1 if labels.size > 0 and labels.max() >= 0 else 0
    log(f"  [INFO] DBSCAN discovered {n_clusters} distinct point groups.")

    # ── STAGE 7 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 7] Multi-Gate Geometric Evaluation")
    candidates = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        c_pts = pts_filtered[idx]
        passed, score = evaluate_gates(cid, c_pts, a, b, c, d, plane_norm)
        if passed:
            candidates.append((c_pts, score, cid))

    if not candidates:
        log("  [WARNING] Zero clusters passed true 3D volumetric threshold gates.")
        return (
            np.zeros((target_H, target_W), dtype=np.uint8),
            tuple(ransac_model),
            int(len(inlier_idx)),
            int(len(outlier_cloud.points)),
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_pts, best_score, best_id = candidates[0]
    log(f"  [INFO] Winning Cluster selected: ID #{best_id} (Score={best_score:.4f})")
    log(f"  [INFO] Winning cluster point count: {len(best_pts):,}")

    # ── STAGE 8 ───────────────────────────────────────────────────────────────
    log("\n━━━ [STAGE 8] Projection — Sensor Space → Cropped Canvas")

    # Use the winning cluster points directly (best_pts already carries only
    # the selected cluster; the dead best_cluster_idx/xyz_workspace references
    # have been removed).
    project_pts = best_pts

    # ── 8a  Scales: depth sensor pixels → RGB canvas pixels ──────────────────
    # The point cloud was built from the depth sensor (raw_depth_shape).
    # The target canvas is the cropped RGB image (rgb_shape).
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

    log("\n  ── [STAGE 8a] Coordinate System Parameters ──")
    log(
        f"    Intrinsics    : fx={info.fx:.3f}  fy={info.fy:.3f}  "
        f"cx={info.cx:.3f}  cy={info.cy:.3f}"
    )
    log(f"    Crop offsets  : left={info.crop_left} px  top={info.crop_top} px")
    log(
        f"    Depth sensor  : {info.raw_depth_shape[1]}W x {info.raw_depth_shape[0]}H px"
    )
    log(f"    RGB canvas    : {info.rgb_shape[1]}W x {info.rgb_shape[0]}H px")
    log(f"    Scale factors : scale_x={scale_x:.6f}  scale_y={scale_y:.6f}")

    # Sanity-check: warn if scale factors look wrong (should be ≈ 0.4–2.0 range)
    if not (0.1 < scale_x < 10.0) or not (0.1 < scale_y < 10.0):
        log(
            f"  [WARNING] Unusual scale factors — check raw_depth_shape and rgb_shape "
            f"are populated correctly (depth={info.raw_depth_shape}, rgb={info.rgb_shape})"
        )

    # ── 8b  3-D point stats before projection ────────────────────────────────
    log("\n  ── [STAGE 8b] 3-D Cluster Bounding Box (world space) ──")
    pt_min = project_pts.min(axis=0)
    pt_max = project_pts.max(axis=0)
    log(
        f"    X range : {pt_min[0]:.4f} m  →  {pt_max[0]:.4f} m  "
        f"(span {(pt_max[0] - pt_min[0]) * 1000:.1f} mm)"
    )
    log(
        f"    Y range : {pt_min[1]:.4f} m  →  {pt_max[1]:.4f} m  "
        f"(span {(pt_max[1] - pt_min[1]) * 1000:.1f} mm)"
    )
    log(
        f"    Z range : {pt_min[2]:.4f} m  →  {pt_max[2]:.4f} m  "
        f"(span {(pt_max[2] - pt_min[2]) * 1000:.1f} mm)"
    )

    z_vals = project_pts[:, 2]
    log(
        f"    Z depth : mean={z_vals.mean():.4f} m  "
        f"std={z_vals.std():.4f} m  "
        f"min={z_vals.min():.4f} m  max={z_vals.max():.4f} m"
    )

    if z_vals.min() <= 0:
        log(
            f"  [ERROR] Some Z values are <= 0 ({(z_vals <= 0).sum()} points). "
            f"These will produce invalid projections and are dropped."
        )
        valid_z = z_vals > 0
        project_pts = project_pts[valid_z]
        log(f"  [INFO] Points remaining after Z<=0 filter: {len(project_pts):,}")
        if len(project_pts) == 0:
            log("  [ERROR] No valid points left after Z filter. Returning empty mask.")
            return (
                np.zeros((target_H, target_W), dtype=np.uint8),
                tuple(ransac_model),
                int(len(inlier_idx)),
                int(len(outlier_cloud.points)),
            )

    # ── 8c  Step-by-step projection with intermediate diagnostics ─────────────
    #
    #   Correct pipeline:
    #     1. Project into full-sensor pixel space using raw intrinsics
    #     2. Subtract the crop offset  → cropped-frame pixel coords
    #     3. Scale cropped-frame pixels → RGB canvas pixels
    #
    #   The previous code scaled fx/fy and cx/cy by scale_x/y before
    #   the crop subtraction, then divided by scale_x/y again, which
    #   computes the correct u only if scale==1.  At scale≠1 the crop
    #   term is scaled once but the focal-length term is scaled twice,
    #   producing a systematic horizontal / vertical shift.
    #
    log("\n  ── [STAGE 8c] Projection Steps ──")

    # Step 1 — project into full sensor pixel space (no scale yet)
    u_sensor = (project_pts[:, 0] * info.fx / project_pts[:, 2]) + info.cx
    v_sensor = (project_pts[:, 1] * info.fy / project_pts[:, 2]) + info.cy
    log(f"    [Step 1] Full-sensor pixel coords (before crop subtraction):")
    log(
        f"      u_sensor  min={u_sensor.min():.1f}  max={u_sensor.max():.1f}  "
        f"mean={u_sensor.mean():.1f}"
    )
    log(
        f"      v_sensor  min={v_sensor.min():.1f}  max={v_sensor.max():.1f}  "
        f"mean={v_sensor.mean():.1f}"
    )

    # Sanity-check: sensor coords should sit inside the full sensor frame
    sensor_W = (
        info.raw_depth_shape[1] if info.raw_depth_shape[1] > 0 else info.rgb_shape[1]
    )
    sensor_H = (
        info.raw_depth_shape[0] if info.raw_depth_shape[0] > 0 else info.rgb_shape[0]
    )
    pct_u_oob = float(((u_sensor < 0) | (u_sensor >= sensor_W)).mean()) * 100
    pct_v_oob = float(((v_sensor < 0) | (v_sensor >= sensor_H)).mean()) * 100
    if pct_u_oob > 5 or pct_v_oob > 5:
        log(
            f"  [WARNING] Many sensor-space coords out of sensor bounds "
            f"(u OOB={pct_u_oob:.1f}%, v OOB={pct_v_oob:.1f}%). "
            f"Check that the point cloud was generated with the same intrinsics."
        )
    else:
        log(f"    [OK] Sensor-space OOB: u={pct_u_oob:.1f}%  v={pct_v_oob:.1f}%")

    # Step 2 — subtract crop offset → cropped frame pixel coords
    u_cropped = u_sensor - info.crop_left
    v_cropped = v_sensor - info.crop_top
    log(
        f"    [Step 2] After crop subtraction (crop_left={info.crop_left}, crop_top={info.crop_top}):"
    )
    log(
        f"      u_cropped min={u_cropped.min():.1f}  max={u_cropped.max():.1f}  "
        f"mean={u_cropped.mean():.1f}"
    )
    log(
        f"      v_cropped min={v_cropped.min():.1f}  max={v_cropped.max():.1f}  "
        f"mean={v_cropped.mean():.1f}"
    )

    # Sanity-check: cropped coords should mostly be inside [0, canvas_size)
    pct_u_neg = float((u_cropped < 0).mean()) * 100
    pct_u_wide = float((u_cropped >= target_W).mean()) * 100
    pct_v_neg = float((v_cropped < 0).mean()) * 100
    pct_v_tall = float((v_cropped >= target_H).mean()) * 100
    log(
        f"    [Step 2 Check] Cropped coords outside canvas: "
        f"u<0={pct_u_neg:.1f}%  u>W={pct_u_wide:.1f}%  "
        f"v<0={pct_v_neg:.1f}%  v>H={pct_v_tall:.1f}%"
    )
    if pct_u_neg > 50 or pct_u_wide > 50:
        log(
            f"  [WARNING] More than 50% of u-coords are outside the canvas width. "
            f"crop_left may be wrong or the point cloud origin does not match the "
            f"cropped image. Expected u_cropped centre near {target_W / 2:.0f} px, "
            f"got {u_cropped.mean():.1f} px."
        )
    if pct_v_neg > 50 or pct_v_tall > 50:
        log(f"  [WARNING] More than 50% of v-coords are outside the canvas height.")

    # Step 3 — scale to RGB canvas (depth sensor res → canvas res)
    u = np.round(u_cropped * scale_x).astype(int)
    v = np.round(v_cropped * scale_y).astype(int)
    log(
        f"    [Step 3] After scale to canvas (scale_x={scale_x:.4f}, scale_y={scale_y:.4f}):"
    )
    log(f"      u  min={u.min()}  max={u.max()}  mean={u.mean():.1f}")
    log(f"      v  min={v.min()}  max={v.max()}  mean={v.mean():.1f}")
    log(f"      Canvas target size : {target_W}W x {target_H}H px")

    # Expected centre of object on canvas (useful for overlay verification)
    log(f"\n  ── [STAGE 8d] Expected mask centroid on canvas ──")
    log(f"      Projected centroid : u={u.mean():.1f} px  v={v.mean():.1f} px")
    log(f"      Canvas centre      : u={target_W / 2:.1f} px  v={target_H / 2:.1f} px")
    log(
        f"      Offset from centre : Δu={u.mean() - target_W / 2:.1f} px  "
        f"Δv={v.mean() - target_H / 2:.1f} px"
    )

    # ── 8e  Bounds filter & mask paint ────────────────────────────────────────
    in_bounds = (u >= 0) & (u < target_W) & (v >= 0) & (v < target_H)
    u_valid, v_valid = u[in_bounds], v[in_bounds]
    projected_pts = int(len(u))
    inside_bounds_pts = int(len(u_valid))
    duplicate_pixel_count = 0
    if inside_bounds_pts > 0:
        uv_pairs = np.stack([u_valid, v_valid], axis=1)
        duplicate_pixel_count = int(
            inside_bounds_pts - len(np.unique(uv_pairs, axis=0))
        )

    pct_valid = len(u_valid) / max(len(u), 1) * 100
    log(f"\n  ── [STAGE 8e] Bounds Filter ──")
    log(f"    Total projected points : {projected_pts:,}")
    log(f"    Inside canvas bounds   : {inside_bounds_pts:,}  ({pct_valid:.1f}%)")
    log(f"    Duplicate pixel hits    : {duplicate_pixel_count:,}")
    if pct_valid < 10:
        log(
            f"  [WARNING] Less than 10% of points are inside the canvas. "
            f"The mask will likely be empty or tiny. Check crop offsets and "
            f"that rgb_shape / raw_depth_shape are correct."
        )

    mask = np.zeros((target_H, target_W), dtype=np.uint8)
    if len(u_valid) == 0:
        log(
            "  [ERROR] Zero projected points inside canvas bounds — returning empty mask."
        )
        return (
            mask,
            tuple(ransac_model),
            int(len(inlier_idx)),
            int(len(outlier_cloud.points)),
        )

    mask[v_valid, u_valid] = 255
    raw_mask_white = int((mask > 0).sum())
    log(f"  [INFO] Raw mask white pixels before morphology: {raw_mask_white:,}")
    raw_mask = mask.copy()

    # ── Morphological clean-up ────────────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    post_close_white = int((mask > 0).sum())
    log(f"  [INFO] White pixels after close: {post_close_white:,}")
    post_close_mask = mask.copy()
    if DEBUG_SKIP_MORPH_OPEN:
        post_open_white = post_close_white
        log("  [INFO] Morphological open skipped for tip-loss debugging.")
        log(f"  [INFO] White pixels after open (skipped): {post_open_white:,}")
    else:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        post_open_white = int((mask > 0).sum())
        log(f"  [INFO] White pixels after open: {post_open_white:,}")

    white_y, white_x = np.where(mask > 0)
    if len(white_x) > 0:
        box_w = white_x.max() - white_x.min()
        box_h = white_y.max() - white_y.min()
        cx_mask = (white_x.min() + white_x.max()) / 2
        cy_mask = (white_y.min() + white_y.max()) / 2
        log(f"\n  ── [STAGE 8f] Final Mask Stats ──")
        log(f"    Bounding box : {box_w}W x {box_h}H px")
        log(f"    Mask centroid: ({cx_mask:.1f}, {cy_mask:.1f}) px")
        log(f"    White pixels : {int((mask > 0).sum()):,}")
    else:
        log("  [WARNING] Mask is completely empty after morphological operations.")

    LAST_PIPELINE_DIAGNOSTICS.update(
        {
            "projected_points": int(projected_pts),
            "inside_bounds_points": int(inside_bounds_pts),
            "duplicate_pixel_hits": int(duplicate_pixel_count),
            "raw_mask_white": int(raw_mask_white),
            "post_close_white": int(post_close_white),
            "post_open_white": int(post_open_white),
        }
    )
    LAST_PIPELINE_MASKS.update(
        {
            "raw_mask": raw_mask,
            "post_close_mask": post_close_mask,
            "final_mask": mask,
        }
    )

    return (
        mask,
        tuple(ransac_model),
        int(len(inlier_idx)),
        int(len(outlier_cloud.points)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_masking(ply_path: Path, info: CaptureInfo) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    mask, _, _, _ = run_masking_from_point_cloud(pcd, info)
    return mask


def run(name: str, samples: Path, out_dir: Path):
    ply_path = samples / "pointcloud" / f"{name}.ply"
    info_path = samples / "info" / f"{name}.txt"

    # Open log file alongside the mask output
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}_masking.log"
    _set_log_file(log_path)

    try:
        info = parse_info(info_path)

        # Echo parsed info so the log is self-contained
        log("\n━━━ [PARSED INFO] ━━━")
        log(f"  rgb_shape       : {info.rgb_shape}")
        log(f"  raw_depth_shape : {info.raw_depth_shape}")
        log(f"  crop_top        : {info.crop_top}")
        log(f"  crop_left       : {info.crop_left}")
        log(f"  fx={info.fx:.3f}  fy={info.fy:.3f}  cx={info.cx:.3f}  cy={info.cy:.3f}")

        mask = run_masking(ply_path, info)

        mask_target_path = out_dir / f"{name}.png"
        cv2.imwrite(str(mask_target_path), mask)

        fg_px = int(np.count_nonzero(mask))

        log("\n" + "=" * 80)
        log("                UNBUFFERED DIAGNOSTIC ENGINE VERBOSE REPORT")
        log("=" * 80)
        log(f"  Target Image Asset Name : {name}")
        log(
            f"  Info file resolution    : H={info.rgb_shape[0]} px, W={info.rgb_shape[1]} px"
        )
        log(
            f"  Crop Offset Regimes     : Top={info.crop_top} px, Left={info.crop_left} px"
        )
        log(f"  Saved Image Output Path : {mask_target_path}")
        log(f"  Log File Path           : {log_path}")
        log(f"  Mask Active White Pixels: {fg_px:,}")
        log(
            f"  Execution Pipeline Status: {'SUCCESS ✓' if fg_px > 0 else 'EMPTY BLACK OUTPUT ✗'}"
        )
        log("=" * 80 + "\n")

    finally:
        _close_log_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="img0000")
    parser.add_argument("--base", default="offline_case/samples")
    parser.add_argument("--out", default="offline_case/samples/mask")
    args = parser.parse_args()
    run(name=args.name, samples=Path(args.base), out_dir=Path(args.out))
