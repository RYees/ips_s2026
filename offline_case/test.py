"""
offline_visualize.py  (test.py)
--------------------------------
Reproduces the RGB | Mask | Depth-TURBO visualization from main.py
using only saved files — no camera required.

Expected folder layout (offline_case/samples/):
    images/       img0000.png   <- cropped RGB
    depth/        img0000.png   <- raw 16-bit depth  (optional, preferred)
    masks/        img0000.png   <- binary mask        (optional)
    pointcloud/   img0000.ply   <- full scene PLY     (fallback for depth)
    info/         img0000.txt   <- capture metadata

Usage:
    python offline_case/test.py
    python offline_case/test.py img0042
    python offline_case/test.py img0042 --save
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Info.txt parser  — reads every field from the capture metadata
# ---------------------------------------------------------------------------


@dataclass
class CaptureInfo:
    # Crop applied to RGB only (depth is never cropped)
    crop_top: int = 0
    crop_bottom: int = 0
    crop_left: int = 0
    crop_right: int = 0

    # Frame dimensions
    raw_rgb_shape: Tuple[int, int, int] = (0, 0, 3)
    raw_depth_shape: Tuple[int, int] = (0, 0)
    cropped_rgb_shape: Tuple[int, int, int] = (0, 0, 3)

    # Depth value range (used for accurate colormap normalization)
    depth_min: int = 0
    depth_max: int = 0
    valid_depth_pixels: int = 0

    # Camera intrinsics (always for the depth frame coordinate space)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    # Plane equation  ax + by + cz + d = 0
    plane_a: float = 0.0
    plane_b: float = 0.0
    plane_c: float = 0.0
    plane_d: float = 0.0
    plane_inliers: int = 0
    non_plane_points: int = 0

    # Mask QA
    foreground_pixels: int = 0


def parse_info(path: Path) -> CaptureInfo:
    info = CaptureInfo()
    text = path.read_text()
    lines = text.splitlines()

    def _ints(s):
        return [int(x) for x in re.findall(r"-?\d+", s)]

    def _floats(s):
        return [
            float(x) for x in re.findall(r"-?[\d]+\.[\d]+(?:[eE][+-]?\d+)?|-?[\d]+", s)
        ]

    for i, line in enumerate(lines):
        s = line.strip()

        # Crop
        if s.startswith("top="):
            parts = dict(kv.split("=") for kv in s.split())
            info.crop_top = int(parts.get("top", 0))
            info.crop_bottom = int(parts.get("bottom", 0))
            info.crop_left = int(parts.get("left", 0))
            info.crop_right = int(parts.get("right", 0))

        # Shapes
        elif s.startswith("Raw RGB shape:"):
            v = _ints(s)
            info.raw_rgb_shape = (
                (v[0], v[1], v[2]) if len(v) >= 3 else info.raw_rgb_shape
            )
        elif s.startswith("Raw depth shape:"):
            v = _ints(s)
            info.raw_depth_shape = (v[0], v[1]) if len(v) >= 2 else info.raw_depth_shape
        elif s.startswith("Cropped RGB shape:"):
            v = _ints(s)
            info.cropped_rgb_shape = (
                (v[0], v[1], v[2]) if len(v) >= 3 else info.cropped_rgb_shape
            )

        # Depth stats
        elif s.startswith("Depth min/max:"):
            v = _ints(s)
            info.depth_min, info.depth_max = (v[0], v[1]) if len(v) >= 2 else (0, 0)
        elif s.startswith("Valid depth pixels:"):
            v = _ints(s)
            info.valid_depth_pixels = v[0] if v else 0

        # Intrinsics matrix (3 rows follow)
        elif s.startswith("Intrinsics matrix:"):
            r0 = _floats(lines[i + 1]) if i + 1 < len(lines) else []
            r1 = _floats(lines[i + 2]) if i + 2 < len(lines) else []
            if len(r0) >= 3:
                info.fx, info.cx = r0[0], r0[2]
            if len(r1) >= 3:
                info.fy, info.cy = r1[1], r1[2]

        # Plane equation
        elif s.startswith("Plane equation:"):
            v = _floats(s)
            if len(v) >= 4:
                info.plane_a, info.plane_b, info.plane_c, info.plane_d = (
                    v[0],
                    v[1],
                    v[2],
                    v[3],
                )

        elif s.startswith("Plane inliers:"):
            v = _ints(s)
            info.plane_inliers = v[0] if v else 0
        elif s.startswith("Non-plane points:"):
            v = _ints(s)
            info.non_plane_points = v[0] if v else 0
        elif s.startswith("Foreground pixels:"):
            v = _ints(s)
            info.foreground_pixels = v[0] if v else 0

    return info


def print_info(info: CaptureInfo):
    print(
        f"  Crop          : top={info.crop_top} bottom={info.crop_bottom} "
        f"left={info.crop_left} right={info.crop_right}"
    )
    print(f"  Raw RGB       : {info.raw_rgb_shape}")
    print(f"  Raw depth     : {info.raw_depth_shape}")
    print(f"  Depth range   : {info.depth_min} – {info.depth_max} mm")
    print(f"  Valid px      : {info.valid_depth_pixels}")
    print(
        f"  Intrinsics    : fx={info.fx:.3f} fy={info.fy:.3f} "
        f"cx={info.cx:.3f} cy={info.cy:.3f}"
    )
    print(
        f"  Plane eq      : {info.plane_a:.6f}x + {info.plane_b:.6f}y + "
        f"{info.plane_c:.6f}z + {info.plane_d:.6f} = 0"
    )
    print(
        f"  Plane inliers : {info.plane_inliers}   non-plane: {info.non_plane_points}"
    )
    print(f"  Foreground px : {info.foreground_pixels}")


# ---------------------------------------------------------------------------
# Depth loading / reconstruction
# ---------------------------------------------------------------------------


def load_depth_png(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth PNG not found: {path}")
    return depth.astype(np.uint16)


def depth_from_ply(ply_path: Path, info: CaptureInfo) -> np.ndarray:
    """Re-project full PLY back to a depth image using capture intrinsics."""
    H, W = info.raw_depth_shape  # depth is never cropped

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print("[WARNING] PLY is empty — depth image will be blank.")
        return np.zeros((H, W), dtype=np.uint16)

    pts = np.asarray(pcd.points)  # (N, 3) metres

    # Filter invalid points
    valid = pts[:, 2] > 0
    pts = pts[valid]

    # Project — use raw intrinsics, no crop shift (depth lives in its own space)
    u = np.round(pts[:, 0] * info.fx / pts[:, 2] + info.cx).astype(int)
    v = np.round(pts[:, 1] * info.fy / pts[:, 2] + info.cy).astype(int)
    z_mm = (pts[:, 2] * 1000.0).astype(np.float32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z_mm = u[in_bounds], v[in_bounds], z_mm[in_bounds]

    # Keep nearest point per pixel
    order = np.argsort(z_mm)
    u, v, z_mm = u[order], v[order], z_mm[order]

    depth_img = np.zeros((H, W), dtype=np.float32)
    depth_img[v, u] = z_mm

    print(
        f"[OK] Projected {len(u):,} PLY points → {W}×{H} depth image  "
        f"non-zero px: {np.count_nonzero(depth_img):,}"
    )
    return depth_img.astype(np.uint16)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def colorize_depth(depth):

    depth = depth.astype(np.float32)

    valid = depth > 0

    depth_vis = np.zeros(depth.shape, dtype=np.uint8)

    if np.any(valid):
        normalized = cv2.normalize(depth[valid], None, 0, 255, cv2.NORM_MINMAX)

        depth_vis[valid] = normalized.reshape(-1).astype(np.uint8)

    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_WINTER)


def build_mask_panel(mask: np.ndarray) -> np.ndarray:
    # Flatten to single channel regardless of how file was loaded
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_viz = (mask > 0).astype(np.uint8) * 255
    mask_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_bgr, [largest], -1, (0, 255, 0), 2)
    return mask_bgr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(name: str, base: Path, save: bool):
    rgb_path = base / "images" / f"{name}.png"
    depth_path = base / "depth" / f"{name}.png"
    mask_path = base / "masks" / f"{name}.png"
    ply_path = base / "pointcloud" / f"{name}.ply"
    info_path = base / "info" / f"{name}.txt"

    # 1. Parse metadata
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    info = parse_info(info_path)
    print(f"[OK] Metadata loaded from {info_path}")
    print_info(info)

    # 2. Load RGB
    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    print(f"[OK] RGB loaded  {rgb_bgr.shape}")

    # 3. Load or reconstruct depth
    if depth_path.exists():
        depth = load_depth_png(depth_path)
        print(
            f"[OK] Depth loaded from PNG  {depth.shape}  "
            f"min={depth.min()} max={depth.max()}"
        )
    elif ply_path.exists():
        print("[INFO] No depth PNG — reconstructing from PLY...")
        depth = depth_from_ply(ply_path, info)
    else:
        raise FileNotFoundError(
            f"Neither depth PNG ({depth_path}) nor PLY ({ply_path}) found."
        )

    # 4. Load mask (optional)
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(f"[OK] Mask loaded  foreground px: {np.count_nonzero(mask):,}")
    else:
        print("[INFO] No mask file — using blank mask.")
        mask = np.zeros(depth.shape[:2], dtype=np.uint8)

    # 5. Colorize depth anchored to belt plane distance
    depth_colored = colorize_depth(depth)

    # 6. Build panels — use native sizes, no forced resize
    panel_rgb = rgb_bgr
    panel_mask = cv2.resize(
        build_mask_panel(mask), (rgb_bgr.shape[1], rgb_bgr.shape[0])
    )
    panel_depth = cv2.resize(depth_colored, (rgb_bgr.shape[1], rgb_bgr.shape[0]))

    combined = np.hstack((panel_rgb, panel_mask, panel_depth))

    # 7. Always save depth PNG to samples/depth/
    depth_out = base / "depth" / f"{name}.png"
    depth_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(depth_out), depth_colored)
    print(f"[SAVED] Depth image → {depth_out}")

    # 8. Optionally save combined viz
    if save:
        out_path = base / "debug" / f"{name}_offline_viz.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined)
        print(f"[SAVED] Combined viz → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline depth visualizer")
    parser.add_argument("name", nargs="?", default="img0000")
    parser.add_argument("--base", default="offline_case/samples")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    run(name=args.name, base=Path(args.base), save=args.save)
