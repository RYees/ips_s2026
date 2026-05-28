"""
offline_visualize.py
--------------------
Reproduces the same RGB | Mask | Depth visualization from main.py
using only saved files — no camera required.

Expected folder layout (mirrors what main.py saves):
    dataset/
        images/       img0000.png          <- RGB image
        depth/        img0000.png          <- Raw 16-bit depth (uint16)
        masks/        img0000.png          <- Binary mask (0 or 255)
        pointcloud/   img0000.ply          <- Open3D point cloud
        info/         img0000.txt          <- Intrinsics / metadata

Usage:
    python offline_visualize.py                    # uses img0000
    python offline_visualize.py img0042            # specific capture name
    python offline_visualize.py img0042 --save     # also save result to disk
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_depth_png(path: Path) -> np.ndarray:
    """Load a raw 16-bit depth map saved by main.py (cv2.IMREAD_UNCHANGED)."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return depth


def depth_from_ply(ply_path: Path, info_path: Path, depth_shape: tuple) -> np.ndarray:
    """
    Fallback: re-project the PLY point cloud back onto a depth image.
    Uses the intrinsics parsed from the info .txt file.
    depth_shape = (H, W) of the depth frame.
    """
    # --- parse intrinsics from info txt ---
    fx = fy = cx = cy = None
    with open(info_path) as f:
        lines = f.readlines()
    # The 3x3 matrix is written as three lines after "Intrinsics matrix:"
    for i, line in enumerate(lines):
        if "Intrinsics matrix:" in line:
            row0 = list(map(float, lines[i + 1].split()))
            row1 = list(map(float, lines[i + 2].split()))
            fx, cx = row0[0], row0[2]
            fy, cy = row1[1], row1[2]
            break

    if fx is None:
        raise RuntimeError("Could not parse intrinsics from info file.")

    # Also parse the crop that was applied so we can adjust cx/cy
    crop_left = crop_top = 0
    for line in lines:
        if line.strip().startswith("top="):
            parts = dict(item.split("=") for item in line.split())
            crop_left = int(parts.get("left", 0))
            crop_top = int(parts.get("top", 0))
            break

    cx_adj = cx - crop_left
    cy_adj = cy - crop_top

    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        print("[WARNING] PLY file is empty; depth image will be blank.")
        return np.zeros(depth_shape, dtype=np.uint16)

    pts = np.asarray(pcd.points)  # (N, 3)  x, y, z in meters
    H, W = depth_shape
    depth_img = np.zeros((H, W), dtype=np.float32)

    for x, y, z in pts:
        if z <= 0:
            continue
        u = int(round(x * fx / z + cx_adj))
        v = int(round(y * fy / z + cy_adj))
        if 0 <= u < W and 0 <= v < H:
            z_mm = z * 1000.0
            # keep the nearest (smallest z) value if multiple points project here
            if depth_img[v, u] == 0 or z_mm < depth_img[v, u]:
                depth_img[v, u] = z_mm

    return depth_img.astype(np.uint16)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Apply TURBO colormap exactly as main.py does."""
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)


def build_mask_overlay(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reproduce the mask panel from main.py: gray+green tint, green contour."""
    mask_viz = (mask > 0).astype(np.uint8) * 255
    mask_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(mask_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_bgr, [largest], -1, (0, 255, 0), 2)

    return mask_bgr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(name: str, base: Path, save: bool):
    # --- resolve paths ---
    rgb_path = base / "images" / f"{name}.png"
    depth_path = base / "depth" / f"{name}.png"
    mask_path = base / "masks" / f"{name}.png"
    ply_path = base / "pointcloud" / f"{name}.ply"
    info_path = base / "info" / f"{name}.txt"

    # --- load RGB ---
    rgb_bgr = cv2.imread(str(rgb_path))
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    print(f"[OK] RGB loaded  {rgb_bgr.shape}")

    # --- load or reconstruct depth ---
    if depth_path.exists():
        depth = load_depth_png(depth_path)
        print(
            f"[OK] Depth loaded from saved PNG  {depth.shape}  dtype={depth.dtype}  "
            f"min={depth.min()} max={depth.max()}"
        )
    elif ply_path.exists() and info_path.exists():
        print("[INFO] No depth PNG found — reconstructing from PLY + intrinsics...")
        # Use the raw depth shape recorded in info.txt
        depth_h = depth_w = None
        with open(info_path) as f:
            for line in f:
                if line.startswith("Depth shape:"):
                    # e.g.  "Depth shape: (576, 640)"
                    nums = [
                        int(n)
                        for n in line.replace("(", "").replace(")", "").split()
                        if n.isdigit()
                    ]
                    if len(nums) == 2:
                        depth_h, depth_w = nums
                    break
        if depth_h is None:
            depth_h, depth_w = rgb_bgr.shape[0], rgb_bgr.shape[1]
        depth = depth_from_ply(ply_path, info_path, (depth_h, depth_w))
        print(
            f"[OK] Depth reconstructed from PLY  {depth.shape}  "
            f"non-zero pixels: {np.count_nonzero(depth)}"
        )
    else:
        raise FileNotFoundError(
            f"Neither depth PNG ({depth_path}) nor PLY ({ply_path}) found."
        )

    # --- load mask (optional) ---
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        print(
            f"[OK] Mask loaded  {mask.shape}  foreground px: {np.count_nonzero(mask)}"
        )
    else:
        print("[INFO] No mask file found — using blank mask.")
        mask = np.zeros(depth.shape, dtype=np.uint8)

    # --- colorize depth ---
    depth_colored = colorize_depth(depth)

    # --- build mask panel ---
    mask_bgr = build_mask_overlay(rgb_bgr, mask)

    # --- resize all three panels to the same display size (matches main.py) ---
    W, H = 320, 240
    panel_rgb = cv2.resize(rgb_bgr, (W, H))
    panel_mask = cv2.resize(mask_bgr, (W, H))
    panel_depth = cv2.resize(depth_colored, (W, H))

    combined = np.hstack((panel_rgb, panel_mask, panel_depth))

    # --- display ---
    win = f"Offline Visualization — {name}   [RGB | Mask | Depth-TURBO]"
    cv2.imshow(win, combined)
    print(f"\n[DISPLAY] Showing combined panel. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- optional save ---
    if save:
        out_path = base / "debug" / f"{name}_offline_viz.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), combined)
        print(f"[SAVED] Visualization saved to {out_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline depth visualizer")
    parser.add_argument(
        "name",
        nargs="?",
        default="img0000",
        help="Capture name, e.g. img0000 (default: img0000)",
    )
    parser.add_argument(
        "--base", default="dataset", help="Root dataset folder (default: dataset/)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the combined visualization to dataset/debug/",
    )
    args = parser.parse_args()

    run(
        name=args.name,
        base=Path(args.base),
        save=args.save,
    )
