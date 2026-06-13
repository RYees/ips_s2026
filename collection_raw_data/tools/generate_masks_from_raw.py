import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.masking import (
    CaptureInfo,
    _close_log_file,
    _set_log_file,
    run_masking_from_point_cloud,
)
from processing.annotation_writer import AnnotationWriter


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape={depth.shape}")
    return depth


def crop_manual(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]
    top = max(0, int(top))
    bottom = max(0, int(bottom))
    left = max(0, int(left))
    right = max(0, int(right))
    if top + bottom >= h or left + right >= w:
        return img, 0, 0
    return (
        img[top : h - bottom if bottom > 0 else h, left : w - right if right > 0 else w],
        left,
        top,
    )


def parse_ints(text):
    return [int(v) for v in re.findall(r"-?\d+", text)]


def parse_floats(text):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)]


def parse_info(info_path: Path):
    info = {
        "crop": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "raw_rgb_shape": None,
        "raw_depth_shape": None,
        "cropped_rgb_shape": None,
        "cropped_depth_shape": None,
        "alignment_mode": "aligned_to_rgb",
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "selected_class": 0,
    }
    lines = info_path.read_text().splitlines()

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("top=") and "left=" in s and "right=" in s:
            vals = parse_ints(s)
            if len(vals) >= 4:
                info["crop"] = {
                    "top": vals[0],
                    "bottom": vals[1],
                    "left": vals[2],
                    "right": vals[3],
                }
        elif s.lower().startswith("raw rgb shape:"):
            vals = parse_ints(s)
            if len(vals) >= 2:
                info["raw_rgb_shape"] = (vals[0], vals[1])
        elif s.lower().startswith("raw depth shape:"):
            vals = parse_ints(s)
            if len(vals) >= 2:
                info["raw_depth_shape"] = (vals[0], vals[1])
        elif s.lower().startswith("cropped rgb shape:"):
            vals = parse_ints(s)
            if len(vals) >= 2:
                info["cropped_rgb_shape"] = (vals[0], vals[1])
        elif s.lower().startswith("cropped depth shape:"):
            vals = parse_ints(s)
            if len(vals) >= 2:
                info["cropped_depth_shape"] = (vals[0], vals[1])
        elif s.lower().startswith("alignment_mode="):
            info["alignment_mode"] = s.split("=", 1)[1].strip()
        elif s.lower().startswith("intrinsics matrix:"):
            row0 = parse_floats(lines[i + 1]) if i + 1 < len(lines) else []
            row1 = parse_floats(lines[i + 2]) if i + 2 < len(lines) else []
            row2 = parse_floats(lines[i + 3]) if i + 3 < len(lines) else []
            if len(row0) >= 3 and len(row1) >= 3 and len(row2) >= 3:
                info["fx"] = row0[0]
                info["fy"] = row1[1]
                info["cx"] = row0[2]
                info["cy"] = row1[2]
        elif s.lower().startswith("selected class:"):
            vals = parse_ints(s)
            if vals:
                info["selected_class"] = int(vals[0])
    return info


def make_intrinsics(width, height, fx, fy, cx, cy):
    return o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )


def mask_stats(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    foreground = int(np.count_nonzero(mask_u8))
    total = int(mask_u8.size)
    ratio = float(foreground / total) if total else 0.0

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    largest_area = int(component_areas.max()) if component_areas.size else 0
    largest_idx = int(np.argmax(component_areas)) + 1 if component_areas.size else -1
    largest_bbox = None
    largest_centroid = None
    if largest_idx > 0:
        x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
        y = int(stats[largest_idx, cv2.CC_STAT_TOP])
        bw = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
        largest_bbox = (x, y, bw, bh)
        largest_centroid = (
            float(centroids[largest_idx][0]),
            float(centroids[largest_idx][1]),
        )
    return {
        "foreground": foreground,
        "ratio": ratio,
        "components": int(num_labels - 1),
        "largest_area": largest_area,
        "largest_bbox": largest_bbox,
        "largest_centroid": largest_centroid,
    }


def overlay_mask(rgb, mask):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    overlay[(mask > 0)] = [0, 255, 0]
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest], -1, (0, 0, 255), 2)
    return overlay


def resolve_depth_crop(depth, info):
    crop = info["crop"]
    raw_rgb_shape = info["raw_rgb_shape"] or depth.shape[:2]
    raw_depth_shape = info["raw_depth_shape"] or depth.shape[:2]
    aligned = raw_depth_shape == raw_rgb_shape or info["alignment_mode"] == "aligned_to_rgb"

    if aligned:
        depth_crop, crop_left, crop_top = crop_manual(
            depth,
            top=crop["top"],
            bottom=crop["bottom"],
            left=crop["left"],
            right=crop["right"],
        )
        return depth_crop, crop_left, crop_top, False

    scale_x = depth.shape[1] / raw_rgb_shape[1]
    scale_y = depth.shape[0] / raw_rgb_shape[0]
    depth_crop, crop_left, crop_top = crop_manual(
        depth,
        top=int(round(crop["top"] * scale_y)),
        bottom=int(round(crop["bottom"] * scale_y)),
        left=int(round(crop["left"] * scale_x)),
        right=int(round(crop["right"] * scale_x)),
    )
    return depth_crop, crop_left, crop_top, True


def adjust_intrinsics(info, depth_shape, crop_left, crop_top, legacy=False):
    fx = info["fx"]
    fy = info["fy"]
    cx = info["cx"]
    cy = info["cy"]
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError("Missing intrinsics in info file.")

    if not legacy:
        return make_intrinsics(
            depth_shape[1], depth_shape[0], fx, fy, cx - crop_left, cy - crop_top
        )

    raw_rgb_shape = info["raw_rgb_shape"]
    if raw_rgb_shape is None:
        return make_intrinsics(
            depth_shape[1], depth_shape[0], fx, fy, cx - crop_left, cy - crop_top
        )

    scale_x = depth_shape[1] / raw_rgb_shape[1]
    scale_y = depth_shape[0] / raw_rgb_shape[0]
    return make_intrinsics(
        depth_shape[1],
        depth_shape[0],
        fx * scale_x,
        fy * scale_y,
        (cx * scale_x) - crop_left,
        (cy * scale_y) - crop_top,
    )


def process_capture(stem: str, dataset_root: Path, overwrite: bool = False):
    images_dir = dataset_root / "images"
    depth_dir = dataset_root / "depth"
    info_dir = dataset_root / "info"
    masks_dir = dataset_root / "masks"
    labels_dir = dataset_root / "labels"
    debug_dir = dataset_root / "debug" / "mask_generation"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = images_dir / f"{stem}.png"
    depth_path = depth_dir / f"{stem}.png"
    info_path = info_dir / f"{stem}.txt"
    mask_path = masks_dir / f"{stem}.png"
    overlay_path = debug_dir / f"{stem}_overlay.png"
    report_path = debug_dir / f"{stem}_report.txt"
    label_path = labels_dir / f"{stem}.txt"
    log_path = debug_dir / f"{stem}_masking.log"

    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing depth image: {depth_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info file: {info_path}")
    if mask_path.exists() and not overwrite:
        print(f"[SKIP] {stem} already has a mask. Use --overwrite to replace it.")
        return False

    _set_log_file(log_path)
    try:
        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path)
        info = parse_info(info_path)

        depth_crop, crop_left, crop_top, legacy = resolve_depth_crop(depth, info)
        if info["cropped_rgb_shape"] is not None:
            expected_h, expected_w = info["cropped_rgb_shape"]
            if (depth_crop.shape[0], depth_crop.shape[1]) != (expected_h, expected_w):
                print(
                    f"[WARN] {stem}: depth crop {depth_crop.shape[:2]} does not match cropped RGB {info['cropped_rgb_shape']}"
                )

        intrinsics = adjust_intrinsics(
            info, depth_crop.shape[:2], crop_left, crop_top, legacy=legacy
        )
        depth_m = depth_crop.astype(np.float32) / 1000.0
        depth_m = np.where((depth_m > 0.2) & (depth_m < 1.5), depth_m, 0.0)
        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        live_info = CaptureInfo(
            raw_depth_shape=depth_crop.shape[:2],
            rgb_shape=rgb.shape[:2],
            crop_top=0,
            crop_left=0,
            fx=float(intrinsics.intrinsic_matrix[0, 0]),
            fy=float(intrinsics.intrinsic_matrix[1, 1]),
            cx=float(intrinsics.intrinsic_matrix[0, 2]),
            cy=float(intrinsics.intrinsic_matrix[1, 2]),
        )
        mask, plane_model, inliers, outliers = run_masking_from_point_cloud(
            pcd, live_info
        )
        writer = AnnotationWriter()
        selected_class = int(info.get("selected_class", 0))

        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(overlay_path), overlay_mask(rgb, mask))
        writer.write(str(label_path), mask, rgb.shape[:2], label_class=selected_class)

        stats = mask_stats(mask)
        report_lines = [
        f"stem: {stem}",
        f"rgb_path: {rgb_path}",
        f"depth_path: {depth_path}",
        f"info_path: {info_path}",
        f"mask_path: {mask_path}",
        f"label_path: {label_path}",
        f"overlay_path: {overlay_path}",
        f"raw_rgb_shape: {info['raw_rgb_shape']}",
        f"raw_depth_shape: {info['raw_depth_shape']}",
        f"cropped_rgb_shape: {info['cropped_rgb_shape']}",
        f"cropped_depth_shape: {info['cropped_depth_shape']}",
        f"alignment_mode: {info['alignment_mode']}",
        f"crop: {info['crop']}",
        f"legacy_crop_adjustment: {legacy}",
        f"selected_class: {selected_class}",
        f"mask_foreground: {stats['foreground']}",
        f"mask_ratio: {stats['ratio']:.8f}",
        f"mask_components: {stats['components']}",
        f"mask_largest_area: {stats['largest_area']}",
        f"mask_largest_bbox: {stats['largest_bbox']}",
        f"mask_largest_centroid: {stats['largest_centroid']}",
        f"plane_model: {tuple(float(v) for v in plane_model)}",
        f"inliers: {int(inliers)}",
        f"outliers: {int(outliers)}",
    ]
        report_path.write_text("\n".join(report_lines) + "\n")
        print(f"[SAVED] {stem}: {mask_path}")
        return True
    finally:
        _close_log_file()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate masks from raw captures in collection_raw_data/data"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Raw dataset root containing images/, depth/, and info/",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Process only one capture stem, e.g. img0001",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks if they already exist",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    info_dir = dataset_root / "info"
    if not info_dir.exists():
        raise SystemExit(f"Info directory does not exist: {info_dir}")

    if args.stem:
        process_capture(args.stem, dataset_root, overwrite=args.overwrite)
        return

    stems = sorted(p.stem for p in info_dir.glob("img*.txt"))
    if not stems:
        raise SystemExit(f"No info files found in {info_dir}")

    processed = 0
    for stem in stems:
        try:
            if process_capture(stem, dataset_root, overwrite=args.overwrite):
                processed += 1
        except Exception as exc:
            print(f"[ERROR] {stem}: {exc}")

    print(f"[DONE] Generated {processed} masks into {dataset_root / 'masks'}")


if __name__ == "__main__":
    main()
