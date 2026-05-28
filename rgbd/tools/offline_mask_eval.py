import argparse
import json
import sys
import re
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.segmentation_helper import SegmentationHelper


def load_rgb(path):
    rgb_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"Could not load RGB image: {path}")
    return cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)


def load_depth(path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not load depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape={depth.shape}")
    return depth


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask image: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8)


def parse_intrinsics_info(info_path):
    info_path = Path(info_path)
    if not info_path.exists():
        raise FileNotFoundError(f"Could not find intrinsics file: {info_path}")

    lines = info_path.read_text().splitlines()
    matrix_lines = []
    capture = False
    for line in lines:
        if line.strip().lower().startswith("intrinsics matrix"):
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("Mask QA:") or stripped.startswith("Artifacts:"):
                break
            matrix_lines.append(stripped)
            if len(matrix_lines) == 3:
                break

    if len(matrix_lines) == 3:
        rows = []
        for line in matrix_lines:
            row = [float(v) for v in line.replace(",", " ").split()]
            if len(row) != 3:
                raise ValueError(f"Invalid intrinsics matrix row in {info_path}: {line}")
            rows.append(row)
        mat = np.array(rows, dtype=np.float64)
        return {
            "width": 0,
            "height": 0,
            "fx": float(mat[0, 0]),
            "fy": float(mat[1, 1]),
            "cx": float(mat[0, 2]),
            "cy": float(mat[1, 2]),
            "matrix": mat,
            "crop": None,
            "raw_rgb_shape": None,
            "raw_depth_shape": None,
            "cropped_rgb_shape": None,
            "cropped_depth_shape": None,
        }

    def parse_shape(value):
        nums = [int(v) for v in re.findall(r"\d+", value)]
        if len(nums) >= 2:
            return tuple(nums)
        return None

    def parse_crop(value):
        match = re.search(
            r"top=(\d+)\s+bottom=(\d+)\s+left=(\d+)\s+right=(\d+)", value
        )
        if not match:
            return None
        return {
            "top": int(match.group(1)),
            "bottom": int(match.group(2)),
            "left": int(match.group(3)),
            "right": int(match.group(4)),
        }

    # Fallback for compact key/value info dumps
    values = {}
    crop = None
    raw_rgb_shape = None
    raw_depth_shape = None
    cropped_rgb_shape = None
    cropped_depth_shape = None
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("crop:"):
            continue
        if stripped.startswith("top=") and "left=" in stripped and "right=" in stripped:
            crop = parse_crop(stripped)
            continue
        if stripped.lower().startswith("raw rgb shape:"):
            raw_rgb_shape = parse_shape(stripped.split(":", 1)[1].strip())
            continue
        if stripped.lower().startswith("raw depth shape:"):
            raw_depth_shape = parse_shape(stripped.split(":", 1)[1].strip())
            continue
        if stripped.lower().startswith("cropped rgb shape:"):
            cropped_rgb_shape = parse_shape(stripped.split(":", 1)[1].strip())
            continue
        if stripped.lower().startswith("cropped depth shape:"):
            cropped_depth_shape = parse_shape(stripped.split(":", 1)[1].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key in {"fx", "fy", "cx", "cy", "width", "height"}:
            values[key] = float(value)

    if {"fx", "fy", "cx", "cy"} <= values.keys():
        return {
            "width": int(values.get("width", 0)),
            "height": int(values.get("height", 0)),
            "fx": float(values["fx"]),
            "fy": float(values["fy"]),
            "cx": float(values["cx"]),
            "cy": float(values["cy"]),
            "matrix": None,
            "crop": crop,
            "raw_rgb_shape": raw_rgb_shape,
            "raw_depth_shape": raw_depth_shape,
            "cropped_rgb_shape": cropped_rgb_shape,
            "cropped_depth_shape": cropped_depth_shape,
        }

    raise ValueError(
        f"Could not parse intrinsics from {info_path}. "
        "Expected either an intrinsics matrix or fx/fy/cx/cy fields."
    )


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
    h, w = mask_u8.shape
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

    border_mask = np.zeros_like(mask_u8, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_pixels = int(np.count_nonzero(mask_u8 & border_mask))
    border_ratio = float(border_pixels / foreground) if foreground else 0.0

    return {
        "shape": (h, w),
        "foreground": foreground,
        "ratio": ratio,
        "components": int(num_labels - 1),
        "largest_area": largest_area,
        "largest_bbox": largest_bbox,
        "largest_centroid": largest_centroid,
        "border_pixels": border_pixels,
        "border_ratio": border_ratio,
    }


def overlay_mask(rgb, mask, color=(0, 255, 0), alpha=0.28):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    overlay = cv2.addWeighted(overlay, 1.0 - alpha, tint, alpha, 0)
    mask_u8 = (mask > 0).astype(np.uint8)
    overlay[mask_u8 > 0] = color
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(overlay, [largest], -1, (0, 255, 255), 2)
    return overlay


def resize_mask(mask, image_shape):
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape == image_shape:
        return mask_u8
    target_h, target_w = image_shape
    return cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def dice_iou(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    sa = int(np.count_nonzero(a))
    sb = int(np.count_nonzero(b))
    dice = (2.0 * inter / (sa + sb)) if (sa + sb) else 0.0
    iou = (inter / union) if union else 0.0
    return {"intersection": inter, "union": union, "dice": dice, "iou": iou}


def depth_to_mask(depth, intrinsics):
    helper = SegmentationHelper(intrinsics)
    mask, plane_model, inliers, outliers, _ = helper.segment(depth, None)
    return mask, plane_model, inliers, outliers


def pointcloud_to_mask(ply_path, intrinsics, image_shape):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) < 3:
        raise ValueError(f"Point cloud has too few points: {ply_path}")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.012, ransac_n=3, num_iterations=2000
    )
    [a, b, c, d] = plane_model
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(outlier_cloud.points)
    for x, y, z in pts:
        if z <= 0:
            continue
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        if 0 <= u < w and 0 <= v < h:
            mask[v, u] = 1

    kernel_close = np.ones((5, 5), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask, plane_model, len(inliers), len(outlier_cloud.points)

    min_area = max(8, int(h * w * 0.001))
    selected = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            selected.append(idx)

    if not selected:
        selected = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]

    cleaned = np.zeros_like(mask)
    for idx in selected:
        cleaned[labels == idx] = 1

    return cleaned, plane_model, len(inliers), len(outlier_cloud.points)


def write_report(path, data):
    path = Path(path)
    with path.open("w") as f:
        for key, value in data.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Offline RGB-D mask evaluation for saved captures"
    )
    parser.add_argument("--rgb", type=Path, required=True, help="Path to RGB image")
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional existing mask image to evaluate directly",
    )
    parser.add_argument("--depth", type=Path, help="Path to raw 16-bit depth image")
    parser.add_argument("--ply", type=Path, help="Path to saved point cloud")
    parser.add_argument(
        "--reference-mask",
        type=Path,
        help="Optional reference mask to compare against",
    )
    parser.add_argument(
        "--intrinsics-file",
        type=Path,
        help="Path to an info text file containing an intrinsics matrix or fx/fy/cx/cy",
    )
    parser.add_argument("--width", type=int, help="Image width override")
    parser.add_argument("--height", type=int, help="Image height override")
    parser.add_argument("--fx", type=float, help="Focal length x")
    parser.add_argument("--fy", type=float, help="Focal length y")
    parser.add_argument("--cx", type=float, help="Optical center x")
    parser.add_argument("--cy", type=float, help="Optical center y")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("offline_eval"),
        help="Directory for evaluation artifacts",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Optional name stem used in output filenames",
    )
    args = parser.parse_args()

    rgb = load_rgb(args.rgb)
    rgb_h, rgb_w = rgb.shape[:2]

    need_intrinsics = args.depth is not None or args.ply is not None
    intrinsics = None
    width = rgb_w
    height = rgb_h
    fx = fy = cx = cy = None
    if need_intrinsics:
        if args.intrinsics_file:
            intr = parse_intrinsics_info(args.intrinsics_file)
            width = int(args.width or intr["width"] or rgb_w)
            height = int(args.height or intr["height"] or rgb_h)
            fx = float(args.fx or intr["fx"])
            fy = float(args.fy or intr["fy"])
            cx = float(args.cx or intr["cx"])
            cy = float(args.cy or intr["cy"])
        else:
            missing = [
                name
                for name in ("width", "height", "fx", "fy", "cx", "cy")
                if getattr(args, name) is None
            ]
            if missing:
                raise SystemExit(
                    "Missing intrinsics. Provide --intrinsics-file or explicit "
                    "--width --height --fx --fy --cx --cy."
                )
            width = int(args.width)
            height = int(args.height)
            fx = float(args.fx)
            fy = float(args.fy)
            cx = float(args.cx)
            cy = float(args.cy)

        intrinsics = make_intrinsics(width, height, fx, fy, cx, cy)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or args.rgb.stem

    report = {
        "rgb_path": str(args.rgb),
        "rgb_shape": tuple(rgb.shape),
    }
    if intrinsics is not None:
        report["intrinsics"] = {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }

    input_mask = None
    if args.mask:
        input_mask = resize_mask(load_mask(args.mask), rgb.shape[:2])
        report["input_mask_path"] = str(args.mask)
        report["input_mask"] = mask_stats(input_mask)

    depth_mask = None
    depth_metrics = None
    if args.depth:
        depth = load_depth(args.depth)
        report["depth_path"] = str(args.depth)
        report["depth_shape"] = tuple(depth.shape)
        depth_mask, depth_plane, depth_inliers, depth_outliers = depth_to_mask(
            depth, intrinsics
        )
        depth_mask = resize_mask(depth_mask, rgb.shape[:2])
        depth_metrics = mask_stats(depth_mask)
        report["depth_plane"] = {
            "a": float(depth_plane[0]),
            "b": float(depth_plane[1]),
            "c": float(depth_plane[2]),
            "d": float(depth_plane[3]),
            "inliers": int(depth_inliers),
            "outliers": int(depth_outliers),
        }
        report["depth_mask"] = depth_metrics

    pcd_mask = None
    pcd_metrics = None
    if args.ply:
        report["ply_path"] = str(args.ply)
        pcd_mask, pcd_plane, pcd_inliers, pcd_outliers = pointcloud_to_mask(
            args.ply, intrinsics, rgb.shape[:2]
        )
        pcd_metrics = mask_stats(pcd_mask)
        report["pointcloud_plane"] = {
            "a": float(pcd_plane[0]),
            "b": float(pcd_plane[1]),
            "c": float(pcd_plane[2]),
            "d": float(pcd_plane[3]),
            "inliers": int(pcd_inliers),
            "outliers": int(pcd_outliers),
        }
        report["pointcloud_mask"] = pcd_metrics

    if depth_mask is not None and pcd_mask is not None:
        agreement = dice_iou(depth_mask, pcd_mask)
        report["depth_vs_pointcloud"] = agreement
        print(
            "[AGREEMENT] "
            f"IoU={agreement['iou']:.4f} Dice={agreement['dice']:.4f} "
            f"intersection={agreement['intersection']} union={agreement['union']}"
        )

    if args.reference_mask:
        ref_mask = load_mask(args.reference_mask)
        ref_mask = resize_mask(ref_mask, rgb.shape[:2])
        report["reference_mask_path"] = str(args.reference_mask)
        report["reference_mask"] = mask_stats(ref_mask)
        if input_mask is not None:
            report["input_vs_reference"] = dice_iou(input_mask, ref_mask)
        if depth_mask is not None:
            report["depth_vs_reference"] = dice_iou(depth_mask, ref_mask)
        if pcd_mask is not None:
            report["pcd_vs_reference"] = dice_iou(pcd_mask, ref_mask)

    if input_mask is not None:
        final_mask = input_mask
        report["selected_mask_source"] = "input_mask"
    elif depth_mask is not None:
        final_mask = depth_mask
        report["selected_mask_source"] = "depth"
    elif pcd_mask is not None:
        final_mask = pcd_mask
        report["selected_mask_source"] = "pointcloud"
    else:
        raise SystemExit("No mask source available. Provide --mask, --depth, and/or --ply.")

    final_stats = mask_stats(final_mask)
    overlay = overlay_mask(rgb, final_mask)
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    report_path = output_dir / f"{stem}_report.txt"
    json_path = output_dir / f"{stem}_report.json"

    cv2.imwrite(str(mask_path), (final_mask > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(overlay_path), overlay)
    write_report(report_path, report)
    json_path.write_text(json.dumps(report, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))

    print(f"[SAVED] Mask: {mask_path}")
    print(f"[SAVED] Overlay: {overlay_path}")
    print(f"[SAVED] Report: {report_path}")
    print(f"[SAVED] JSON: {json_path}")
    print(
        "[QA] "
        f"foreground={final_stats['foreground']} "
        f"components={final_stats['components']} "
        f"border_ratio={final_stats['border_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
