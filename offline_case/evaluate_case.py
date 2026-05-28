import json
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
RGBD_ROOT = ROOT / "rgbd"
if str(RGBD_ROOT) not in sys.path:
    sys.path.insert(0, str(RGBD_ROOT))

from tools.offline_mask_eval import (
    dice_iou,
    depth_to_mask,
    load_depth,
    load_mask,
    load_rgb,
    mask_stats,
    overlay_mask,
    parse_intrinsics_info,
    pointcloud_to_mask,
    resize_mask,
    make_intrinsics,
)


def build_case_paths(samples_root, stem):
    samples_root = Path(samples_root)
    return {
        "samples_root": samples_root,
        "stem": stem,
        "rgb": samples_root / "images" / f"{stem}.png",
        "depth": samples_root / "depth" / f"{stem}.png",
        "ply": samples_root / "pointcloud" / f"{stem}.ply",
        "reference": samples_root / "labels" / f"{stem}.txt",
    }


def list_available_stems(samples_root):
    images_dir = Path(samples_root) / "images"
    if not images_dir.exists():
        return []
    return sorted({path.stem for path in images_dir.glob("*.png")})


def pointcloud_depth_viz(ply_path, intrinsics, image_shape, colormap_name="TURBO"):
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise ValueError(f"Point cloud is empty: {ply_path}")

    h, w = image_shape
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    for x, y, z in pts:
        if z <= 0:
            continue
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        if 0 <= u < w and 0 <= v < h and z < depth_map[v, u]:
            depth_map[v, u] = z

    valid = np.isfinite(depth_map)
    if not np.any(valid):
        raise ValueError(
            f"Could not project any pointcloud samples into image space: {ply_path}"
        )

    filled = depth_map.copy()
    finite_vals = filled[valid]
    fill_value = float(np.max(finite_vals))
    filled[~valid] = fill_value
    depth_mm = (filled * 1000.0).astype(np.float32)
    depth_vis = cv2.normalize(depth_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = getattr(cv2, f"COLORMAP_{colormap_name.upper()}", cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(depth_vis, cmap)


def depth_png_viz(depth, colormap_name="TURBO"):
    depth = np.asarray(depth)
    valid = depth > 0
    if not np.any(valid):
        raise ValueError("Depth image has no valid pixels")
    depth_mm = depth.astype(np.float32)
    depth_vis = np.zeros_like(depth_mm, dtype=np.uint8)
    depth_vis[valid] = cv2.normalize(
        depth_mm[valid], None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    cmap = getattr(cv2, f"COLORMAP_{colormap_name.upper()}", cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(depth_vis, cmap)


def fallback_intrinsics_for_shape(image_shape):
    h, w = image_shape
    focal = float(max(w, h))
    return make_intrinsics(w, h, focal, focal, w / 2.0, h / 2.0)


def main():
    samples_root = Path(__file__).resolve().parent / "samples"
    available = list_available_stems(samples_root)
    default_stem = available[0] if len(available) == 1 else None
    parser = argparse.ArgumentParser(description="Offline mask evaluation case runner")
    parser.add_argument("--samples-root", type=Path, default=samples_root)
    parser.add_argument(
        "--stem",
        type=str,
        default=default_stem,
        help="Sample stem name, for example img0935",
    )
    parser.add_argument(
        "--reference-mask",
        type=Path,
        help="Optional reference mask for IoU/Dice comparison",
    )
    parser.add_argument(
        "--intrinsics-file",
        type=Path,
        help="Optional info.txt file with intrinsics matrix or fx/fy/cx/cy",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for overlays, mask, and report",
    )
    parser.add_argument(
        "--depth-colormap",
        type=str,
        default="TURBO",
        help="OpenCV colormap name for the pointcloud depth visualization",
    )
    args = parser.parse_args()

    if args.stem is None:
        if not available:
            raise SystemExit(
                f"No samples found in {samples_root / 'images'}. Add one or pass --stem."
            )
        raise SystemExit(
            f"Multiple sample stems found in {samples_root / 'images'}: {', '.join(available)}. "
            "Pass --stem to choose one."
        )

    paths = build_case_paths(args.samples_root, args.stem)
    rgb = load_rgb(paths["rgb"])
    rgb_h, rgb_w = rgb.shape[:2]

    intrinsics = None
    width = rgb_w
    height = rgb_h
    fx = fy = cx = cy = None
    if args.intrinsics_file:
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
                    "Depth/pointcloud masking needs intrinsics. Provide "
                    "--intrinsics-file or explicit --width --height --fx --fy --cx --cy."
                )
            width = int(args.width)
            height = int(args.height)
            fx = float(args.fx)
            fy = float(args.fy)
            cx = float(args.cx)
            cy = float(args.cy)

        intrinsics = make_intrinsics(width, height, fx, fy, cx, cy)
    elif args.width and args.height and args.fx and args.fy and args.cx and args.cy:
        width = int(args.width)
        height = int(args.height)
        fx = float(args.fx)
        fy = float(args.fy)
        cx = float(args.cx)
        cy = float(args.cy)
        intrinsics = make_intrinsics(width, height, fx, fy, cx, cy)
    else:
        intrinsics = fallback_intrinsics_for_shape(rgb.shape[:2])
        width = intrinsics.width
        height = intrinsics.height
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]
        report_intrinsics_source = "fallback_from_rgb_shape"

    output_dir = args.output_dir or (paths["samples_root"] / "output" / args.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CASE] {args.stem}")
    print(f"[CASE] RGB: {paths['rgb']}")
    print(f"[CASE] Depth: {paths['depth'] if paths['depth'].exists() else 'missing'}")
    print(f"[CASE] PLY: {paths['ply'] if paths['ply'].exists() else 'missing'}")

    report = {
        "sample_stem": args.stem,
        "samples_root": str(paths["samples_root"]),
        "rgb_path": str(paths["rgb"]),
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
        report["intrinsics_source"] = locals().get(
            "report_intrinsics_source", "provided_or_file"
        )

    input_mask = None
    if args.reference_mask and args.reference_mask.exists():
        input_mask = resize_mask(load_mask(args.reference_mask), rgb.shape[:2])
        report["reference_mask_path"] = str(args.reference_mask)
        report["reference_mask"] = mask_stats(input_mask)

    depth_mask = None
    if paths["depth"].exists():
        depth_img = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Could not load depth file: {paths['depth']}")
        report["depth_path"] = str(paths["depth"])
        report["depth_shape"] = tuple(depth_img.shape)
        if depth_img.ndim == 2:
            depth = depth_img
            report["depth_mode"] = "raw"
            depth_mask, depth_plane, depth_inliers, depth_outliers = depth_to_mask(
                depth, intrinsics
            )
            depth_mask = resize_mask(depth_mask, rgb.shape[:2])
            report["depth_plane"] = {
                "a": float(depth_plane[0]),
                "b": float(depth_plane[1]),
                "c": float(depth_plane[2]),
                "d": float(depth_plane[3]),
                "inliers": int(depth_inliers),
                "outliers": int(depth_outliers),
            }
            report["depth_mask"] = mask_stats(depth_mask)
        else:
            report["depth_mode"] = "visualization"
            report["depth_visualization"] = {
                "shape": tuple(depth_img.shape),
                "colormap": args.depth_colormap,
            }

    pcd_mask = None
    if paths["ply"].exists():
        report["ply_path"] = str(paths["ply"])
        pcd_mask, pcd_plane, pcd_inliers, pcd_outliers = pointcloud_to_mask(
            paths["ply"], intrinsics, rgb.shape[:2]
        )
        pcd_mask = resize_mask(pcd_mask, rgb.shape[:2])
        report["pointcloud_plane"] = {
            "a": float(pcd_plane[0]),
            "b": float(pcd_plane[1]),
            "c": float(pcd_plane[2]),
            "d": float(pcd_plane[3]),
            "inliers": int(pcd_inliers),
            "outliers": int(pcd_outliers),
        }
        report["pointcloud_mask"] = mask_stats(pcd_mask)
        report["pointcloud_depth_viz"] = f"generated with {args.depth_colormap}"
        if not paths["depth"].exists():
            paths["depth"].parent.mkdir(parents=True, exist_ok=True)
            depth_viz = pointcloud_depth_viz(
                paths["ply"],
                intrinsics,
                rgb.shape[:2],
                colormap_name=args.depth_colormap,
            )
            cv2.imwrite(str(paths["depth"]), depth_viz)
            report["generated_depth_path"] = str(paths["depth"])

    if input_mask is not None and pcd_mask is not None:
        report["reference_vs_pointcloud"] = dice_iou(input_mask, pcd_mask)
    if depth_mask is not None and pcd_mask is not None:
        report["depth_vs_pointcloud"] = dice_iou(depth_mask, pcd_mask)

    if depth_mask is not None:
        final_mask = depth_mask
        report["selected_mask_source"] = "depth"
    elif pcd_mask is not None:
        final_mask = pcd_mask
        report["selected_mask_source"] = "pointcloud"
    elif input_mask is not None:
        final_mask = input_mask
        report["selected_mask_source"] = "reference_mask"
    else:
        raise SystemExit("No usable mask source found in the case directory.")

    final_stats = mask_stats(final_mask)
    overlay = overlay_mask(rgb, final_mask)
    depth_vis_path = output_dir / f"{args.stem}_depthviz.png"
    if paths["depth"].exists():
        depth_img = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise FileNotFoundError(f"Could not load depth file: {paths['depth']}")
        if depth_img.ndim == 2:
            if depth_img.dtype != np.uint8:
                depth_viz = depth_png_viz(depth_img, colormap_name=args.depth_colormap)
            else:
                cmap = getattr(
                    cv2,
                    f"COLORMAP_{args.depth_colormap.upper()}",
                    cv2.COLORMAP_TURBO,
                )
                depth_viz = cv2.applyColorMap(depth_img, cmap)
        else:
            depth_viz = depth_img
        cv2.imwrite(str(depth_vis_path), depth_viz)
    elif paths["ply"].exists():
        depth_viz = pointcloud_depth_viz(
            paths["ply"], intrinsics, rgb.shape[:2], colormap_name=args.depth_colormap
        )
        cv2.imwrite(str(depth_vis_path), depth_viz)

    mask_path = output_dir / f"{args.stem}_mask.png"
    overlay_path = output_dir / f"{args.stem}_overlay.png"
    report_path = output_dir / f"{args.stem}_report.txt"
    json_path = output_dir / f"{args.stem}_report.json"

    cv2.imwrite(str(mask_path), (final_mask > 0).astype("uint8") * 255)
    cv2.imwrite(str(overlay_path), overlay)
    report_path.write_text(
        "\n".join(
            [
                f"{key}: {value}"
                if not isinstance(value, dict)
                else "\n".join([f"{key}:"] + [f"  {k}: {v}" for k, v in value.items()])
                for key, value in report.items()
            ]
        )
        + "\n"
    )
    json_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"[SAVED] Mask: {mask_path}")
    print(f"[SAVED] Overlay: {overlay_path}")
    if depth_vis_path.exists():
        print(f"[SAVED] Depth visualization: {depth_vis_path}")
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
