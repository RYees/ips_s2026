import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import sys

# Ensure backend modules can be imported
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from masking import CaptureInfo, run_masking_from_point_cloud, parse_info


def run_batch_offline_debug():
    # Points to 'offline_case/dataset' relative to this script
    dataset_dir = ROOT_DIR / "dataset"

    pc_dir = dataset_dir / "pointcloud"
    info_dir = dataset_dir / "info"
    rgb_dir = dataset_dir / "cropped_rgb"
    mask_out_dir = dataset_dir / "masks"

    mask_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = dataset_dir / "offline_metrics_debug.csv"

    print(f"\n[DIAGNOSTIC] Looking for pointcloud files in:")
    print(f"  👉 {pc_dir.resolve()}\n")

    ply_files = sorted(list(pc_dir.glob("*.ply")))
    if not ply_files:
        print(f"[ERROR] No .ply files found inside that folder!")
        print(f"Please check if your files are actually located there.")
        return

    print(f"[START] Batch processing {len(ply_files)} frames...")

    with open(log_path, "w") as log_file:
        log_file.write(
            "frame,points,crop_left,crop_top,canvas_cx,projection_cx,calculated_shift_px\n"
        )

        for ply_path in ply_files:
            frame_name = ply_path.stem
            info_path = info_dir / f"{frame_name}.txt"
            rgb_path = rgb_dir / f"{frame_name}.png"

            if info_path.exists():
                info = parse_info(info_path)
            else:
                info = CaptureInfo()
                info.crop_left = 250
                info.crop_top = 0
                info.fx, info.fy, info.cx, info.cy = 750.14, 749.78, 636.17, 366.08

            pcd = o3d.io.read_point_cloud(str(ply_path))

            if rgb_path.exists():
                test_img = cv2.imread(str(rgb_path))
                img_shape = test_img.shape[:2]
            else:
                img_shape = (720, 510)

            info.rgb_shape = img_shape

            mask, plane_model, inliers, outliers = run_masking_from_point_cloud(
                pcd, info
            )

            canvas_cx = img_shape[1] / 2
            projection_cx = info.cx - info.crop_left
            shift_px = canvas_cx - projection_cx

            mask_path = mask_out_dir / f"{frame_name}.png"
            cv2.imwrite(str(mask_path), mask)

            log_line = f"{frame_name},{len(pcd.points)},{info.crop_left},{info.crop_top},{canvas_cx},{projection_cx:.2f},{shift_px:.2f}\n"
            log_file.write(log_line)
            print(f"  [PROCESSED] {frame_name} -> Mask generated.")

    print(f"\n[SUCCESS] Completed! Logs saved to: {log_path.resolve()}")


if __name__ == "__main__":
    run_batch_offline_debug()
