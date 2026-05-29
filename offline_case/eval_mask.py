import argparse
from pathlib import Path
import cv2
import numpy as np

from pointcloud_object_detector import detect_objects


# ------------------------------------------------------------
# LOAD RGB
# ------------------------------------------------------------


def load_rgb(base, name):
    path = base / "images" / f"{name}.png"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


# ------------------------------------------------------------
# VISUALIZE 3D RESULT ON RGB
# ------------------------------------------------------------


def draw_3d_overlay(rgb, clusters, best_id):

    out = rgb.copy()

    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]

    for i, (_, _, cluster) in enumerate(clusters):
        cid = i + 1
        color = (0, 255, 0) if cid == best_id else colors[i % len(colors)]

        # sparse projection for debug
        for p in cluster[::50]:
            x, y, z = p.astype(int)
            if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                out[y, x] = color

    cv2.putText(
        out,
        f"BEST: {best_id}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return out


# ------------------------------------------------------------
# VISUALIZE MASK PROPERLY (IMPORTANT PART)
# ------------------------------------------------------------


def visualize_mask(mask):

    # raw binary mask (what model outputs)
    raw = mask.copy()

    # make readable version
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    vis[mask > 0] = (0, 255, 0)

    # add contours for clarity
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)

    return raw, vis


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------


def run(name, base):

    base = Path(base)

    rgb = load_rgb(base, name)
    ply_path = base / "pointcloud" / f"{name}.ply"

    # fake intrinsics (replace later with info.txt parser)
    info = {
        "shape": rgb.shape[:2],
        "fx": 525.0,
        "fy": 525.0,
        "cx": rgb.shape[1] / 2,
        "cy": rgb.shape[0] / 2,
    }

    clusters, best_id, best_cluster, mask = detect_objects(ply_path, info)

    out_dir = base / "mask_eval"
    mask_dir = base / "mask"

    out_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # SAVE RAW MASK (IMPORTANT)
    # --------------------------------------------------------

    raw_mask_path = mask_dir / f"{name}.png"
    cv2.imwrite(str(raw_mask_path), mask)

    print(f"[OK] Saved raw mask → {raw_mask_path}")

    # --------------------------------------------------------
    # MASK VISUALIZATION (NEW)
    # --------------------------------------------------------

    raw, vis_mask = visualize_mask(mask)

    cv2.imwrite(str(out_dir / f"{name}_mask_raw.png"), raw)
    cv2.imwrite(str(out_dir / f"{name}_mask_vis.png"), vis_mask)

    # --------------------------------------------------------
    # 3D OVERLAY VISUALIZATION
    # --------------------------------------------------------

    vis_3d = draw_3d_overlay(rgb, clusters, best_id)

    cv2.imwrite(str(out_dir / f"{name}_pc_eval.png"), vis_3d)

    print(f"[OK] Saved evaluation → {out_dir}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="img0000")
    parser.add_argument("--base", default="offline_case/samples")

    args = parser.parse_args()
    run(args.name, args.base)
