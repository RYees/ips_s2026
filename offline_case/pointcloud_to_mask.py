import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
from sklearn.cluster import DBSCAN
import re


# ------------------------------------------------------------
# Parse intrinsics + plane from info.txt
# ------------------------------------------------------------


def parse_info(path: Path):
    text = path.read_text().splitlines()

    fx = fy = cx = cy = 0.0
    plane = None
    raw_depth_shape = (0, 0)

    def _floats(s):
        return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", s)]

    def _ints(s):
        return [int(x) for x in re.findall(r"-?\d+", s)]

    for i, line in enumerate(text):
        s = line.strip()

        if s.startswith("Raw depth shape:"):
            v = _ints(s)
            if len(v) >= 2:
                raw_depth_shape = (v[0], v[1])

        elif s.startswith("Intrinsics matrix:"):
            r0 = _floats(text[i + 1]) if i + 1 < len(text) else []
            r1 = _floats(text[i + 2]) if i + 2 < len(text) else []
            if len(r0) >= 3:
                fx, cx = r0[0], r0[2]
            if len(r1) >= 3:
                fy, cy = r1[1], r1[2]

        elif s.startswith("Plane equation:"):
            v = _floats(s)
            if len(v) >= 4:
                plane = v[:4]

    return fx, fy, cx, cy, plane, raw_depth_shape


# ------------------------------------------------------------
# Plane filtering (remove conveyor belt)
# ------------------------------------------------------------


def remove_plane(points, plane, threshold=0.01):
    a, b, c, d = plane
    norm = np.sqrt(a * a + b * b + c * c)

    dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
    mask = dist > threshold
    return points[mask]


# ------------------------------------------------------------
# Project points → image space
# ------------------------------------------------------------


def project(points, fx, fy, cx, cy):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    z[z == 0] = 1e-6

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    return u, v


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------


def run(name, base):
    base = Path(base)

    ply_path = base / "pointcloud" / f"{name}.ply"
    info_path = base / "info" / f"{name}.txt"
    out_path = base / "mask" / f"{name}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    if not info_path.exists():
        raise FileNotFoundError(f"INFO not found: {info_path}")

    print(f"[INFO] Loading {ply_path}")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)

    print(f"[INFO] Raw points: {len(points):,}")

    # Parse calibration + plane
    fx, fy, cx, cy, plane, shape = parse_info(info_path)

    if plane is None:
        raise ValueError("Plane equation not found in info.txt")

    # STEP 1: remove conveyor plane
    points = remove_plane(points, plane, threshold=0.01)
    print(f"[INFO] After plane removal: {len(points):,}")

    if len(points) == 0:
        print("[WARN] No points left after filtering")
        return

    # STEP 2: clustering (objects)
    clustering = DBSCAN(eps=0.02, min_samples=30).fit(points)
    labels = clustering.labels_

    # STEP 3: create empty mask
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)

    # STEP 4: project clusters into image space
    unique_labels = set(labels)

    obj_id = 1

    for label in unique_labels:
        if label == -1:
            continue  # noise

        cluster_pts = points[labels == label]

        u, v = project(cluster_pts, fx, fy, cx, cy)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[valid], v[valid]

        mask[v, u] = obj_id
        obj_id += 1

    # Save mask
    cv2.imwrite(str(out_path), mask)

    print(f"[OK] Saved mask → {out_path}")
    print(f"[INFO] Objects found: {obj_id - 1}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="img0000")
    parser.add_argument("--base", default="offline_case/samples")

    args = parser.parse_args()
    run(args.name, args.base)
