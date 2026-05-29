import numpy as np
import open3d as o3d
import cv2
from pathlib import Path


# ------------------------------------------------------------
# LOAD POINT CLOUD
# ------------------------------------------------------------


def load_pcd(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points)


# ------------------------------------------------------------
# RANSAC PLANE REMOVAL
# ------------------------------------------------------------


def remove_plane(points, distance_threshold=0.005):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
    )

    outliers = pcd.select_by_index(inliers, invert=True)
    return np.asarray(outliers.points), plane_model


# ------------------------------------------------------------
# DBSCAN CLUSTERING
# ------------------------------------------------------------


def cluster_points(points, eps=0.01, min_points=200):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    clusters = []
    for i in range(labels.max() + 1):
        c = points[labels == i]
        if len(c) > 0:
            clusters.append(c)

    return clusters


# ------------------------------------------------------------
# SCORING (REAL 3D PHYSICS)
# ------------------------------------------------------------


def score_cluster(cluster):
    z = cluster[:, 2]
    height = np.mean(z)

    mins = np.min(cluster, axis=0)
    maxs = np.max(cluster, axis=0)
    dims = maxs - mins
    volume = dims[0] * dims[1] * dims[2]

    bbox = 1.0 / (volume + 1e-6)

    center = np.mean(cluster, axis=0)
    dist = np.linalg.norm(cluster - center, axis=1)
    compact = 1.0 / (np.mean(dist) + 1e-6)

    density = len(cluster) / (volume + 1e-6)

    score = 0.40 * height + 0.20 * bbox + 0.25 * compact + 0.15 * density

    return score


# ------------------------------------------------------------
# PROJECT 3D CLUSTER → 2D MASK
# ------------------------------------------------------------


def project_to_mask(cluster, fx, fy, cx, cy, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)

    pts = cluster
    valid = pts[:, 2] > 0
    pts = pts[valid]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[inside], v[inside]

    mask[v, u] = 255
    return mask


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------


def detect_objects(ply_path, info):
    print(f"[INFO] Loading {ply_path}")

    points = load_pcd(ply_path)
    print(f"[INFO] Raw points: {len(points):,}")

    points, plane = remove_plane(points)
    print(f"[INFO] After plane removal: {len(points):,}")

    clusters = cluster_points(points)
    print(f"[INFO] Clusters found: {len(clusters)}")

    results = []

    for i, c in enumerate(clusters):
        s = score_cluster(c)
        results.append((i + 1, s, c))

    results.sort(key=lambda x: x[1], reverse=True)

    best_id, best_score, best_cluster = results[0]

    print("\n[RESULTS]")
    for i, s, _ in results:
        print(f"Cluster {i}: score = {s:.4f}")

    print(f"\n[FINAL OBJECT] Cluster {best_id}")

    # --------------------------------------------------------
    # CREATE MASK FROM BEST CLUSTER
    # --------------------------------------------------------

    H, W = info["shape"]
    fx, fy = info["fx"], info["fy"]
    cx, cy = info["cx"], info["cy"]

    mask = project_to_mask(best_cluster, fx, fy, cx, cy, H, W)

    return results, best_id, best_cluster, mask
