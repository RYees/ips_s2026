import open3d as o3d
import numpy as np
import os


def analyze_raw_geometry(ply_path):
    print("==========================================================")
    print("      RAW TOF CAMERA GEOMETRY DETAILED ANALYSIS           ")
    print("==========================================================")

    if not os.path.exists(ply_path):
        print(f"[ERROR] Cannot find point cloud file: {ply_path}")
        return

    # 1. Load the raw point cloud file
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    print(f"[DATA] Successfully loaded cloud.")
    print(f"[DATA] Total 3D coordinates captured: {len(pts)}")

    # 2. Extract the conveyor floor plane to locate outliers
    # We use a broad threshold to capture everything sitting on or above the belt
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.012, ransac_n=3, num_iterations=2000
    )

    # Isolate all points that are not part of the flat conveyor surface
    outliers_pcd = pcd.select_by_index(inliers, invert=True)
    print(f"[DATA] Points belonging to floor surface: {len(inliers)}")
    print(
        f"[DATA] Points belonging to outliers (Objects + Noise): {len(outliers_pcd.points)}"
    )

    if len(outliers_pcd.points) < 5:
        print("[ERROR] No outlier points available to analyze.")
        return

    # 3. Cluster every distinct group using a broad DBSCAN connectivity search
    # eps=0.02 (2 cm radius), min_points=5 (include tiny objects like pins/wires)
    labels = np.array(
        outliers_pcd.cluster_dbscan(eps=0.02, min_points=5, print_progress=False)
    )

    unique_labels = np.unique(labels)
    valid_labels = unique_labels[unique_labels >= 0]

    print(
        f"[CLUSTER] Found {len(valid_labels)} distinct 3D shapes in the outlier space.\n"
    )
    print(
        f"{'Cluster ID':<12} | {'Point Count':<12} | {'Width (mm)':<10} | {'Length (mm)':<10} | {'Height (mm)':<10} | {'Type Inference'}"
    )
    print("-" * 85)

    # 4. Loop through each cluster and print its real-world dimensions
    for cluster_id in valid_labels:
        indices = np.where(labels == cluster_id)[0]
        cluster_pcd = outliers_pcd.select_by_index(list(indices))
        cluster_pts = np.asarray(cluster_pcd.points)

        count = len(cluster_pts)

        # Calculate the oriented bounding box around the shape
        try:
            obb = cluster_pcd.get_oriented_bounding_box()
            extent = obb.get_extent()  # Real-world dimensions in meters

            # Sort dimensions from smallest to largest to easily evaluate thickness vs footprint
            dims_mm = sorted(extent * 1000.0)
            thick_h = dims_mm[
                0
            ]  # Minimum extent is almost always vertical height/thickness
            mid_w = dims_mm[1]  # Medium extent
            max_l = dims_mm[2]  # Maximum extent
        except:
            # Fallback if cluster points are linear/coplanar and OBB generation fails
            min_bounds = np.min(cluster_pts, axis=0)
            max_bounds = np.max(cluster_pts, axis=0)
            diff = (max_bounds - min_bounds) * 1000.0
            dims_mm = sorted(diff)
            thick_h, mid_w, max_l = dims_mm[0], dims_mm[1], dims_mm[2]

        # Geometric Type Inference Engine based on your objective
        if count > 15000:
            inferred_type = "Massive 3D Volume Object (Box)"
        elif thick_h < 3.0 and count > 1000:
            inferred_type = "Flat Surface Reflection / Glare Noise"
        elif thick_h >= 2.0 and max_l < 60.0:
            inferred_type = "Small 3D Object (Nail / Component)"
        elif max_l > 40.0 and mid_w < 6.0:
            inferred_type = "Thin Linear Structure (Steel Wire)"
        else:
            inferred_type = "Scatter Artifact / Noise Dust"

        print(
            f"Cluster #{cluster_id:<4} | {count:<12} | {mid_w:<10.1f} | {max_l:<10.1f} | {thick_h:<10.1f} | {inferred_type}"
        )

    print("==========================================================")


if __name__ == "__main__":
    # Point directly to your captured PLY data file
    analyze_raw_geometry(
        "/home/cpsstudent/Documents/ips_s2026/rgbd/dataset/pointcloud/img0941.ply"
    )
