import open3d as o3d
import numpy as np
import cv2
import os


def debug_exact_projection(ply_path):
    print("=== COORD PROJECTION & MULTI-OBJECT VALIDATION RADAR ===")
    if not os.path.exists(ply_path):
        print(f"[ERROR] Missing file: {ply_path}")
        return

    # 1. Load the data
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[DATA] Loaded point cloud containing {len(pcd.points)} points.")

# Open 3D viewer
o3d.visualization.draw_geometries([pcd])

    # 2. Extract the conveyor floor plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.012, ransac_n=3, num_iterations=2000
    )
    [a, b, c, d] = plane_model
    plane_norm = np.sqrt(a * a + b * b + c * c)

    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # 3. Cluster using DBSCAN
    labels = np.array(
        outlier_cloud.cluster_dbscan(eps=0.018, min_points=5, print_progress=False)
    )
    if labels.size == 0 or not np.any(labels >= 0):
        print("[ERROR] No dense structures found in outlier space.")
        return

    unique_labels = np.unique(labels[labels >= 0])

    # 4. Filter via Spatial Contact Test (Mirroring our new multi-object solution)
    valid_object_indices = []

    print("\n--- PROCESSING ALL DETECTED 3D CLUSTERS ---")
    for cluster_id in unique_labels:
        temp_indices = np.where(labels == cluster_id)[0]
        temp_cloud = outlier_cloud.select_by_index(list(temp_indices))
        cluster_pts = np.asarray(temp_cloud.points)

        # Distance from cluster bottom to the conveyor surface
        distances = (
            np.abs(
                a * cluster_pts[:, 0]
                + b * cluster_pts[:, 1]
                + c * cluster_pts[:, 2]
                + d
            )
            / plane_norm
        )
        min_dist_to_plane = np.min(distances)
        max_height = np.max(distances)

        if min_dist_to_plane < 0.006 and max_height > 0.0015:
            valid_object_indices.extend(temp_indices)
            print(
                f"[PASSED] Cluster #{cluster_id}: {len(cluster_pts)} pts. Touches belt (Base Dist={min_dist_to_plane * 1000:.1f}mm)."
            )
        else:
            print(
                f"[FILTERED] Cluster #{cluster_id}: {len(cluster_pts)} pts. GHOST NOISE (Base floats at {min_dist_to_plane * 1000:.1f}mm)."
            )

    if len(valid_object_indices) == 0:
        print("[WARNING] Zero clusters passed the spatial contact test.")
        return

    final_validated_cloud = outlier_cloud.select_by_index(valid_object_indices)
    pts = np.asarray(final_validated_cloud.points)
    print(
        f"\n[TOTAL TARGETS] Combined validated physical points to project: {len(pts)}"
    )

    # 5. Simulate 2D Projection with crop displacement
    w, h = 510, 540
    fx = 461.328
    fy = 461.328
    cy = 239.5

    cx_unshifted = 319.5
    cx_shifted = 319.5 - 250.0  # 69.5

    test_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    print("\n--- SIMULATING PROJECTION LOCATIONS FOR VALIDATED OBJECTS ---")
    for cx_val, label in zip(
        [cx_unshifted, cx_shifted],
        ["UNSHIFTED (Standard Matrix)", "SHIFTED (Crop Compensated)"],
    ):
        u_coords = []
        for x, y, z in pts:
            if z <= 0:
                continue
            u = int((x * fx / z) + cx_val)
            v = int((y * fy / z) + cy)
            if 0 <= u < w and 0 <= v < h:
                u_coords.append(u)

        if len(u_coords) > 0:
            print(
                f"[{label}] Points landed inside window: {len(u_coords)} pts. Average column (u) = {int(np.mean(u_coords))}"
            )
        else:
            print(f"[{label}] Out of window bounds!")

    # 6. Generate final diagnostic layout
    for x, y, z in pts:
        if z <= 0:
            continue
        # Draw Unshifted Math in Cyan
        u_un = int((x * fx / z) + cx_unshifted)
        v_un = int((y * fy / z) + cy)
        if 0 <= u_un < w and 0 <= v_un < h:
            test_canvas[v_un, u_un] = [255, 255, 0]

        # Draw Shifted Math in Red
        u_sh = int((x * fx / z) + cx_shifted)
        v_sh = int((y * fy / z) + cy)
        if 0 <= u_sh < w and 0 <= v_sh < h:
            test_canvas[v_sh, u_sh] = [0, 0, 255]

    output_name = "projection_diagnostic_radar.png"
    cv2.imwrite(output_name, test_canvas)
    print(f"\n[SAVED] Diagnostic tracking map written to: {output_name}")


if __name__ == "__main__":
    debug_exact_projection(
        "/home/cpsstudent/Documents/ips_s2026/rgbd/dataset/pointcloud/img0941.ply"
    )
