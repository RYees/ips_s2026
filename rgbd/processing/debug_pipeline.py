import open3d as o3d
import numpy as np
import os


def debug_saved_pointcloud(ply_path, distance_threshold=0.01):
    print("=== STARTING SAVED POINT CLOUD DIAGNOSTIC ===")

    if not os.path.exists(ply_path):
        print(f"[ERROR] Could not find the PLY file at: {ply_path}")
        print("Please check your saved data directory and copy the exact path.")
        return

    # 1. Load the exact 3D point cloud your application saved
    pcd = o3d.io.read_point_cloud(ply_path)
    total_points = len(pcd.points)
    print(f"[DATA] Successfully loaded point cloud.")
    print(f"[DATA] Total 3D Points captured: {total_points}")

    if total_points < 3:
        print("[ERROR] Point cloud has insufficient points to process.")
        return

    # 2. Run your RANSAC Plane Detection
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
    )
    [a, b, c, d] = plane_model

    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    num_inliers = len(inlier_cloud.points)
    num_outliers = len(outlier_cloud.points)

    print(f"\n=== RANSAC GEOMETRY RESULTS ===")
    print(f"[RANSAC] Plane Equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(
        f"[RANSAC] Floor/Plane Inliers (Red): {num_inliers} ({num_inliers / total_points * 100:.2f}%)"
    )
    print(
        f"[RANSAC] Target Object Outliers (Green): {num_outliers} ({num_outliers / total_points * 100:.2f}%)"
    )

    # 3. Test DBSCAN Clustering on the outliers
    print(f"\n=== 3D CLUSTERING TEST ===")
    if num_outliers > 10:
        labels = np.array(
            outlier_cloud.cluster_dbscan(eps=0.02, min_points=30, print_progress=False)
        )
        if labels.size > 0 and np.any(labels >= 0):
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            print(f"[CLUSTER] Found {len(unique_labels)} dense clusters.")
            for cluster_id, count in zip(unique_labels, counts):
                print(f"   -> Cluster #{cluster_id}: contains {count} points")

            largest_cluster_idx = np.argmax(counts)
            print(
                f"[CLUSTER] Largest isolated object cluster is #{largest_cluster_idx} with {counts[largest_cluster_idx]} points."
            )
        else:
            print(
                "[CLUSTER] DBSCAN failed to find any dense clusters. Everything was labeled as noise."
            )
    else:
        print("[CLUSTER] Too few outliers to run clustering analysis.")

    # 4. Launch the Interactive 3D Visualizer
    print(f"\n[INFO] Launching interactive 3D window...")
    print(
        " -> INSTRUCTIONS: Use left click to rotate, right click to pan, scroll wheel to zoom."
    )
    print(" -> Look at it from a perfect horizontal side-view profile.")
    print(
        " -> Are there green points floating ABOVE the red plane? Or are they stuck flat on it?"
    )

    inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red = Floor surface
    outlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green = Detected object data

    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud], window_name="3D Geometry Diagnostic"
    )


if __name__ == "__main__":
    # TODO: Replace this path with the exact location of a saved .ply file
    # that produced a bad mask output (e.g., from image 0931)
    TARGET_PLY_FILE = "/home/cpsstudent/Documents/ips_s2026/rgbd/dataset/pointcloud/img0936.ply"

    debug_saved_pointcloud(TARGET_PLY_FILE, distance_threshold=0.01)
