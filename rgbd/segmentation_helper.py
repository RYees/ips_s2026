import numpy as np
import open3d as o3d
import cv2

class SegmentationHelper:
    def __init__(self, intrinsics, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold # max distance from a point to the plane to be considered an inlier
        self.ransac_n = ransac_n # number of points to sample for plane fitting
        self.num_iterations = num_iterations # number of RANSAC iterations

    def segment(self, depth_image, color_image):
        depth_m = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
        h, w = depth_image.shape

        valid_mask = (depth_m > 0.2) & (depth_m < 1.5)
        depth_m = np.where(valid_mask, depth_m, 0) # Removes invalid depth values (either too near or too far) to avoid noise
        print(f"[DEBUG] Valid depth pixels after filtering: {np.count_nonzero(depth_m)}")

        depth_o3d = o3d.geometry.Image(depth_m) # Prepare Open3D RGBD image
        #rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        #rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        #color_o3d = o3d.geometry.Image(rgb_image)

        #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #    color_o3d, depth_o3d,
        #    convert_rgb_to_intensity=False,
        #    depth_scale=1.0,
        #    depth_trunc=3.0
        #)

        #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsics) # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            self.intrinsics,
            depth_scale=1.0,
            depth_trunc=3.0,
            stride=1
        )
        print(f"[DEBUG] Full point cloud has {len(pcd.points)} points.")

        if len(pcd.points) < self.ransac_n: # Skips plane detection if the point cloud has too few points
            print("[WARNING] Not enough points for plane segmentation.")
            return np.zeros((h, w), dtype=np.uint8)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations
        ) # Segment dominant plane using RANSAC - used to find the best model (like a plane, line, etc.) that fits the majority of points, even if some points (outliers) don’t follow the pattern.
        [a, b, c, d] = plane_model
        print(f"[INFO] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        print(f"[DEBUG] Plane inliers: {len(inlier_cloud.points)}")
        print(f"[DEBUG] Non-plane points: {len(outlier_cloud.points)}")

        # Create binary mask from non-plane points
        fx = self.intrinsics.intrinsic_matrix[0][0]
        fy = self.intrinsics.intrinsic_matrix[1][1]
        cx = self.intrinsics.intrinsic_matrix[0][2]
        cy = self.intrinsics.intrinsic_matrix[1][2]

        mask = np.zeros((h, w), dtype=np.uint8) # Generate mask from object points (non-plane points)
        for x, y, z in np.asarray(outlier_cloud.points): # Projects 3D onject points back to 2D pixel coordinates
            if z <= 0:
                continue
            u = int((x * fx / z) + cx)
            v = int((y * fy / z) + cy)
            if 0 <= u < w and 0 <= v < h:
                mask[v, u] = 1 # Marks foreground pixels in a binary mask

        # Visualize
        inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        outlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        return mask, plane_model, len(inlier_cloud.points), len(outlier_cloud.points), pcd


