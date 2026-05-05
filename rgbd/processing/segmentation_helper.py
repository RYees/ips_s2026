import numpy as np
import open3d as o3d
import cv2


class SegmentationHelper:
    def __init__(
        self,
        intrinsics,
        distance_threshold=0.01,  # RANSAC inlier threshold
        ransac_n=3,
        num_iterations=2000,
        plane_offset=0.015,  # Minimum height above table to be "object"
    ):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.plane_offset = plane_offset

    def segment(self, depth_image, color_image=None):
        h, w = depth_image.shape
        # 1. Convert Depth to Point Cloud
        depth_m = depth_image.astype(np.float32) / 1000.0
        # Filter range for typical tabletop (0.2m to 1.5m)
        valid_mask = (depth_m > 0.2) & (depth_m < 1.5)
        depth_m_filtered = np.where(valid_mask, depth_m, 0)

        depth_o3d = o3d.geometry.Image(depth_m_filtered)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, self.intrinsics, depth_scale=1.0, depth_trunc=2.0, stride=1
        )

        if len(pcd.points) < self.ransac_n:
            print("[WARNING] Not enough points for RANSAC.")
            return np.zeros((h, w), dtype=np.uint8), None, 0, 0, pcd

        # 2. RANSAC plane detection
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        a, b, c, d = plane_model

        # 3. Vectorized "Above Plane" Filtering
        # Select points NOT in the plane
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        points = np.asarray(outlier_cloud.points)

        if len(points) == 0:
            return np.zeros((h, w), dtype=np.uint8), plane_model, len(inliers), 0, pcd

        # Math: dist = ax + by + cz + d
        # We use dot product for speed
        distances = np.dot(points, [a, b, c]) + d

        # The plane normal (a,b,c) direction is arbitrary.
        # We check both sides. Usually, the object is on the side
        # pointing toward the camera (negative or positive depending on RANSAC)
        pos_mask = distances > self.plane_offset
        neg_mask = distances < -self.plane_offset

        # Pick the side that has fewer points (the object) vs the floor/background
        # Or specifically the side that is closer to the camera (Z decreases)
        # Here we use the positive side as default, but flip if empty
        chosen_indices = np.where(pos_mask)[0]
        if len(chosen_indices) < 10 and np.any(neg_mask):
            chosen_indices = np.where(neg_mask)[0]

        kept_points_count = len(chosen_indices)
        mask = np.zeros((h, w), dtype=np.uint8)

        if kept_points_count > 0:
            pts_to_project = points[chosen_indices]

            # 4. Vectorized Projection to Camera Space
            fx = self.intrinsics.intrinsic_matrix[0][0]
            fy = self.intrinsics.intrinsic_matrix[1][1]
            cx = self.intrinsics.intrinsic_matrix[0][2]
            cy = self.intrinsics.intrinsic_matrix[1][2]

            xs, ys, zs = (
                pts_to_project[:, 0],
                pts_to_project[:, 1],
                pts_to_project[:, 2],
            )

            # Avoid division by zero
            zs = np.where(zs == 0, 0.001, zs)

            u = ((xs * fx / zs) + cx).astype(int)
            v = ((ys * fy / zs) + cy).astype(int)

            # Filter coordinates that fall outside the image frame
            valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            mask[v[valid_uv], u[valid_uv]] = 255  # Use 255 for CV2 compatibility

        # 5. Post-Processing
        # Point cloud projection is "sparse" (dots). We must thicken it.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Keep only the largest object
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            # Stats: [label, [left, top, width, height, area]]
            # We skip index 0 (the background)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8) * 255

        # Final cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask, plane_model, len(inliers), kept_points_count, pcd
