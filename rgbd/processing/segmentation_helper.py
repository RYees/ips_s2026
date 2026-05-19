import numpy as np
import open3d as o3d
import cv2


class SegmentationHelper:
    def __init__(
        self, intrinsics, distance_threshold=0.012, ransac_n=3, num_iterations=2000
    ):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations

        # Operational margins for conveyor workspace
        self.belt_margin_x_ratio = 0.12
        self.belt_margin_y_ratio = 0.02
        self.min_component_area_ratio = 0.0005  # Lowered to protect small nails/wires
        self.edge_strip_width_ratio = 0.15

    def _apply_belt_roi(self, mask):
        h, w = mask.shape
        roi = mask.copy()
        x_margin = max(0, int(w * self.belt_margin_x_ratio))
        y_margin = max(0, int(h * self.belt_margin_y_ratio))
        if x_margin or y_margin:
            roi[:y_margin, :] = 0
            roi[-y_margin:, :] = 0
            roi[:, :x_margin] = 0
            roi[:, -x_margin:] = 0
        return roi

    def _clean_mask(self, mask):
        h, w = mask.shape
        if np.count_nonzero(mask) == 0:
            return mask

        # Retain all valid disjoint structural components
        min_area = int(h * w * self.min_component_area_ratio)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(cleaned, [cnt], -1, 1, thickness=cv2.FILLED)

        return cleaned

    def segment(self, depth_image, color_image):
        h, w = depth_image.shape
        depth_m = depth_image.astype(np.float32) / 1000.0

        valid_mask = (depth_m > 0.15) & (depth_m < 1.5)
        depth_m = np.where(valid_mask, depth_m, 0)

        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, self.intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        if len(pcd.points) < 3:
            return np.zeros((h, w), dtype=np.uint8), [0, 0, 0, 0], 0, 0, None

        # 1. Compute the structural floor baseline via RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        [a, b, c, d] = plane_model
        plane_norm = np.sqrt(a * a + b * b + c * c)

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # 2. Extract clusters via 3D DBSCAN spatial connectivity mapping
        final_object_cloud = o3d.geometry.PointCloud()
        mask = np.zeros((h, w), dtype=np.uint8)

        if len(outlier_cloud.points) > 5:
            labels = np.array(
                outlier_cloud.cluster_dbscan(
                    eps=0.018, min_points=5, print_progress=False
                )
            )

            if labels.size > 0 and np.any(labels >= 0):
                unique_labels = np.unique(labels[labels >= 0])
                valid_object_indices = []

                # Evaluate EVERY cluster found in the scene
                for cluster_id in unique_labels:
                    temp_indices = np.where(labels == cluster_id)[0]
                    temp_cloud = outlier_cloud.select_by_index(list(temp_indices))
                    cluster_pts = np.asarray(temp_cloud.points)

                    # Compute the absolute perpendicular distance of each point to the belt plane
                    # Formula: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
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

                    # --- MULTI-OBJECT CRITERIA ENGINE ---
                    # A cluster is a true physical object if its base makes spatial contact
                    # with the belt surface (min_dist < 6mm) and it has measurable thickness.
                    if min_dist_to_plane < 0.006 and max_height > 0.0015:
                        valid_object_indices.extend(temp_indices)
                        print(
                            f"[TRACKER] Valid Object Confirmed (ID #{cluster_id}): {len(cluster_pts)} pts, Base Dist={min_dist_to_plane * 1000:.1f}mm, Max H={max_height * 1000:.1f}mm"
                        )
                    else:
                        print(
                            f"[REJECTED] Ghost Reflection Filtered (ID #{cluster_id}): Base sits {min_dist_to_plane * 1000:.1f}mm away from belt surface."
                        )

                if len(valid_object_indices) > 0:
                    final_object_cloud = outlier_cloud.select_by_index(
                        valid_object_indices
                    )

        # 3. Project all validated 3D targets onto the 2D mask matrix
        fx = self.intrinsics.intrinsic_matrix[0][0]
        fy = self.intrinsics.intrinsic_matrix[1][1]
        cx = self.intrinsics.intrinsic_matrix[0][2]
        cy = self.intrinsics.intrinsic_matrix[1][2]

        # Force the manual crop displacement correction to align the mask projection
        if cx > 200:
            cx = cx - 250.0

        points_3d = np.asarray(final_object_cloud.points)
        if len(points_3d) > 0:
            for x, y, z in points_3d:
                if z <= 0:
                    continue
                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)
                if 0 <= u < w and 0 <= v < h:
                    mask[v, u] = 1

        # 4. Workspace post-processing and filtering
        mask = self._apply_belt_roi(mask)
        mask = self._clean_mask(mask)

        print(
            f"[DEBUG] Mask Generation Complete. Total objects foreground pixels: {np.count_nonzero(mask)}"
        )
        return (
            mask,
            plane_model,
            len(inlier_cloud.points),
            len(final_object_cloud.points),
            final_object_cloud,
        )
