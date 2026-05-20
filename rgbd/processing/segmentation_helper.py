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

        print(
            "[INIT] Utilizing local OpenCV Contrast Gate Engine. (Network Bypass Active)"
        )

        # Operational margins for conveyor workspace
        self.belt_margin_x_ratio = 0.12
        self.belt_margin_y_ratio = 0.02
        self.min_component_area_ratio = 0.0005

    def _get_contrast_2d_mask(self, color_image):
        """
        Generates a fast visual footprint of the object based on color variance.
        Cardboard boxes stand out starkly against the dark conveyor belt.
        """
        # Convert to grayscale to evaluate structure
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Apply an adaptive threshold to isolate high-contrast objects from the dark belt
        _, binary_gate = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)

        # Clean up stray pixels using a quick morphological closure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned_gate = cv2.morphologyEx(binary_gate, cv2.MORPH_CLOSE, kernel)

        # Convert to a 0/1 mask format
        return (cleaned_gate > 0).astype(np.uint8)

    def segment(self, depth_image, color_image):
        h, w = depth_image.shape

        # 1. Generate the local 2D visual footprint gate
        visual_gate_mask = self._get_contrast_2d_mask(color_image)

        # 2. Build the 3D Point Cloud from Depth Image
        depth_m = depth_image.astype(np.float32) / 1000.0
        valid_mask = (depth_m > 0.15) & (depth_m < 1.5)
        depth_m = np.where(valid_mask, depth_m, 0)

        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, self.intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        if len(pcd.points) < 3:
            return np.zeros((h, w), dtype=np.uint8), [0, 0, 0, 0], 0, 0, None

        # 3. Compute conveyor floor plane via RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        [a, b, c, d] = plane_model
        plane_norm = np.sqrt(a * a + b * b + c * c)

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Real-time Standard Deviation floor noise analysis
        floor_pts = np.asarray(inlier_cloud.points)
        floor_distances = (
            np.abs(a * floor_pts[:, 0] + b * floor_pts[:, 1] + c * floor_pts[:, 2] + d)
            / plane_norm
        )
        sigma = np.std(floor_distances)
        adaptive_cutoff = max(0.005, 3.5 * sigma)
        print(f"[DYNAMIC ENGINE] Floor Noise Cutoff: {adaptive_cutoff * 1000:.2f}mm")

        # 4. Extract 3D Clusters via DBSCAN Spatial Mapping
        final_object_cloud = o3d.geometry.PointCloud()
        final_mask = np.zeros((h, w), dtype=np.uint8)

        if len(outlier_cloud.points) > 5:
            labels = np.array(
                outlier_cloud.cluster_dbscan(
                    eps=0.018, min_points=5, print_progress=False
                )
            )

            if labels.size > 0 and np.any(labels >= 0):
                unique_labels = np.unique(labels[labels >= 0])
                valid_object_indices = []

                fx = self.intrinsics.intrinsic_matrix[0][0]
                fy = self.intrinsics.intrinsic_matrix[1][1]
                cx = self.intrinsics.intrinsic_matrix[0][2]
                cy = self.intrinsics.intrinsic_matrix[1][2]

                for cluster_id in unique_labels:
                    temp_indices = np.where(labels == cluster_id)[0]
                    temp_cloud = outlier_cloud.select_by_index(list(temp_indices))
                    cluster_pts = np.asarray(temp_cloud.points)

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

                    # Project this specific cluster down to verify its location
                    pixels_inside_visual_gate = 0
                    total_projected_pixels = 0

                    for x, y, z in cluster_pts:
                        if z <= 0:
                            continue

                        # 1. Calculate raw projected pixel positions
                        u = int((x * fx / z) + cx)
                        v = int((y * fy / z) + cy)

                        # 2. CRITICAL BOUNDARY GATE:
                        # This outer 'if' statement must completely wrap the mask lookup.
                        # If 'u' is 519 or 581, it skips the inside block entirely and NEVER crashes.
                        if 0 <= u < w and 0 <= v < h:
                            total_projected_pixels += 1

                            # Safely read inside the verified bounds
                            if visual_gate_mask[v, u] > 0:
                                pixels_inside_visual_gate += 1

                    overlap_ratio = (
                        (pixels_inside_visual_gate / total_projected_pixels)
                        if total_projected_pixels > 0
                        else 0
                    )

                    # --- PHYSICAL + VISUAL HYBRID GATE ---
                    # Object must rest on the belt and align with a high-contrast region
                    if (
                        min_dist_to_plane <= adaptive_cutoff
                        and max_height > adaptive_cutoff
                        and overlap_ratio > 0.25
                    ):
                        valid_object_indices.extend(temp_indices)
                        print(
                            f"[PASSED] Cluster #{cluster_id}: {len(cluster_pts)} pts maps perfectly to visual object footprint."
                        )
                    else:
                        print(
                            f"[FILTERED] Cluster #{cluster_id}: {len(cluster_pts)} pts. Base={min_dist_to_plane * 1000:.1f}mm, Overlap={overlap_ratio * 100:.1f}%. Dropped."
                        )

                if len(valid_object_indices) > 0:
                    final_object_cloud = outlier_cloud.select_by_index(
                        valid_object_indices
                    )

        # 5. Project verified object coordinates to final output mask
        points_3d = np.asarray(final_object_cloud.points)
        if len(points_3d) > 0:
            for x, y, z in points_3d:
                if z <= 0:
                    continue
                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)
                if 0 <= u < w and 0 <= v < h:
                    final_mask[v, u] = 1

        final_mask = self._apply_belt_roi(final_mask)
        return (
            final_mask,
            plane_model,
            len(inlier_cloud.points),
            len(final_object_cloud.points),
            final_object_cloud,
        )

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
