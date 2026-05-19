import numpy as np
import open3d as o3d
import cv2


class SegmentationHelper:
    def __init__(
        self, intrinsics, distance_threshold=0.01, ransac_n=3, num_iterations=1000
    ):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold  # max distance from a point to the plane to be considered an inlier
        self.ransac_n = ransac_n  # number of points to sample for plane fitting
        self.num_iterations = num_iterations  # number of RANSAC iterations
        self.point_radius = 2
        self.border_margin_ratio = 0.0
        self.belt_margin_x_ratio = 0.12
        self.belt_margin_y_ratio = 0.02
        self.min_component_area_ratio = 0.001
        self.edge_strip_width_ratio = 0.18
        self.edge_strip_height_ratio = 0.35
        self.edge_strip_aspect_ratio = 2.5
        self.residual_threshold = max(0.008, self.distance_threshold * 1.2)

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

    def _clean_mask(self, mask, residual_map, residual_threshold):
        h, w = mask.shape
        if np.count_nonzero(mask) == 0:
            return mask

        # Remove small isolated pixel groups
        min_area = int(h * w * self.min_component_area_ratio)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(cleaned, [cnt], -1, 1, thickness=cv2.FILLED)

        # Clear side artifacts using a mathematical bounding grid profile
        final_mask = cleaned.copy()
        edge_w = int(w * self.edge_strip_width_ratio)
        edge_h = int(h * self.edge_strip_height_ratio)

        for side in ["left", "right"]:
            x_start = 0 if side == "left" else w - edge_w
            x_end = edge_w if side == "left" else w
            roi_strip = final_mask[:, x_start:x_end]

            strip_contours, _ = cv2.findContours(
                roi_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for sc in strip_contours:
                sx, sy, sw, sh = cv2.boundingRect(sc)
                if (
                    sh > 0
                    and (sw / sh) > self.edge_strip_aspect_ratio
                    and sw > (edge_w * 0.4)
                ):
                    if sy < edge_h or sy > (h - edge_h - sh):
                        cv2.drawContours(
                            final_mask[:, x_start:x_end],
                            [sc],
                            -1,
                            0,
                            thickness=cv2.FILLED,
                        )

        return final_mask

    def segment(self, depth_image, color_image):
        depth_m = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters
        h, w = depth_image.shape

        valid_mask = (depth_m > 0.2) & (depth_m < 1.5)
        depth_m = np.where(valid_mask, depth_m, 0)
        print(
            f"[DEBUG] Valid depth pixels after filtering: {np.count_nonzero(depth_m)}"
        )

        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, self.intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        if len(pcd.points) < 3:
            print("[WARNING] Empty point cloud cluster.")
            return np.zeros((h, w), dtype=np.uint8), [0, 0, 0, 0], 0, 0, None

        # 1. Estimate conveyor floor plane structure via RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        [a, b, c, d] = plane_model

        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)

        total_outlier_count = len(outlier_cloud.points)
        print(f"[DEBUG] Plane floor inliers: {len(inlier_cloud.points)}")
        print(f"[DEBUG] Initial raw outliers: {total_outlier_count}")

        # 2. ADAPTIVE 3D CLUSTERING FOR ANY SHAPE/SIZE OBJECT
        clean_object_cloud = o3d.geometry.PointCloud()

        # Low initial seed threshold (8 points) ensures thin objects like steel wires/nails are processed
        if total_outlier_count > 8:
            # Tight connectivity (1.5 cm neighborhood) keeps thin assets distinct from floor reflections
            labels = np.array(
                outlier_cloud.cluster_dbscan(
                    eps=0.015, min_points=8, print_progress=False
                )
            )

            if labels.size > 0 and np.any(labels >= 0):
                unique_labels, counts = np.unique(
                    labels[labels >= 0], return_counts=True
                )

                # Sort clusters by total point counts descending
                sorted_indices = np.argsort(counts)[::-1]
                chosen_cluster_idx = None

                # Examine clusters to isolate structural items from ultra-flat glare shapes
                for idx in sorted_indices:
                    cluster_id = unique_labels[idx]
                    cluster_count = counts[idx]

                    temp_indices = np.where(labels == cluster_id)[0]
                    temp_cloud = outlier_cloud.select_by_index(list(temp_indices))

                    # Evaluate physical height/thickness along the Z axis relative to the belt plane
                    bbox = temp_cloud.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()  # [width, height, depth_thickness]
                    height_thickness = extent[2]

                    print(
                        f"[CLUSTER TRACE] ID #{cluster_id}: {cluster_count} points, Height Extent={height_thickness * 1000:.1f}mm"
                    )

                    # Real assets rising above the floor trigger the thickness test (>2mm).
                    # Substantial clusters fall back to count checks to handle low-profile objects safely.
                    if height_thickness > 0.002 or cluster_count > 100:
                        chosen_cluster_idx = cluster_id
                        break

                # Fallback directly to the largest available group if all appear flat
                if chosen_cluster_idx is None:
                    chosen_cluster_idx = unique_labels[sorted_indices[0]]

                valid_indices = np.where(labels == chosen_cluster_idx)[0]
                clean_object_cloud = outlier_cloud.select_by_index(list(valid_indices))
                print(
                    f"[SUCCESS] Target Identified: Isolated cluster #{chosen_cluster_idx} containing {len(clean_object_cloud.points)} points."
                )
            else:
                print(
                    "[WARNING] No dense 3D structures formed. Preserving default outliers."
                )
                clean_object_cloud = outlier_cloud
        else:
            clean_object_cloud = outlier_cloud

        # 3. Project the dynamically isolated 3D cluster coordinates back onto the 2D pixel mask
        fx = self.intrinsics.intrinsic_matrix[0][0]
        fy = self.intrinsics.intrinsic_matrix[1][1]
        cx = self.intrinsics.intrinsic_matrix[0][2]
        cy = self.intrinsics.intrinsic_matrix[1][2]

        mask = np.zeros((h, w), dtype=np.uint8)
        points_3d = np.asarray(clean_object_cloud.points)

        if len(points_3d) > 0:
            for x, y, z in points_3d:
                if z <= 0:
                    continue
                # Map standard spatial matrices back onto the active cropped coordinate space
                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)

                if 0 <= u < w and 0 <= v < h:
                    mask[v, u] = 1

        print(
            f"[DEBUG] Projected isolated cluster mask contains {np.count_nonzero(mask)} active pixels."
        )

        # 4. Standard post-processing ROI filters and cleaners
        mask = self._apply_belt_roi(mask)

        yy, xx = np.mgrid[0:h, 0:w]
        plane_norm = np.sqrt(a * a + b * b + c * c)
        res_x = (xx - cx) * depth_m / fx
        res_y = (yy - cy) * depth_m / fy
        residual_map = np.abs(a * res_x + b * res_y + c * depth_m + d) / max(
            plane_norm, 1e-8
        )

        mask = self._clean_mask(
            mask, residual_map=residual_map, residual_threshold=self.distance_threshold
        )
        print(
            f"[DEBUG] Mask projection: final cleaned foreground={int(np.count_nonzero(mask))}"
        )

        return (
            mask,
            plane_model,
            len(inlier_cloud.points),
            len(clean_object_cloud.points),
            clean_object_cloud,
        )
