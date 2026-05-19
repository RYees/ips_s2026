import numpy as np
import open3d as o3d
import cv2

class SegmentationHelper:
    def __init__(self, intrinsics, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold # max distance from a point to the plane to be considered an inlier
        self.ransac_n = ransac_n # number of points to sample for plane fitting
        self.num_iterations = num_iterations # number of RANSAC iterations
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
        print(
            f"[DEBUG] Belt ROI: x_margin={x_margin}, y_margin={y_margin}, "
            f"foreground={int(np.count_nonzero(roi))}"
        )
        return roi

    def _empty_result(self, h, w):
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        empty_pcd = o3d.geometry.PointCloud()
        return empty_mask, (0.0, 0.0, 1.0, 0.0), 0, 0, empty_pcd

    def _clean_mask(self, mask, residual_map=None, residual_threshold=None):
        h, w = mask.shape
        cleaned = (mask > 0).astype(np.uint8)
        raw_foreground = int(np.count_nonzero(cleaned))
        print(f"[DEBUG] Mask cleanup: raw foreground={raw_foreground}")

        close_kernel = np.ones((5, 5), np.uint8)
        open_kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
        after_morph = int(np.count_nonzero(cleaned))
        print(f"[DEBUG] Mask cleanup: after morphology={after_morph}")

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        if num_labels <= 1:
            print("[DEBUG] Mask cleanup: no connected components survived")
            return cleaned

        min_area = max(8, int(h * w * self.min_component_area_ratio))
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        selected_indices = []
        candidate_logs = []

        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            touches_left = x <= 1
            touches_right = x + bw >= w - 2
            touches_border = touches_left or touches_right or y <= 1 or y + bh >= h - 2
            aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))
            strip_like = (
                (touches_left or touches_right)
                and bw <= int(w * self.edge_strip_width_ratio)
                and bh >= int(h * self.edge_strip_height_ratio)
                and aspect_ratio >= self.edge_strip_aspect_ratio
            )
            mean_residual = 0.0
            max_residual = 0.0
            if residual_map is not None:
                comp_residuals = residual_map[labels == idx]
                if comp_residuals.size:
                    mean_residual = float(np.mean(comp_residuals))
                    max_residual = float(np.max(comp_residuals))
            residual_ok = (
                residual_threshold is None
                or mean_residual >= residual_threshold * 1.05
            )
            centroid = np.array(centroids[idx], dtype=np.float32)
            center_dist = float(np.linalg.norm(centroid - center))
            score = float(area) / (1.0 + center_dist)
            if strip_like:
                score *= 0.1
            candidate_logs.append(
                {
                    "idx": idx,
                    "area": area,
                    "bbox": (x, y, bw, bh),
                    "centroid": (float(centroids[idx][0]), float(centroids[idx][1])),
                    "touches_border": touches_border,
                    "strip_like": strip_like,
                    "mean_residual": mean_residual,
                    "max_residual": max_residual,
                    "residual_ok": residual_ok,
                    "score": score,
                }
            )

        if candidate_logs:
            top_candidates = sorted(candidate_logs, key=lambda item: item["score"], reverse=True)[:3]
            for cand in top_candidates:
                print(
                    "[DEBUG] Mask cleanup candidate: "
                    f"idx={cand['idx']} area={cand['area']} "
                    f"bbox={cand['bbox']} centroid=({cand['centroid'][0]:.1f}, {cand['centroid'][1]:.1f}) "
                    f"touches_border={cand['touches_border']} strip_like={cand['strip_like']} "
                    f"mean_residual={cand['mean_residual']:.4f} max_residual={cand['max_residual']:.4f} "
                    f"residual_ok={cand['residual_ok']} "
                    f"score={cand['score']:.4f}"
                )

        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            touches_left = x <= 1
            touches_right = x + bw >= w - 2
            aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))
            strip_like = (
                (touches_left or touches_right)
                and bw <= int(w * self.edge_strip_width_ratio)
                and bh >= int(h * self.edge_strip_height_ratio)
                and aspect_ratio >= self.edge_strip_aspect_ratio
            )
            mean_residual = 0.0
            if residual_map is not None:
                comp_residuals = residual_map[labels == idx]
                if comp_residuals.size:
                    mean_residual = float(np.mean(comp_residuals))
            residual_ok = (
                residual_threshold is None
                or mean_residual >= residual_threshold * 1.05
            )
            if strip_like or not residual_ok:
                continue

            selected_indices.append(idx)

        if not selected_indices:
            non_strip_indices = []
            for idx in range(1, num_labels):
                area = int(stats[idx, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                x = int(stats[idx, cv2.CC_STAT_LEFT])
                bw = int(stats[idx, cv2.CC_STAT_WIDTH])
                touches_left = x <= 1
                touches_right = x + bw >= w - 2
                strip_like = (
                    (touches_left or touches_right)
                    and bw <= int(w * self.edge_strip_width_ratio)
                    and int(stats[idx, cv2.CC_STAT_HEIGHT]) >= int(h * self.edge_strip_height_ratio)
                    and max(bw / max(int(stats[idx, cv2.CC_STAT_HEIGHT]), 1), int(stats[idx, cv2.CC_STAT_HEIGHT]) / max(bw, 1)) >= self.edge_strip_aspect_ratio
                )
                if not strip_like:
                    non_strip_indices.append(idx)
            if non_strip_indices:
                best_idx = max(non_strip_indices, key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))
            else:
                best_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            print(
                f"[DEBUG] Mask cleanup: no component met selection rules, "
                f"falling back to largest component idx={best_idx}"
            )
            selected_indices = [best_idx]

        selected_indices = sorted(set(selected_indices))
        component = np.zeros_like(cleaned)
        for idx in selected_indices:
            component[labels == idx] = 1

        selected_area = int(np.count_nonzero(component))
        print(
            f"[DEBUG] Mask cleanup: connected components={num_labels - 1}, "
            f"min_area={min_area}, selected_indices={selected_indices}, "
            f"selected_area={selected_area}"
        )

        if len(selected_indices) > 1:
            print(
                f"[DEBUG] Mask cleanup: keeping {len(selected_indices)} components before hull"
            )

        refined = np.zeros_like(component)
        for idx in selected_indices:
            single_component = np.zeros_like(component)
            single_component[labels == idx] = 1
            points = np.column_stack(np.where(single_component > 0))
            if points.shape[0] >= 3:
                hull = cv2.convexHull(points[:, ::-1].astype(np.int32))
                hull_mask = np.zeros_like(single_component)
                cv2.fillConvexPoly(hull_mask, hull, 1)
                single_component = hull_mask
            refined = cv2.bitwise_or(refined, single_component)

        component = refined

        dilate_kernel = np.ones((3, 3), np.uint8)
        component = cv2.dilate(component, dilate_kernel, iterations=1)
        final_area = int(np.count_nonzero(component))
        print(f"[DEBUG] Mask cleanup: final area after dilation={final_area}")
        return component.astype(np.uint8)

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
            return self._empty_result(h, w)

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

        yy, xx = np.mgrid[0:h, 0:w]
        z = depth_m
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        plane_norm = np.sqrt(a * a + b * b + c * c)
        residual = np.abs(a * x + b * y + c * z + d) / max(plane_norm, 1e-8)
        adaptive_threshold = self.residual_threshold
        if residual.size:
            valid_residuals = residual[valid_mask]
            if valid_residuals.size:
                median_residual = float(np.median(valid_residuals))
                mad_residual = float(np.median(np.abs(valid_residuals - median_residual)))
                adaptive_threshold = max(
                    self.residual_threshold,
                    median_residual + 4.0 * 1.4826 * mad_residual,
                )
                print(
                    f"[DEBUG] Residual stats: median={median_residual:.4f}, "
                    f"mad={mad_residual:.4f}, adaptive_threshold={adaptive_threshold:.4f}"
                )

        mask = ((z > 0) & valid_mask & (residual > adaptive_threshold)).astype(np.uint8)
        projected_foreground = int(np.count_nonzero(mask))
        print(
            f"[DEBUG] Mask projection: base_threshold={self.residual_threshold:.4f}, "
            f"adaptive_threshold={adaptive_threshold:.4f}, "
            f"projected foreground={projected_foreground}"
        )

        mask = self._apply_belt_roi(mask)

        mask = self._clean_mask(mask, residual_map=residual, residual_threshold=adaptive_threshold)
        print(f"[DEBUG] Mask projection: cleaned foreground={int(np.count_nonzero(mask))}")

        # Visualize
        inlier_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        outlier_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        return mask, plane_model, len(inlier_cloud.points), len(outlier_cloud.points), pcd
