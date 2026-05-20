import numpy as np
import open3d as o3d
import cv2
from ultralytics import SAM


class SegmentationHelper:
    def __init__(
        self, intrinsics, distance_threshold=0.012, ransac_n=3, num_iterations=2000
    ):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations

        # Load the lightweight MobileSAM model
        print("[INIT] Loading MobileSAM Foundation Model...")
        self.sam_model = SAM("mobile_sam.pt")

        # Operational margins for conveyor workspace
        self.belt_margin_x_ratio = 0.12
        self.belt_margin_y_ratio = 0.02

    def _get_sam_2d_mask(self, color_image):
        """
        Uses MobileSAM to extract the pure visual boundary of the box on the belt,
        completely ignoring the 3D infrared glare artifacts.
        """
        h, w, _ = color_image.shape
        # Run zero-shot prompt-free instance segmentation on the 2D frame
        results = self.sam_model(color_image, verbose=False)

        sam_mask = np.zeros((h, w), dtype=np.uint8)
        if results and len(results[0].masks) > 0:
            # Sort masks by area to find the primary object on the belt (the box)
            masks_data = results[0].masks.data.cpu().numpy()
            mask_areas = [np.sum(m) for m in masks_data]
            largest_mask_idx = np.argmax(mask_areas)

            # Convert boolean mask to binary uint8 matrix
            sam_mask = (masks_data[largest_mask_idx] > 0.5).astype(np.uint8)
            print(
                f"[SAM 2D] Isolated visual object mask footprint containing {np.sum(sam_mask)} pixels."
            )
        else:
            print("[SAM 2D] WARNING: No visual objects detected in color stream.")
        return sam_mask

    def segment(self, depth_image, color_image):
        h, w = depth_image.shape

        # 1. Generate the 2D visual object mask gatekeeper using MobileSAM
        sam_gatekeeper_mask = self._get_sam_2d_mask(color_image)

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

        # Calculate standard deviation noise profile of the floor in real-time
        floor_pts = np.asarray(inlier_cloud.points)
        floor_distances = (
            np.abs(a * floor_pts[:, 0] + b * floor_pts[:, 1] + c * floor_pts[:, 2] + d)
            / plane_norm
        )
        sigma = np.std(floor_distances)
        adaptive_cutoff = max(0.005, 3.5 * sigma)
        print(
            f"[DYNAMIC ENGINE] Floor Noise Cutoff set to: {adaptive_cutoff * 1000:.2f}mm"
        )

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

                # Camera intrinsic centers for mapping validation
                fx = self.intrinsics.intrinsic_matrix[0][0]
                fy = self.intrinsics.intrinsic_matrix[1][1]
                cx = self.intrinsics.intrinsic_matrix[0][2]
                cy = self.intrinsics.intrinsic_matrix[1][2]

                # Process and cross-reference every individual 3D cluster found
                for cluster_id in unique_labels:
                    temp_indices = np.where(labels == cluster_id)[0]
                    temp_cloud = outlier_cloud.select_by_index(list(temp_indices))
                    cluster_pts = np.asarray(temp_cloud.points)

                    # Compute basic spatial parameters
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

                    # Project this specific cluster's points into 2D pixel space to see where it lands
                    pixels_inside_sam_gate = 0
                    total_projected_pixels = 0

                    for x, y, z in cluster_pts:
                        if z <= 0:
                            continue
                        u = int((x * fx / z) + cx)
                        v = int((y * fy / z) + cy)
                        if 0 <= u < w and 0 <= v < h:
                            total_projected_pixels += 1
                            # Check if this 3D point maps inside SAM's 2D object mask
                            if sam_gatekeeper_mask[v, u] > 0:
                                pixels_inside_sam_gate += 1

                    # Calculate alignment ratio
                    gate_overlap_ratio = (
                        (pixels_inside_sam_gate / total_projected_pixels)
                        if total_projected_pixels > 0
                        else 0
                    )

                    # --- COMPLEMENTARY FUSION RULE ---
                    # To pass, a cluster must rest near the belt (min_dist <= cutoff)
                    # AND its projected 2D coordinates must match SAM's visual object footprint (> 30% overlap)
                    if (
                        min_dist_to_plane <= adaptive_cutoff
                        and gate_overlap_ratio > 0.30
                    ):
                        valid_object_indices.extend(temp_indices)
                        print(
                            f"[FUSION PASSED] Cluster #{cluster_id}: {len(cluster_pts)} pts. Overlaps 2D SAM Mask by {gate_overlap_ratio * 100:.1f}%. Real Object Verified."
                        )
                    else:
                        print(
                            f"[FUSION REJECTED] Cluster #{cluster_id}: {len(cluster_pts)} pts. Overlaps SAM Mask by {gate_overlap_ratio * 100:.1f}%. Flagged as Ghost Noise."
                        )

                if len(valid_object_indices) > 0:
                    final_object_cloud = outlier_cloud.select_by_index(
                        valid_object_indices
                    )

        # 5. Project the verified 3D object points to the final output mask
        points_3d = np.asarray(final_object_cloud.points)
        if len(points_3d) > 0:
            fx = self.intrinsics.intrinsic_matrix[0][0]
            fy = self.intrinsics.intrinsic_matrix[1][1]
            cx = self.intrinsics.intrinsic_matrix[0][2]
            cy = self.intrinsics.intrinsic_matrix[1][2]

            for x, y, z in points_3d:
                if z <= 0:
                    continue
                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)
                if 0 <= u < w and 0 <= v < h:
                    final_mask[v, u] = 1

        # Clean borders and fill tiny holes
        final_mask = self._clean_mask(final_mask)
        return (
            final_mask,
            plane_model,
            len(inlier_cloud.points),
            len(final_object_cloud.points),
            final_object_cloud,
        )

    def _clean_mask(self, mask):
        # Quick close operation to merge individual projected 3D pixels into a solid mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
