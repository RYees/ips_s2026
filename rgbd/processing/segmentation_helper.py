import numpy as np
import open3d as o3d
import cv2


class SegmentationHelper:
    def __init__(
        self,
        intrinsics,
        distance_threshold=0.005,  # tighter plane fit
        ransac_n=3,
        num_iterations=2000,
        plane_offset=0.01,  # how far above plane = object
    ):
        self.intrinsics = intrinsics
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.plane_offset = plane_offset

    def segment(self, depth_image, color_image=None):
        depth_m = depth_image.astype(np.float32) / 1000.0
        h, w = depth_image.shape

        # ---------------------------
        # 1. Depth filtering
        # ---------------------------
        valid_mask = (depth_m > 0.2) & (depth_m < 1.5)
        depth_m = np.where(valid_mask, depth_m, 0)

        depth_o3d = o3d.geometry.Image(depth_m)

        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, self.intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        if len(pcd.points) < self.ransac_n:
            print("[WARNING] Not enough points.")
            return np.zeros((h, w), dtype=np.uint8)

        # ---------------------------
        # 2. RANSAC plane detection
        # ---------------------------
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )

        a, b, c, d = plane_model
        print(f"[INFO] Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # Get NON-plane points
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        points = np.asarray(outlier_cloud.points)

        print(f"[DEBUG] Non-plane points: {len(points)}")

        # ---------------------------
        # 3. Keep ONLY points ABOVE plane
        # ---------------------------
        fx = self.intrinsics.intrinsic_matrix[0][0]
        fy = self.intrinsics.intrinsic_matrix[1][1]
        cx = self.intrinsics.intrinsic_matrix[0][2]
        cy = self.intrinsics.intrinsic_matrix[1][2]

        mask = np.zeros((h, w), dtype=np.uint8)

        kept_points = 0

        for x, y, z in points:
            dist = a * x + b * y + c * z + d

            # 🔥 CORE FILTER
            if dist > self.plane_offset:
                if z <= 0:
                    continue

                u = int((x * fx / z) + cx)
                v = int((y * fy / z) + cy)

                if 0 <= u < w and 0 <= v < h:
                    mask[v, u] = 1
                    kept_points += 1

        print(f"[DEBUG] Points above plane: {kept_points}")

        # ---------------------------
        # 4. Remove noise (largest component only)
        # ---------------------------
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)

        # ---------------------------
        # 5. Morphological cleanup
        # ---------------------------
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask, plane_model, len(inliers), kept_points, pcd
