import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
from datetime import datetime
import re
import sys
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from camera.camera_interface import CameraInterface
from processing.annotation_writer import AnnotationWriter
from processing.utils import frame_to_bgr_image
from offline_case.masking import CaptureInfo, run_masking_from_point_cloud
from tkinter import ttk


# Crops the input image to a square region centered in the image, extra_crop pixels are removed from each side.
def crop_manual(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]
    # Enforce non-negative regions
    top = max(0, top)
    bottom = max(0, bottom)
    left = max(0, left)
    right = max(0, right)

    # Prevent invalid crop bounds that would create an empty image
    if top + bottom >= h or left + right >= w:
        return img, 0, 0

    new_img = img[
        top : h - bottom if bottom > 0 else h, left : w - right if right > 0 else w
    ]
    return new_img, left, top


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class RGBDCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB-D Data Collector (Orbbec)")
        self.root.focus_force()

        self.cam = CameraInterface()  # Initialize Orbbec camera interface
        self.cam.setup_streams()  # Sets up the RGB-D streams (color + depth)
        intrinsics = self.cam.get_intrinsics()  # Retrieves intrinsic camera parameters
        print(
            "Width:", intrinsics.width
        )  # The number of pixels horizontally in the image (camera resolution)
        print(
            "Height:", intrinsics.height
        )  # The number of pixels vertically in the image
        fx = intrinsics.intrinsic_matrix[
            0, 0
        ]  # Focal length in x direction (in pixels)
        fy = intrinsics.intrinsic_matrix[
            1, 1
        ]  # Focal length in y direction (in pixels)
        cx = intrinsics.intrinsic_matrix[
            0, 2
        ]  # Optical center x-coordinate (in pixels)
        cy = intrinsics.intrinsic_matrix[
            1, 2
        ]  # Optical center y-coordinate (in pixels)
        print("fx:", fx)
        print("fy:", fy)
        print("cx:", cx)
        print("cy:", cy)
        # self.seg = SegmentationHelper(intrinsics) # Uncomment this if you are not cropping the images
        self.writer = AnnotationWriter()

        # Directories
        base_path = Path("dataset")
        self.img_dir = base_path / "images"
        self.original_rgb_dir = base_path / "original_rgb"
        self.label_dir = base_path / "labels"
        self.depth_dir = base_path / "depth"
        self.mask_dir = base_path / "masks"
        self.info_dir = base_path / "info"
        self.pc_dir = base_path / "pointcloud"
        self.debug_dir = base_path / "debug"
        for d in [
            self.img_dir,
            self.original_rgb_dir,
            self.label_dir,
            self.depth_dir,
            self.mask_dir,
            self.info_dir,
            self.pc_dir,
            self.debug_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = (
            self.debug_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.log_file = open(self.log_path, "a", buffering=1)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TeeStream(self._orig_stdout, self.log_file)
        sys.stderr = TeeStream(self._orig_stderr, self.log_file)
        print(f"[INFO] Session log file: {self.log_path}")

        self.counter = self.next_capture_index()
        self.current_img_name = None
        self.capture_crop = {"top": 0, "bottom": 0, "left": 250, "right": 520}

        self.captured_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.captured_pcd = None
        self.captured_mask_stats = None
        self.captured_plane_model = None
        self.captured_plane_inliers = None
        self.captured_non_plane_pts = None
        self.captured_intrinsics = None
        self.raw_rgb_shape = None
        self.raw_depth_shape = None
        self.cropped_rgb_shape = None
        self.cropped_depth_shape = None
        self.is_capturing = True

        self.video_frame = tk.Frame(root)
        self.video_frame.pack(side=tk.TOP)
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(side=tk.BOTTOM, pady=5)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.class_var = tk.StringVar(value="0")
        tk.Label(self.btn_frame, text="Class:").grid(row=1, column=0)
        self.class_selector = tk.OptionMenu(self.btn_frame, self.class_var, "0", "1")
        self.class_selector.grid(row=1, column=1)
        tk.Label(self.btn_frame, text="(0: Copper, 1: Steel)").grid(
            row=1, column=2, columnspan=2
        )

        self.capture_btn = tk.Button(
            self.btn_frame, text="Capture (Enter)", command=self.capture_frame
        )
        self.capture_btn.grid(row=0, column=0, padx=5)
        self.save_btn = tk.Button(
            self.btn_frame, text="Save (S)", command=self.save_data, state=tk.DISABLED
        )
        self.save_btn.grid(row=0, column=1, padx=5)
        self.retake_btn = tk.Button(
            self.btn_frame,
            text="Retake (R)",
            command=self.retake_frame,
            state=tk.DISABLED,
        )
        self.retake_btn.grid(row=0, column=2, padx=5)
        self.quit_btn = tk.Button(
            self.btn_frame, text="Quit (Q)", command=self.quit_app
        )
        self.quit_btn.grid(row=0, column=3, padx=5)
        self.pcd_btn = tk.Button(
            self.btn_frame,
            text="Preview PointCloud (P)",
            command=self.preview_pointcloud_interactive,
        )
        self.pcd_btn.grid(row=0, column=4, padx=5)

        self.root.bind("<Return>", lambda e: self.capture_frame())
        self.root.bind("s", lambda e: self.save_data())
        self.root.bind("S", lambda e: self.save_data())
        self.root.bind("r", lambda e: self.retake_frame())
        self.root.bind("R", lambda e: self.retake_frame())
        self.root.bind("q", lambda e: self.quit_app())
        self.root.bind("Q", lambda e: self.quit_app())
        self.root.bind("p", lambda e: self.preview_pointcloud_interactive())
        self.root.bind("P", lambda e: self.preview_pointcloud_interactive())

        self.Q()

    def analyze_mask(self, mask):
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = mask_u8.shape
        foreground = int(np.count_nonzero(mask_u8))
        total = int(mask_u8.size)
        ratio = float(foreground / total) if total else 0.0

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        component_areas = (
            stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
        )
        largest_area = int(component_areas.max()) if component_areas.size else 0
        largest_idx = (
            int(np.argmax(component_areas)) + 1 if component_areas.size else -1
        )
        largest_bbox = None
        largest_centroid = None
        if largest_idx > 0:
            x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
            y = int(stats[largest_idx, cv2.CC_STAT_TOP])
            bw = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
            largest_bbox = (x, y, bw, bh)
            largest_centroid = (
                float(centroids[largest_idx][0]),
                float(centroids[largest_idx][1]),
            )

        border_mask = np.zeros_like(mask_u8, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_pixels = int(np.count_nonzero(mask_u8 & border_mask))
        border_ratio = float(border_pixels / foreground) if foreground else 0.0

        return {
            "shape": (h, w),
            "foreground": foreground,
            "ratio": ratio,
            "components": int(num_labels - 1),
            "largest_area": largest_area,
            "largest_bbox": largest_bbox,
            "largest_centroid": largest_centroid,
            "border_pixels": border_pixels,
            "border_ratio": border_ratio,
        }

    def next_capture_index(self):
        pattern = re.compile(r"img(\d+)")
        indices = []
        for directory, suffixes in (
            (self.img_dir, [".png"]),
            (self.original_rgb_dir, [".png"]),
            (self.label_dir, [".txt"]),
            (self.depth_dir, [".png"]),
            (self.mask_dir, [".png"]),
            (self.info_dir, [".txt"]),
            (self.pc_dir, [".ply"]),
            (self.debug_dir, [".png"]),
        ):
            for path in directory.iterdir():
                if path.suffix not in suffixes:
                    continue
                match = pattern.match(path.stem)
                if match:
                    indices.append(int(match.group(1)))
        return (max(indices) + 1) if indices else 0

    def save_mask_debug_overlay(self, img_name, rgb, mask):
        overlay = rgb.copy()
        mask_u8 = self.match_mask_to_image(mask, overlay.shape[:2])
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        overlay = cv2.addWeighted(overlay, 0.82, green, 0.18, 0)
        overlay[mask_u8 > 0] = np.array([0, 255, 0], dtype=np.uint8)

        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(overlay, [largest], -1, (0, 255, 255), 2)

        debug_path = self.debug_dir / f"{img_name}_overlay.png"
        cv2.imwrite(str(debug_path), overlay)
        print(f"[DEBUG] Overlay saved to {debug_path}")

    def match_mask_to_image(self, mask, image_shape):
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.shape == image_shape:
            return mask_u8

        target_h, target_w = image_shape
        print(
            f"[WARNING] Mask/image shape mismatch: mask={mask_u8.shape}, "
            f"image={image_shape}. Resizing mask for overlay/export."
        )
        return cv2.resize(
            mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )

    def print_dataset_counts(self):
        counts = {
            "images": len(list(self.img_dir.glob("*.png"))),
            "depth": len(list(self.depth_dir.glob("*.png"))),
            "labels": len(list(self.label_dir.glob("*.txt"))),
            "masks": len(list(self.mask_dir.glob("*.png"))),
            "pointcloud": len(list(self.pc_dir.glob("*.ply"))),
            "info": len(list(self.info_dir.glob("*.txt"))),
            "debug": len(list(self.debug_dir.glob("*.png"))),
        }
        print(
            "[COUNT] " + ", ".join(f"{name}={value}" for name, value in counts.items())
        )
        return counts

    def update_video(self):
        try:
            if self.is_capturing:
                color_frame, depth_frame = self.cam.get_frames()
                if color_frame is not None and depth_frame is not None:
                    rgb = frame_to_bgr_image(
                        color_frame
                    )  # Convert RGB to BGR for OpenCV
                    preview = cv2.resize(rgb, (960, 540))  # Resize for display
                    img = Image.fromarray(
                        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                    )  # Convert BGR to RGB for PIL
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"[ERROR] update_video failed: {e}")
        self.root.after(30, self.update_video)

    def Q(self):
        self.update_video()

    def capture_frame(self):
        if not self.is_capturing:
            return

        if self.current_img_name is None:
            self.current_img_name = f"img{self.counter:04d}"

        color_frame, depth_frame = self.cam.get_frames()
        if color_frame is None or depth_frame is None:
            print("[ERROR] No frame available to capture")
            return

        rgb = frame_to_bgr_image(color_frame)
        original_rgb = rgb.copy()

        # 1. Safely pull the raw 16-bit depth values
        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth = depth_raw.reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        ).copy()

        self.raw_rgb_shape = rgb.shape
        self.raw_depth_shape = depth.shape

        # 2. Keep your original manual cropping boundaries
        crop = self.capture_crop
        rgb, crop_x, crop_y = crop_manual(
            rgb,
            top=crop["top"],
            bottom=crop["bottom"],
            left=crop["left"],
            right=crop["right"],
        )
        depth, _, _ = crop_manual(
            depth,
            top=crop["top"],
            bottom=crop["bottom"],
            left=crop["left"],
            right=crop["right"],
        )
        self.cropped_rgb_shape = rgb.shape
        self.cropped_depth_shape = depth.shape

        # 3. Adjust the intrinsics to match your cropped frame bounds
        intrinsics = self.cam.get_intrinsics()
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        adjusted_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb.shape[1],
            height=rgb.shape[0],
            fx=fx,
            fy=fy,
            cx=cx - crop_x,
            cy=cy - crop_y,
        )

        # 4. Generate the 3D Point Cloud geometry from clean scaled metrics
        depth_m = depth.astype(np.float32) / 1000.0  # Convert mm to meters
        depth_m = np.where(
            (depth_m > 0.2) & (depth_m < 1.5), depth_m, 0.0
        )  # Filter out noise

        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, adjusted_intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )

        # 5. Run the offline 3D masking pipeline on the live point cloud
        live_info = CaptureInfo(
            raw_depth_shape=self.raw_depth_shape,
            rgb_shape=rgb.shape[:2],
            crop_top=0,
            crop_left=0,
            fx=fx,
            fy=fy,
            cx=adjusted_intrinsics.intrinsic_matrix[0, 2],
            cy=adjusted_intrinsics.intrinsic_matrix[1, 2],
        )
        mask, plane_model, inlier_count, outlier_count = run_masking_from_point_cloud(
            pcd, live_info
        )

        # Save arrays into app memory state exactly as before
        self.captured_rgb = rgb
        self.captured_original_rgb = original_rgb
        self.captured_depth = depth
        self.captured_mask = mask
        self.captured_pcd = pcd
        self.captured_intrinsics = adjusted_intrinsics
        self.captured_plane_model = plane_model
        self.captured_plane_inliers = inlier_count
        self.captured_non_plane_pts = outlier_count
        self.captured_mask_stats = self.analyze_mask(mask)

        # 6. Render the UI Preview Window layouts
        mask_viz = (mask * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            mask_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_bgr, [largest], -1, (0, 255, 0), 2)

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        w, h = 320, 240
        combined = np.hstack(
            (
                cv2.resize(rgb, (w, h)),
                cv2.resize(mask_bgr, (w, h)),
                cv2.resize(depth_colored, (w, h)),
            )
        )

        img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.is_capturing = False
        self.capture_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        self.retake_btn.config(state=tk.NORMAL)

    def save_info_txt(
        self,
        img_name,
        selected_class,
        crop,
        raw_rgb_shape,
        raw_depth_shape,
        cropped_rgb_shape,
        cropped_depth_shape,
        rgb,
        depth,
        intrinsics,
        plane_eq,
        plane_inliers,
        non_plane_pts,
        mask_stats=None,
    ):
        txt_path = self.info_dir / f"{img_name}.txt"
        with open(txt_path, "w") as f:
            f.write("Crop:\n")
            f.write(
                f"  top={crop['top']} bottom={crop['bottom']} left={crop['left']} right={crop['right']}\n"
            )
            f.write(f"Raw RGB shape: {raw_rgb_shape}\n")
            f.write(f"Raw depth shape: {raw_depth_shape}\n")
            f.write(f"Cropped RGB shape: {cropped_rgb_shape}\n")
            f.write(f"Cropped depth shape: {cropped_depth_shape}\n")
            f.write(f"RGB shape: {rgb.shape}\n")
            f.write(f"Depth shape: {depth.shape}\n")
            f.write(f"Depth dtype: {depth.dtype}\n")
            f.write(f"Depth min/max: {depth.min()}/{depth.max()}\n")
            f.write(f"Valid depth pixels: {np.count_nonzero(depth)}\n")
            f.write("Intrinsics matrix:\n")
            for row in intrinsics.intrinsic_matrix:
                f.write("  " + " ".join(f"{val:.6f}" for val in row) + "\n")
            f.write(
                f"Plane equation: {plane_eq[0]:.6f}x + {plane_eq[1]:.6f}y + {plane_eq[2]:.6f}z + {plane_eq[3]:.6f} = 0\n"
            )
            f.write(f"Plane inliers: {plane_inliers}\n")
            f.write(f"Non-plane points: {non_plane_pts}\n")
            f.write(f"Selected class: {selected_class}\n")
            if mask_stats is not None:
                f.write("Mask QA:\n")
                f.write(f"  Foreground pixels: {mask_stats['foreground']}\n")
                f.write(f"  Foreground ratio: {mask_stats['ratio']:.8f}\n")
                f.write(f"  Connected components: {mask_stats['components']}\n")
                f.write(f"  Largest component area: {mask_stats['largest_area']}\n")
                f.write(f"  Border pixels: {mask_stats['border_pixels']}\n")
                f.write(f"  Border ratio: {mask_stats['border_ratio']:.8f}\n")
                f.write(f"  Largest bbox: {mask_stats['largest_bbox']}\n")
                f.write(f"  Largest centroid: {mask_stats['largest_centroid']}\n")
        print(f"[INFO] Info saved to {txt_path}")

    def save_data(self):
        if (
            self.captured_rgb is None
            or self.captured_mask is None
            or self.captured_depth is None
        ):
            print("[WARNING] No complete data frames available to save.")
            return

        img_name = self.current_img_name or f"img{self.counter:04d}"
        print(f"\n[INFO] Saving data packages for: {img_name}...")

        # 1. Save clean RGB image file
        cv2.imwrite(str(self.img_dir / f"{img_name}.png"), self.captured_rgb)
        print(f"[SAVED] RGB image saved to: {self.img_dir / f'{img_name}.png'}")

        # 1b. Save original uncropped RGB image file
        cv2.imwrite(
            str(self.original_rgb_dir / f"{img_name}.png"), self.captured_original_rgb
        )
        print(
            f"[SAVED] Original RGB image saved to: {self.original_rgb_dir / f'{img_name}.png'}"
        )

        # 2. Save Raw 16-bit depth values mapping real metric millimeters
        depth_vis = cv2.normalize(
            self.captured_depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(self.depth_dir / f"{img_name}.png"), depth_colored)
        print(
            f"[SAVED] Colored depth map saved to: {self.depth_dir / f'{img_name}.png'}"
        )

        # 3. Handle YOLO label text formatting
        selected_class = int(self.class_var.get())
        label_mask = self.match_mask_to_image(
            self.captured_mask, self.captured_rgb.shape[:2]
        )
        self.writer.write(
            str(self.label_dir / f"{img_name}.txt"),
            label_mask,
            self.captured_rgb.shape[:2],
            label_class=selected_class,
        )
        print(f"[SAVED] YOLO text file saved to: {self.label_dir / f'{img_name}.txt'}")

        # 4. Save Binary Segmentation Mask file
        cv2.imwrite(str(self.mask_dir / f"{img_name}.png"), label_mask * 255)
        print(
            f"[SAVED] Mask tracking image saved to: {self.mask_dir / f'{img_name}.png'}"
        )

        # 5. Save verified Open3D Point Cloud (.ply spatial model)
        pcd_path = self.pc_dir / f"{img_name}.ply"
        if self.captured_pcd is not None and len(self.captured_pcd.points) > 0:
            o3d.io.write_point_cloud(str(pcd_path), self.captured_pcd)
            print(
                f"[SAVED] Verified 3D Point Cloud saved successfully ({len(self.captured_pcd.points)} points)"
            )
        else:
            print(
                "[ERROR] Cannot save .ply file: tracked application point cloud is empty!"
            )

        if self.captured_plane_model is not None:
            self.save_info_txt(
                img_name,
                selected_class,
                self.capture_crop,
                self.raw_rgb_shape,
                self.raw_depth_shape,
                self.cropped_rgb_shape,
                self.cropped_depth_shape,
                self.captured_rgb,
                self.captured_depth,
                self.captured_intrinsics or self.cam.get_intrinsics(),
                self.captured_plane_model,
                self.captured_plane_inliers,
                self.captured_non_plane_pts,
                mask_stats=self.captured_mask_stats,
            )
        else:
            print("[WARNING] Skipping info.txt save because capture metadata is missing.")

        self.print_dataset_counts()
        self.counter = self.next_capture_index()
        self.current_img_name = None
        self.reset_capture_state()
        print(f"[SUCCESS] Packout cycle {img_name} complete.\n")

    def preview_pointcloud_interactive(self):
        if self.captured_pcd is not None:
            print("[INFO] Launching interactive 3D point cloud viewer...")
            o3d.visualization.draw_geometries([self.captured_pcd])
        else:
            print("[WARNING] No point cloud to preview.")

    def retake_frame(self):
        print("[RETAKE] Retaking frame.")
        self.reset_capture_state()

    def reset_capture_state(self):
        self.captured_rgb = None
        self.captured_original_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.captured_pcd = None
        self.captured_mask_stats = None
        self.captured_plane_model = None
        self.captured_plane_inliers = None
        self.captured_non_plane_pts = None
        self.captured_intrinsics = None
        self.raw_rgb_shape = None
        self.raw_depth_shape = None
        self.cropped_rgb_shape = None
        self.cropped_depth_shape = None
        self.current_img_name = None
        self.is_capturing = True
        self.capture_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.retake_btn.config(state=tk.DISABLED)

    def quit_app(self):
        print("[INFO] Quitting application.")
        self.cam.stop()
        try:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        except Exception:
            pass
        try:
            self.log_file.close()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RGBDCollectorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
