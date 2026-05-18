import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import open3d as o3d

from camera.camera_interface import CameraInterface
from processing.segmentation_helper import SegmentationHelper
from processing.annotation_writer import AnnotationWriter
from processing.utils import frame_to_bgr_image
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
        self.label_dir = base_path / "labels"
        self.depth_dir = base_path / "depth"
        self.mask_dir = base_path / "masks"
        self.info_dir = base_path / "info"
        self.pc_dir = base_path / "pointcloud"
        self.debug_dir = base_path / "debug"
        for d in [
            self.img_dir,
            self.label_dir,
            self.depth_dir,
            self.mask_dir,
            self.info_dir,
            self.pc_dir,
            self.debug_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.counter = len(list(self.img_dir.glob("*.png")))

        self.captured_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.captured_pcd = None
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
        component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
        largest_area = int(component_areas.max()) if component_areas.size else 0
        largest_idx = int(np.argmax(component_areas)) + 1 if component_areas.size else -1
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

    def save_mask_debug_overlay(self, img_name, rgb, mask):
        overlay = rgb.copy()
        mask_u8 = self.match_mask_to_image(mask, overlay.shape[:2])
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        overlay = cv2.addWeighted(overlay, 0.82, green, 0.18, 0)
        overlay[mask_u8 > 0] = np.array([0, 255, 0], dtype=np.uint8)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        return cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

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
            "[COUNT] "
            + ", ".join(f"{name}={value}" for name, value in counts.items())
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

        color_frame, depth_frame = self.cam.get_frames()
        if color_frame is None or depth_frame is None:
            print("[ERROR] No frame available to capture")
            return

        rgb = frame_to_bgr_image(color_frame)  # Convert RGB to BGR for OpenCV
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        )  # convert depth frame to numpy array
        depth_height, depth_width = depth.shape
        # rgb = cv2.resize(rgb, (depth_width, depth_height)) # Resize RGB to match depth dimensions
        print(f"****[DEBUG] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

        # === Crop both to center square ===
        # === Apply crop to all sides ===
        rgb, crop_x, crop_y = crop_manual(rgb, top=0, bottom=0, left=250, right=520)
        depth, _, _ = crop_manual(depth, top=0, bottom=0, left=250, right=520)

        # === Get original intrinsics ===
        intrinsics = (
            self.cam.get_intrinsics()
        )  # Recalculate intrinsics based on the cropped image
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        # === Build adjusted intrinsics ===
        adjusted_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=rgb.shape[1],  # new square width
            height=rgb.shape[0],  # new square height
            fx=fx,
            fy=fy,
            cx=cx - crop_x,
            cy=cy - crop_y,
        )

        print(f"[DEBUG] RGB shape: {rgb.shape}")  # Shape: (height, width, 3)
        print(f"[DEBUG] Depth shape: {depth.shape}")
        print(f"[DEBUG] Depth dtype: {depth.dtype}")
        print(f"[DEBUG] Depth min/max: {np.min(depth)}, {np.max(depth)}")
        print(f"[DEBUG] Valid depth pixels: {np.count_nonzero(depth)}")

        # === Run segmentation on cropped data ===
        self.seg = SegmentationHelper(adjusted_intrinsics)
        mask, plane_eq, plane_inliers, non_plane_pts, pcd = self.seg.segment(
            depth, rgb
        )  # binary mask, plane equation, inliers count, non-plane points count, and point cloud

        self.captured_rgb = rgb
        self.captured_depth = depth
        self.captured_mask = mask
        self.captured_pcd = pcd
        mask_stats = self.analyze_mask(mask)
        self.save_mask_debug_overlay(img_name := f"img{self.counter:04d}", rgb, mask)
        print(
            "[MASK-QA] "
            f"shape={mask_stats['shape']} "
            f"foreground={mask_stats['foreground']} "
            f"ratio={mask_stats['ratio']:.6f} "
            f"components={mask_stats['components']} "
            f"largest_area={mask_stats['largest_area']} "
            f"border_pixels={mask_stats['border_pixels']} "
            f"border_ratio={mask_stats['border_ratio']:.6f}"
        )

        intrinsics = self.cam.get_intrinsics()
        self.save_info_txt(
            img_name,
            rgb,
            depth,
            adjusted_intrinsics,
            plane_eq,
            plane_inliers,
            non_plane_pts,
            mask_stats,
        )  # Change adjusted_intrinsics to intrinsics if you are not cropping the images

        mask_viz = (mask * 255).astype(
            np.uint8
        )  # Convert binary mask to uint8 for visualization
        mask_bgr = cv2.cvtColor(
            mask_viz, cv2.COLOR_GRAY2BGR
        )  # Convert to BGR for OpenCV visualization
        contours, _ = cv2.findContours(
            mask_viz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # Find contours in the mask and outline the largest one
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_bgr, [largest], -1, (0, 255, 0), 2)

        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )  # Normalize depth for visualization
        depth_colored = cv2.applyColorMap(
            depth_vis, cv2.COLORMAP_JET
        )  # Apply color map to depth image

        w, h = 320, 240
        combined = np.hstack(
            (
                cv2.resize(rgb, (w, h)),
                cv2.resize(mask_bgr, (w, h)),
                cv2.resize(depth_colored, (w, h)),
            )
        )  # Combine RGB, mask, and depth images horizontally

        img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.is_capturing = False
        self.capture_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL)
        self.retake_btn.config(state=tk.NORMAL)
        print(f"[INFO] Frame captured for class: {self.class_var.get()}")

    def save_info_txt(
        self,
        img_name,
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
        if self.captured_rgb is None or self.captured_mask is None:
            print("[WARNING] No frame to save.")
            return

        img_name = f"img{self.counter:04d}"
        cv2.imwrite(str(self.img_dir / f"{img_name}.png"), self.captured_rgb)
        depth_vis = cv2.normalize(
            self.captured_depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(self.depth_dir / f"{img_name}.png"), depth_colored)

        selected_class = int(self.class_var.get())
        label_mask = self.match_mask_to_image(self.captured_mask, self.captured_rgb.shape[:2])
        label_written = self.writer.write(
            str(self.label_dir / f"{img_name}.txt"),
            label_mask,
            self.captured_rgb.shape[:2],
            label_class=selected_class,
        )
        if not label_written:
            print(
                f"[WARNING] No valid contour was written for {img_name}; "
                "check the saved overlay and mask QA logs."
            )

        mask_path = self.mask_dir / f"{img_name}.png"
        cv2.imwrite(str(mask_path), label_mask * 255)
        print(f"[SAVED] Mask image saved to {mask_path}")

        pcd_path = self.pc_dir / f"{img_name}.ply"
        o3d.io.write_point_cloud(str(pcd_path), self.captured_pcd)
        print(f"[SAVED] Point cloud saved to {pcd_path}")
        print(f"[SAVED] {img_name}")

        self.print_dataset_counts()

        self.counter += 1
        self.reset_capture_state()

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
        self.captured_depth = None
        self.captured_mask = None
        self.captured_pcd = None
        self.is_capturing = True
        self.capture_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.retake_btn.config(state=tk.DISABLED)

    def quit_app(self):
        print("[INFO] Quitting application.")
        self.cam.stop()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RGBDCollectorApp(root)
        root.mainloop()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
