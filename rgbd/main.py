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


# Crops the input image to a region; extra_crop pixels are removed from each side.
def crop_manual(img, top=0, bottom=0, left=0, right=0):
    h, w = img.shape[:2]
    top = max(0, top)
    bottom = max(0, bottom)
    left = max(0, left)
    right = max(0, right)

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
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


class RGBDCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Sorting Rig Control Panel")
        self.root.geometry("1400x900")
        self.root.focus_force()

        # Core Hardware Sub-System Initialization
        self.cam = CameraInterface()
        self.cam.setup_streams()
        intrinsics = self.cam.get_intrinsics()

        print("Width:", intrinsics.width)
        print("Height:", intrinsics.height)
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        self.writer = AnnotationWriter()

        # Dataset Storage Layout Routing
        base_path = Path("dataset")
        self.img_dir = base_path / "images"
        self.crop_rgb_dir = base_path / "cropped_rgb"
        self.label_dir = base_path / "labels"
        self.depth_dir = base_path / "depth"
        self.mask_dir = base_path / "masks"
        self.info_dir = base_path / "info"
        self.pc_dir = base_path / "pointcloud"
        self.debug_dir = base_path / "debug"
        for d in [
            self.img_dir,
            self.crop_rgb_dir,
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

        self.counter = self.next_capture_index()
        self.current_img_name = None

        # Application State Tracking Buffers
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
        self.is_capturing = True

        # Build Layout Panelling with Sliders
        self.setup_ui_layout()

        # Keyboard Shortcuts
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

    def setup_ui_layout(self):
        # Dual Screen Panel Spacing Layout
        self.left_panel = tk.Frame(self.root, width=950, bg="#2b2b2b")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_panel = tk.Frame(
            self.root, width=450, bg="#3c3f41", bd=2, relief=tk.SUNKEN
        )
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Video Label
        self.video_label = tk.Label(self.left_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Controllers Frame
        self.controls_frame = tk.LabelFrame(
            self.right_panel,
            text=" Execution Pipeline Controllers ",
            bg="#3c3f41",
            fg="white",
        )
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self.capture_btn = tk.Button(
            self.controls_frame,
            text="Capture Frame (Enter)",
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="black",
            command=self.capture_frame,
        )
        self.capture_btn.pack(fill=tk.X, padx=10, pady=5)

        self.retake_btn = tk.Button(
            self.controls_frame,
            text="Retake Frame (R)",
            font=("Arial", 11),
            state=tk.DISABLED,
            command=self.retake_frame,
        )
        self.retake_btn.pack(fill=tk.X, padx=10, pady=5)

        # Class Label Selection Block
        self.class_frame = tk.LabelFrame(
            self.right_panel,
            text=" Quality Assurance Label Classification ",
            bg="#3c3f41",
            fg="white",
        )
        self.class_frame.pack(fill=tk.X, padx=10, pady=10)

        self.class_var = tk.StringVar(value="0")
        tk.Radiobutton(
            self.class_frame,
            text="Class 00: Copper Mass Asset",
            variable=self.class_var,
            value="0",
            bg="#3c3f41",
            fg="white",
            selectcolor="#2b2b2b",
        ).pack(anchor=tk.W, padx=10, pady=2)
        tk.Radiobutton(
            self.class_frame,
            text="Class 01: Steel Structural Scrap",
            variable=self.class_var,
            value="1",
            bg="#3c3f41",
            fg="white",
            selectcolor="#2b2b2b",
        ).pack(anchor=tk.W, padx=10, pady=2)

        self.save_btn = tk.Button(
            self.right_panel,
            text="COMMIT DATASET SNAPSHOT (S)",
            font=("Arial", 12, "bold"),
            bg="#008CBA",
            fg="black",
            state=tk.DISABLED,
            command=self.save_data,
        )
        self.save_btn.pack(fill=tk.X, padx=20, pady=20)

        # Modern Slider Adjustments Frame
        self.crop_frame = tk.LabelFrame(
            self.right_panel,
            text=" Realtime Hardware Crop Adjustments (px) ",
            bg="#3c3f41",
            fg="white",
        )
        self.crop_frame.pack(fill=tk.X, padx=10, pady=10)

        self.crop_top_var = tk.IntVar(value=0)
        self.crop_bottom_var = tk.IntVar(value=0)
        self.crop_left_var = tk.IntVar(value=250)
        self.crop_right_var = tk.IntVar(value=520)

        self.create_slider(self.crop_frame, "Top Margin:", self.crop_top_var, 0, 500)
        self.create_slider(
            self.crop_frame, "Bottom Margin:", self.crop_bottom_var, 0, 500
        )
        self.create_slider(self.crop_frame, "Left Margin:", self.crop_left_var, 0, 500)
        self.create_slider(
            self.crop_frame, "Right Margin:", self.crop_right_var, 0, 600
        )

        self.pcd_btn = tk.Button(
            self.right_panel,
            text="Preview PointCloud (P)",
            command=self.preview_pointcloud_interactive,
        )
        self.pcd_btn.pack(fill=tk.X, padx=20, pady=5)

        self.quit_btn = tk.Button(
            self.right_panel,
            text="Shutdown Subsystem Enclosure (Q)",
            command=self.quit_app,
            bg="#f44336",
            fg="black",
        )
        self.quit_btn.pack(fill=tk.X, padx=20, pady=5)

    def create_slider(self, parent, label_text, var, from_, to):
        f = tk.Frame(parent, bg="#3c3f41")
        f.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(
            f, text=label_text, width=12, anchor=tk.W, bg="#3c3f41", fg="white"
        ).pack(side=tk.LEFT)
        s = ttk.Scale(f, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True)
        l = tk.Label(f, text=str(var.get()), width=4, bg="#3c3f41", fg="white")
        l.pack(side=tk.RIGHT)
        var.trace_add("write", lambda *args: l.config(text=str(var.get())))

    def update_video(self):
        try:
            if self.is_capturing:
                frames = self.cam.get_frames()
                if frames is not None and frames[0] is not None:
                    color_frame = frames[0]
                    rgb = frame_to_bgr_image(color_frame)

                    t = self.crop_top_var.get()
                    b = self.crop_bottom_var.get()
                    l = self.crop_left_var.get()
                    r = self.crop_right_var.get()

                    img_cropped, _, _ = crop_manual(
                        rgb, top=t, bottom=b, left=l, right=r
                    )

                    h, w = img_cropped.shape[:2]
                    max_h, max_w = 700, 930
                    scale = min(max_w / w, max_h / h, 1.0)
                    if scale < 1.0:
                        img_cropped = cv2.resize(
                            img_cropped,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    img = Image.fromarray(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
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

        frames = self.cam.get_frames()
        if frames is None or frames[0] is None or frames[1] is None:
            print("[ERROR] No frame available to capture")
            return

        color_frame, depth_frame = frames[0], frames[1]
        rgb = frame_to_bgr_image(color_frame)
        original_rgb = rgb.copy()

        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth = depth_raw.reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        ).copy()

        self.raw_rgb_shape = rgb.shape
        self.raw_depth_shape = depth.shape

        # Get values from sliders
        t = self.crop_top_var.get()
        b = self.crop_bottom_var.get()
        l = self.crop_left_var.get()
        r = self.crop_right_var.get()

        # 1. Crop RGB image normally
        rgb, crop_x, crop_y = crop_manual(rgb, top=t, bottom=b, left=l, right=r)

        # ─────────────────────────────────────────────────────────────────────
        # CRITICAL RESOLUTION-PROPORTIONAL DEPTH CROPPING
        # ─────────────────────────────────────────────────────────────────────
        scale_x = depth.shape[1] / original_rgb.shape[1]  # e.g., 640 / 1280 = 0.5
        scale_y = depth.shape[0] / original_rgb.shape[0]  # e.g., 576 / 720 = 0.8

        depth_t = int(round(t * scale_y))
        depth_b = int(round(b * scale_y))
        depth_l = int(round(l * scale_x))
        depth_r = int(round(r * scale_x))

        depth, d_crop_x, d_crop_y = crop_manual(
            depth, top=depth_t, bottom=depth_b, left=depth_l, right=depth_r
        )
        # ─────────────────────────────────────────────────────────────────────

        self.cropped_rgb_shape = rgb.shape
        self.cropped_depth_shape = depth.shape

        # Intrinsics handling
        intrinsics = self.cam.get_intrinsics()
        fx = intrinsics.intrinsic_matrix[0, 0]
        fy = intrinsics.intrinsic_matrix[1, 1]
        cx = intrinsics.intrinsic_matrix[0, 2]
        cy = intrinsics.intrinsic_matrix[1, 2]

        # ─────────────────────────────────────────────────────────────────────
        # FIXED: RESOLUTION-UNIFIED DEPTH INTRINSICS GENERATION
        # ─────────────────────────────────────────────────────────────────────
        # The intrinsic pinhole projection must strictly match the spatial
        # width and height of the depth matrix passed to Open3D.
        adjusted_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=depth.shape[1],  # Fixed to depth width (e.g. 640)
            height=depth.shape[0],  # Fixed to depth height (e.g. 576)
            fx=fx * scale_x,  # Scale fx to depth plane metrics
            fy=fy * scale_y,  # Scale fy to depth plane metrics
            cx=(cx * scale_x) - depth_l,  # Align cx relative to depth crop boundaries
            cy=(cy * scale_y) - depth_t,  # Align cy relative to depth crop boundaries
        )

        # Generate 3D Point Cloud geometries safely using depth scaling spaces
        depth_m = depth.astype(np.float32) / 1000.0
        depth_m = np.where((depth_m > 0.2) & (depth_m < 1.5), depth_m, 0.0)

        depth_o3d = o3d.geometry.Image(depth_m)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d, adjusted_intrinsics, depth_scale=1.0, depth_trunc=3.0, stride=1
        )
        # ─────────────────────────────────────────────────────────────────────

        # Map correct tracking telemetry variables down to the masking algorithm info block
        live_info = CaptureInfo(
            raw_depth_shape=self.raw_depth_shape,
            rgb_shape=rgb.shape[:2],
            crop_top=crop_y,
            crop_left=crop_x,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        mask, plane_model, inlier_count, outlier_count = run_masking_from_point_cloud(
            pcd, live_info
        )

        # ─────────────────────────────────────────────────────────────────────
        # REALTIME LIVE INTRINSICS MATRIX DIAGNOSTIC REPORT REINTEGRATION
        # ─────────────────────────────────────────────────────────────────────
        log_lines = [
            "\n" + "=" * 80,
            "             REALTIME LIVE INTRINSICS MATRIX DIAGNOSTIC REPORT",
            "=" * 80,
            "  Timestamp Code Generation : "
            + datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            f"  [HARDWARE] Raw Color Sensor Matrix Shape : {original_rgb.shape[1]}W x {original_rgb.shape[0]}H px",
            f"  [HARDWARE] Raw Depth Sensor Matrix Shape : {self.raw_depth_shape[1]}W x {self.raw_depth_shape[0]}H px",
            f"  [SLIDERS]  Active Crop Offsets Input     : Top={t}px, Bottom={b}px, Left={l}px, Right={r}px",
            f"  [PIPELINE] Cropped App Canvas Boundary  : {rgb.shape[1]}W x {rgb.shape[0]}H px",
            f"  [MATRIX]   Intrinsic fx (Focal X)        : {fx:.6f}",
            f"  [MATRIX]   Intrinsic fy (Focal Y)        : {fy:.6f}",
            f"  [MATRIX]   Intrinsic cx (Center X)       : {cx:.6f}",
            f"  [MATRIX]   Intrinsic cy (Center Y)       : {cy:.6f}",
        ]

        visual_cx = rgb.shape[1] / 2
        shifted_math_cx = cx - crop_x
        h_mismatch = visual_cx - shifted_math_cx

        log_lines.extend(
            [
                f"  [ALIGN]    Image Canvas Center Column    : {visual_cx} px",
                f"  [ALIGN]    Shifted Projection Matrix cx  : {shifted_math_cx:.2f} px",
                f"  [ALIGN]    HORIZONTAL CENTER CAL MISMATCH: {h_mismatch:.2f} pixels",
                "=" * 80 + "\n",
            ]
        )

        debug_output_string = "\n".join(log_lines)
        print(debug_output_string, flush=True)

        with open("live_capture_debug.log", "a") as f_debug:
            f_debug.write(debug_output_string)
        # ─────────────────────────────────────────────────────────────────────

        # Save values to local state arrays
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

        # Multi-Window Output Display Rendering Logic
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

        w_panel, h_panel = 320, 240
        combined = np.hstack(
            (
                cv2.resize(original_rgb, (w_panel, h_panel)),
                cv2.resize(mask_bgr, (w_panel, h_panel)),
                cv2.resize(depth_colored, (w_panel, h_panel)),
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

        cv2.imwrite(str(self.img_dir / f"{img_name}.png"), self.captured_original_rgb)
        cv2.imwrite(str(self.crop_rgb_dir / f"{img_name}.png"), self.captured_rgb)

        depth_vis = cv2.normalize(
            self.captured_depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(self.depth_dir / f"{img_name}.png"), depth_colored)

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

        cv2.imwrite(str(self.mask_dir / f"{img_name}.png"), label_mask * 255)

        pcd_path = self.pc_dir / f"{img_name}.ply"
        if self.captured_pcd is not None and len(self.captured_pcd.points) > 0:
            o3d.io.write_point_cloud(str(pcd_path), self.captured_pcd)
            print(
                f"[SAVED] Verified 3D Point Cloud saved successfully ({len(self.captured_pcd.points)} points)"
            )
        else:
            print("[ERROR] Cannot save .ply file: tracked cloud is empty!")

        current_crop = {
            "top": self.crop_top_var.get(),
            "bottom": self.crop_bottom_var.get(),
            "left": self.crop_left_var.get(),
            "right": self.crop_right_var.get(),
        }

        if self.captured_plane_model is not None:
            self.save_info_txt(
                img_name,
                selected_class,
                current_crop,
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
            print(
                "[WARNING] Skipping info.txt save because capture metadata is missing."
            )

        self.print_dataset_counts()
        self.counter = self.next_capture_index()
        self.current_img_name = None
        self.reset_capture_state()
        print(f"[SUCCESS] Packout cycle {img_name} complete.\n")

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
        largest_bbox, largest_centroid = None, None
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
            (self.crop_rgb_dir, [".png"]),
            (self.label_dir, [".txt"]),
            (self.depth_dir, [".png"]),
            (self.mask_dir, [".png"]),
            (self.info_dir, [".txt"]),
            (self.pc_dir, [".ply"]),
            (self.debug_dir, [".png"]),
        ):
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.suffix not in suffixes:
                    continue
                match = pattern.match(path.stem)
                if match:
                    indices.append(int(match.group(1)))
        return (max(indices) + 1) if indices else 0

    def match_mask_to_image(self, mask, image_shape):
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.shape == image_shape:
            return mask_u8
        return cv2.resize(
            mask_u8, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST
        )

    def print_dataset_counts(self):
        counts = {
            "images": len(list(self.img_dir.glob("*.png"))),
            "cropped_rgb": len(list(self.crop_rgb_dir.glob("*.png"))),
            "depth": len(list(self.depth_dir.glob("*.png"))),
            "labels": len(list(self.label_dir.glob("*.txt"))),
            "masks": len(list(self.mask_dir.glob("*.png"))),
            "pointcloud": len(list(self.pc_dir.glob("*.ply"))),
            "info": len(list(self.info_dir.glob("*.txt"))),
        }
        print(
            "[COUNT] " + ", ".join(f"{name}={value}" for name, value in counts.items())
        )
        return counts

    def preview_pointcloud_interactive(self):
        if self.captured_pcd is not None:
            print("[INFO] Launching interactive 3D point cloud viewer...")
            o3d.visualization.draw_geometries([self.captured_pcd])
        else:
            print("[WARNING] No point cloud to preview. Capture a frame first.")

    def retake_frame(self):
        print("[RETAKE] Retaking frame.")
        self.reset_capture_state()

    def reset_capture_state(self):
        self.captured_rgb = None
        self.captured_original_rgb = None
        self.captured_depth = None
        self.captured_mask = None
        self.captured_pcd = None
        # Add any missing cleanup bindings here...


# ─────────────────────────────────────────────────────────────────────
# FIXED EXECUTION RUNTIME BLOCK: HOLDS UI ENGINE OPEN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = RGBDCollectorApp(root)

    # This loop blocks the terminal thread and processes GUI events
    # until you explicitly click "Shutdown Subsystem Enclosure" or press Q
    root.mainloop()
# ───────────────────
