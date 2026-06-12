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


class IndustrialSortingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Industrial Sorting Rig Control Panel")
        self.root.geometry("1400x900")

        # Core state variables
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

        # Initialize folders
        self.output_base = Path("offline_case/samples")
        self.output_base.mkdir(parents=True, exist_ok=True)
        (self.output_base / "images").mkdir(exist_ok=True)
        (self.output_base / "depth").mkdir(exist_ok=True)
        (self.output_base / "pointcloud").mkdir(exist_ok=True)
        (self.output_base / "info").mkdir(exist_ok=True)
        (self.output_base / "mask").mkdir(exist_ok=True)

        # Logging redirects
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        log_filename = (
            self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.log_file = open(log_filename, "w")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self.log_file)
        sys.stderr = TeeStream(sys.stderr, self.log_file)

        print(f"[INIT] Session logger initialized. Writing to {log_filename}")

        # Hardware interface initialization
        print("[INIT] Attaching camera sensor subsystem...")
        self.cam = CameraInterface()
        self.cam.start()
        print("[INIT] Camera stream activated successfully.")

        self.setup_ui()
        self.update_live_feed()

    def setup_ui(self):
        # Master Layout Panels
        self.left_panel = tk.Frame(self.root, width=950, bg="#2b2b2b")
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_panel = tk.Frame(
            self.root, width=450, bg="#3c3f41", bd=2, relief=tk.SUNKEN
        )
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Canvas Displays
        self.video_label = tk.Label(self.left_panel, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Interaction Control Panel
        self.controls_frame = tk.LabelFrame(
            self.right_panel,
            text=" Execution Pipeline Controllers ",
            bg="#3c3f41",
            fg="white",
        )
        self.controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self.capture_btn = tk.Button(
            self.controls_frame,
            text="CAPTURE FRAME",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="black",
            command=self.capture_current_frame,
        )
        self.capture_btn.pack(fill=tk.X, padx=10, pady=5)

        self.retake_btn = tk.Button(
            self.controls_frame,
            text="RETAKE FRAME",
            font=("Arial", 11),
            state=tk.DISABLED,
            command=self.retake_frame,
        )
        self.retake_btn.pack(fill=tk.X, padx=10, pady=5)

        # Class Categorization Block
        self.class_frame = tk.LabelFrame(
            self.right_panel,
            text=" Quality Assurance Label Classification ",
            bg="#3c3f41",
            fg="white",
        )
        self.class_frame.pack(fill=tk.X, padx=10, pady=10)

        self.selected_class = tk.IntVar(value=0)
        tk.Radiobutton(
            self.class_frame,
            text="Class 00: Copper Mass Asset",
            variable=self.selected_class,
            value=0,
            bg="#3c3f41",
            fg="white",
            selectcolor="#2b2b2b",
        ).pack(anchor=tk.W, padx=10, pady=2)
        tk.Radiobutton(
            self.class_frame,
            text="Class 01: Steel Structural Scrap",
            variable=self.selected_class,
            value=1,
            bg="#3c3f41",
            fg="white",
            selectcolor="#2b2b2b",
        ).pack(anchor=tk.W, padx=10, pady=2)

        self.save_btn = tk.Button(
            self.right_panel,
            text="COMMIT DATASET SNAPSHOT",
            font=("Arial", 12, "bold"),
            bg="#008CBA",
            fg="black",
            state=tk.DISABLED,
            command=self.commit_snapshot,
        )
        self.save_btn.pack(fill=tk.X, padx=20, pady=20)

        # Realtime Dynamic Matrix Sliders
        self.crop_frame = tk.LabelFrame(
            self.right_panel,
            text=" Realtime Hardware Crop Adjustments (px) ",
            bg="#3c3f41",
            fg="white",
        )
        self.crop_frame.pack(fill=tk.X, padx=10, pady=10)

        self.crop_top_var = tk.IntVar(value=0)
        self.crop_bottom_var = tk.IntVar(value=0)
        self.crop_left_var = tk.IntVar(value=0)
        self.crop_right_var = tk.IntVar(value=0)

        self.create_slider(self.crop_frame, "Top Margin:", self.crop_top_var, 0, 500)
        self.create_slider(
            self.crop_frame, "Bottom Margin:", self.crop_bottom_var, 0, 500
        )
        self.create_slider(self.crop_frame, "Left Margin:", self.crop_left_var, 0, 500)
        self.create_slider(
            self.crop_frame, "Right Margin:", self.crop_right_var, 0, 500
        )

        self.pcd_btn = tk.Button(
            self.right_panel, text="Launch open3D Previewer", command=self.preview_pcd
        )
        self.pcd_btn.pack(fill=tk.X, padx=20, pady=5)

        self.quit_btn = tk.Button(
            self.right_panel,
            text="Shutdown Subsystem Enclosure",
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

    def update_live_feed(self):
        if self.is_capturing:
            frames = self.cam.get_frames()
            if frames is not None:
                color_frame, _, _ = frames
                if color_frame is not None:
                    img_bgr = frame_to_bgr_image(color_frame)

                    # Read UI values dynamically
                    t = self.crop_top_var.get()
                    b = self.crop_bottom_var.get()
                    l = self.crop_left_var.get()
                    r = self.crop_right_var.get()

                    img_cropped, _, _ = crop_manual(
                        img_bgr, top=t, bottom=b, left=l, right=r
                    )

                    # Frame rate sizing logic
                    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
                    h, w = img_rgb.shape[:2]

                    # Protect Tkinter allocation bounds
                    max_h, max_w = 750, 900
                    scale = min(max_w / w, max_h / h, 1.0)
                    if scale < 1.0:
                        img_rgb = cv2.resize(
                            img_rgb,
                            (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(image=img_pil)
                    self.video_label.img_tk = img_tk
                    self.video_label.config(image=img_tk)

        self.root.after(30, self.update_live_feed)

    def capture_current_frame(self):
        print("\n[CAPTURE] Freezing live video frame buffer...")
        frames = self.cam.get_frames()
        if frames is None:
            print("[ERROR] Hardware pipeline frame capture timed out or empty.")
            return

        color_frame, depth_frame, pcd = frames
        if color_frame is None or depth_frame is None or pcd is None:
            print("[ERROR] Synchronized camera stream component array incomplete.")
            return

        self.is_capturing = False
        self.capture_btn.config(state=tk.DISABLED)
        self.retake_btn.config(state=tk.NORMAL)

        # Step 1: Retain pristine uncropped original color matrix
        self.captured_original_rgb = frame_to_bgr_image(color_frame)
        self.raw_rgb_shape = self.captured_original_rgb.shape

        # Step 2: Grab incoming spatial structures
        self.captured_depth = np.asanyarray(depth_frame.get_data())
        self.raw_depth_shape = self.captured_depth.shape

        # Step 3: Parse slider crop boundaries
        t = self.crop_top_var.get()
        b = self.crop_bottom_var.get()
        l = self.crop_left_var.get()
        r = self.crop_right_var.get()

        # Step 4: Deduct cropped variants
        self.captured_rgb, actual_left_offset, actual_top_offset = crop_manual(
            self.captured_original_rgb, top=t, bottom=b, left=l, right=r
        )
        self.cropped_rgb_shape = self.captured_rgb.shape
        self.cropped_depth_shape = (
            self.captured_depth.shape
        )  # Uncropped tracking variant

        # Step 5: Gather and package spatial tracking indices
        self.captured_pcd = pcd
        intrinsic_profile = self.cam.get_rgb_intrinsics()

        info = CaptureInfo()
        info.raw_depth_shape = (self.raw_depth_shape[0], self.raw_depth_shape[1])
        info.rgb_shape = (self.cropped_rgb_shape[0], self.cropped_rgb_shape[1])
        info.crop_top = actual_top_offset
        info.crop_left = actual_left_offset
        info.fx = intrinsic_profile.fx
        info.fy = intrinsic_profile.fy
        info.cx = intrinsic_profile.cx
        info.cy = intrinsic_profile.cy

        self.captured_intrinsics = intrinsic_profile

        # ─────────────────────────────────────────────────────────────────────
        # CRITICAL RE-ENGINEERED DIAGNOSTIC ENGINE & FILE LOGGING MODULE
        # ─────────────────────────────────────────────────────────────────────
        log_lines = [
            "\n" + "=" * 80,
            "             REALTIME LIVE INTRINSICS MATRIX DIAGNOSTIC REPORT",
            "=" * 80,
            f"  Timestamp Code Generation : {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            f"  [HARDWARE] Raw Color Sensor Matrix Shape : {self.raw_rgb_shape[1]}W x {self.raw_rgb_shape[0]}H px",
            f"  [HARDWARE] Raw Depth Sensor Matrix Shape : {self.raw_depth_shape[1]}W x {self.raw_depth_shape[0]}H px",
            f"  [SLIDERS]  Active Crop Offsets Input     : Top={t}px, Bottom={b}px, Left={l}px, Right={r}px",
            f"  [PIPELINE] Cropped App Canvas Boundary  : {self.cropped_rgb_shape[1]}W x {self.cropped_rgb_shape[0]}H px",
            f"  [MATRIX]   Intrinsic fx (Focal X)        : {intrinsic_profile.fx:.6f}",
            f"  [MATRIX]   Intrinsic fy (Focal Y)        : {intrinsic_profile.fy:.6f}",
            f"  [MATRIX]   Intrinsic cx (Center X)       : {intrinsic_profile.cx:.6f}",
            f"  [MATRIX]   Intrinsic cy (Center Y)       : {intrinsic_profile.cy:.6f}",
        ]

        # Calculate exact visual midpoint offsets
        visual_cx = self.cropped_rgb_shape[1] / 2
        visual_cy = self.cropped_rgb_shape[0] / 2
        shifted_math_cx = intrinsic_profile.cx - actual_left_offset
        shifted_math_cy = intrinsic_profile.cy - actual_top_offset
        h_mismatch = visual_cx - shifted_math_cx

        log_lines.extend(
            [
                f"  [ALIGN]    Image Canvas Center Column    : {visual_cx} px",
                f"  [ALIGN]    Shifted Projection Matrix cx  : {shifted_math_cx:.2f} px",
                f"  [ALIGN]    HORIZONTAL CENTER CAL MISMATCH: {h_mismatch:.2f} pixels",
                "=" * 80 + "\n",
            ]
        )

        # Write to screen and directly append to local file asset
        debug_output_string = "\n".join(log_lines)
        print(debug_output_string, flush=True)

        with open("live_capture_debug.log", "a") as f_debug:
            f_debug.write(debug_output_string)

        # ─────────────────────────────────────────────────────────────────────

        print(
            "[PROCESSING] Handing array data pointers over to Masking Pinhole Engine..."
        )
        mask, plane_model, inliers_count, non_plane_count = (
            run_masking_from_point_cloud(pcd, info)
        )

        self.captured_mask = mask
        self.captured_plane_model = plane_model
        self.captured_plane_inliers = inliers_count
        self.captured_non_plane_pts = non_plane_count

        # Post Process Mask Overlay display logic
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_rgb[mask > 0] = [
            0,
            255,
            0,
        ]  # Light up winning object in vibrant emerald green

        overlay_img = cv2.addWeighted(self.captured_rgb, 0.7, mask_rgb, 0.3, 0)
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

        h, w = overlay_rgb.shape[:2]
        max_h, max_w = 750, 900
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            overlay_rgb = cv2.resize(
                overlay_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        img_pil = Image.fromarray(overlay_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)

        self.save_btn.config(state=tk.NORMAL)

    def commit_snapshot(self):
        if self.captured_rgb is None:
            return

        # Sequential file indexing locator
        img_idx = 0
        while True:
            name = f"img{img_idx:04d}"
            if not (self.output_base / "images" / f"{name}.png").exists():
                self.current_img_name = name
                break
            img_idx += 1

        name = self.current_img_name
        print(f"\n[DATASET] Committing pipeline record arrays as index token: {name}")

        cv2.imwrite(str(self.output_base / "images" / f"{name}.png"), self.captured_rgb)
        cv2.imwrite(str(self.output_base / "mask" / f"{name}.png"), self.captured_mask)
        np.save(str(self.output_base / "depth" / f"{name}.npy"), self.captured_depth)
        o3d.io.write_point_cloud(
            str(self.output_base / "pointcloud" / f"{name}.ply"), self.captured_pcd
        )

        # Extract intrinsic profile details
        fx = self.captured_intrinsics.fx
        fy = self.captured_intrinsics.fy
        cx = self.captured_intrinsics.cx
        cy = self.captured_intrinsics.cy

        t = self.crop_top_var.get()
        b = self.crop_bottom_var.get()
        l = self.crop_left_var.get()
        r = self.crop_right_var.get()

        info_text = (
            f"Crop:\n  top={t} bottom={b} left={l} right={r}\n"
            f"Raw RGB shape: {self.raw_rgb_shape}\n"
            f"Raw depth shape: {self.raw_depth_shape}\n"
            f"Cropped RGB shape: {self.cropped_rgb_shape}\n"
            f"Cropped depth shape: {self.cropped_depth_shape}\n"
            f"RGB shape: {self.cropped_rgb_shape}\n"
            f"Depth shape: {self.raw_depth_shape}\n"
            f"Depth dtype: {self.captured_depth.dtype}\n"
            f"Depth min/max: {self.captured_depth.min()}/{self.captured_depth.max()}\n"
            f"Valid depth pixels: {np.count_nonzero(self.captured_depth)}\n"
            f"Intrinsics matrix:\n"
            f"  {fx:.6f} 0.000000 {cx:.6f}\n"
            f"  0.000000 {fy:.6f} {cy:.6f}\n"
            f"  0.000000 0.000000 1.000000\n"
        )

        if self.captured_plane_model is not None:
            pa, pb, pc, pd = self.captured_plane_model
            info_text += (
                f"Plane equation: {pa:.6f}x + {pb:.6f}y + {pc:.6f}z + {pd:.6f} = 0\n"
            )
            info_text += f"Plane inliers: {self.captured_plane_inliers}\n"
            info_text += f"Non-plane points: {self.captured_non_plane_pts}\n"

        info_text += f"Selected class: {self.selected_class.get()}\n"

        # Structural validation verification metrics
        fg_pixels = np.count_nonzero(self.captured_mask)
        total_pixels = self.captured_mask.size
        fg_ratio = fg_pixels / total_pixels if total_pixels > 0 else 0

        info_text += (
            f"Mask QA:\n"
            f"  Foreground pixels: {fg_pixels}\n"
            f"  Foreground ratio: {fg_ratio:.8f}\n"
        )

        (self.output_base / "info" / f"{name}.txt").write_text(info_text)
        print(f"[DATASET] Commit sequence successful for image token context: {name}")

        self.save_btn.config(state=tk.DISABLED)
        self.reset_capture_state()

    def preview_pcd(self):
        if self.captured_pcd is not None:
            print("[VISUALIZATION] Spawning Open3D parallel renderer viewport...")
            o3d.visualization.draw_geometries(
                [self.captured_pcd], window_name="Rig Core Spatial Point Cloud Profile"
            )
        else:
            print("[WARNING] Active point cloud buffer empty. Capture a frame first.")

    def retake_frame(self):
        print("[RETAKE] Releasing frame buffer lock.")
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
        print("[INFO] Initiating camera enclosure hardware teardown...")
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
    root = tk.Tk()
    app = IndustrialSortingApp(root)
    root.mainloop()
