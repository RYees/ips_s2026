import numpy as np
import cv2
from pyorbbecsdk import *

class CameraInterface:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def setup_streams(self):
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        color_profile = color_profiles.get_default_video_stream_profile()
        depth_profile = depth_profiles.get_default_video_stream_profile()

        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)

        self.config.set_align_mode(OBAlignMode.SW_MODE)

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

        print("[INFO] Camera started")

        # Warmup (VERY IMPORTANT)
        for _ in range(20):
            self.pipeline.wait_for_frames(100)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return None, None

        # Safe alignment
        try:
            aligned = self.align_filter.process(frames)
            if aligned is not None:
                frames = aligned.as_frame_set()
        except:
            pass

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # =========================
        # 🔥 COLOR (MJPEG SAFE)
        # =========================
        color_format = color_frame.get_format()
        color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        if color_format == OBFormat.MJPG:
            color = cv2.imdecode(color_data, cv2.IMREAD_COLOR)

        elif color_format == OBFormat.RGB:
            color = color_data.reshape(
                (color_frame.get_height(), color_frame.get_width(), 3)
            )
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        elif color_format == OBFormat.BGR:
            color = color_data.reshape(
                (color_frame.get_height(), color_frame.get_width(), 3)
            )

        else:
            print(f"[WARNING] Unsupported color format: {color_format}")
            return None, None

        # =========================
        # 🔥 DEPTH (VIEWER-LIKE)
        # =========================
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth = depth.reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        )

        depth_scale = depth_frame.get_depth_scale()
        depth_m = depth * depth_scale

        # Remove invalid values
        depth_m[depth_m == 0] = np.nan

        # ---- Step 1: Denoise ----
        depth_filtered = cv2.medianBlur(
            np.nan_to_num(depth_m).astype(np.float32), 5
        )

        # ---- Step 2: Edge-preserving smoothing ----
        depth_filtered = cv2.bilateralFilter(depth_filtered, 9, 75, 75)

        # ---- Step 3: Adaptive range ----
        valid = depth_filtered[depth_filtered > 0]

        if valid.size == 0:
            return color, None

        min_d = np.percentile(valid, 2)
        max_d = np.percentile(valid, 98)

        if max_d - min_d < 0.2:
            max_d = min_d + 0.2

        depth_norm = (depth_filtered - min_d) / (max_d - min_d)
        depth_norm = np.clip(depth_norm, 0, 1)

        depth_8bit = (depth_norm * 255).astype(np.uint8)

        # ---- Step 4: Histogram equalization ----
        depth_8bit = cv2.equalizeHist(depth_8bit)

        # ---- Step 5: Colormap ----
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # Optional contrast boost
        depth_colormap = cv2.convertScaleAbs(depth_colormap, alpha=1.2, beta=10)

        return color, depth_colormap

    def stop(self):
        self.pipeline.stop()