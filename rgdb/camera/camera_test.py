import numpy as np
import cv2
from pyorbbecsdk import *

class CameraInterface:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        # Stability
        self.prev_depth = None
        self.min_d = None
        self.max_d = None

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

        # Warmup
        for _ in range(20):
            self.pipeline.wait_for_frames(100)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return None, None

        # Alignment
        try:
            aligned = self.align_filter.process(frames)
            if aligned:
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
            print("[WARNING] Unsupported color format")
            return None, None

        # =========================
        # 🔥 DEPTH (STABLE + SINGLE BG)
        # =========================
        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        depth = depth.reshape((depth_frame.get_height(), depth_frame.get_width()))

        depth_scale = depth_frame.get_depth_scale()
        depth_m = depth * depth_scale

        # Remove invalid
        depth_m[depth_m == 0] = np.nan

        # Light smoothing (no distortion)
        depth_filtered = cv2.GaussianBlur(
            np.nan_to_num(depth_m).astype(np.float32), (5, 5), 0
        )

        valid = depth_filtered[depth_filtered > 0]
        if valid.size == 0:
            return color, None

        # 🔥 Background detection
        bg_depth = np.median(valid)
        bg_tol = 0.03
        bg_mask = np.abs(depth_filtered - bg_depth) < bg_tol

        # 🔥 Stable adaptive range
        curr_min = np.percentile(valid, 5)
        curr_max = np.percentile(valid, 95)

        if self.min_d is None:
            self.min_d = curr_min
            self.max_d = curr_max
        else:
            alpha = 0.1
            self.min_d = (1 - alpha) * self.min_d + alpha * curr_min
            self.max_d = (1 - alpha) * self.max_d + alpha * curr_max

        if self.max_d - self.min_d < 0.1:
            self.max_d = self.min_d + 0.1

        # Normalize
        depth_norm = (depth_filtered - self.min_d) / (self.max_d - self.min_d)
        depth_norm = np.clip(depth_norm, 0, 1)

        # Temporal smoothing
        if self.prev_depth is None:
            self.prev_depth = depth_norm
        else:
            beta = 0.2
            depth_norm = (1 - beta) * self.prev_depth + beta * depth_norm
            self.prev_depth = depth_norm

        depth_8bit = (depth_norm * 255).astype(np.uint8)

        # Colormap
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # 🔥 FORCE SINGLE BACKGROUND COLOR (GREEN)
        depth_colormap[bg_mask] = [0, 255, 0]

        return color, depth_colormap

    def stop(self):
        self.pipeline.stop()