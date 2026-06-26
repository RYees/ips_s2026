from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBAlignMode,
    OBError,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBPropertyID,
    OBSensorType,
    OBStreamType,
    Pipeline,
)
import open3d as o3d
import numpy as np
import cv2


class CameraInterface:
    """Orbbec RGB-D camera wrapper.

    This class provides a single stable capture backend for the project.
    It exposes a clean public API without changing the existing behavior used by main.py.
    """

    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.color_profile = None
        self.intrinsics = None
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.consecutive_missed_frames = 0
        self.total_frames_seen = 0
        self.total_frame_sets_with_depth = 0
        self.last_color_frame = None
        self.last_color_ts = None
        self.last_depth_frame = None
        self.last_depth_ts = None

    def print_default_camera_settings(self, device):
        print("\n=== DEFAULT CAMERA SETTINGS ===")
        props_to_check = [
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
            OBPropertyID.OB_PROP_COLOR_GAIN_INT,
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
            OBPropertyID.OB_PROP_IR_GAIN_INT,
            OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        ]
        for prop_id in props_to_check:
            try:
                val = device.get_int_property(prop_id)
                print(f"  {prop_id.name}: {val}")
            except Exception:
                try:
                    val = device.get_bool_property(prop_id)
                    print(f"  {prop_id.name}: {val}")
                except Exception:
                    print(f"  {prop_id.name}: Not supported")
        print("=== END SETTINGS ===\n")

    def setup_streams(self):
        # Prefer explicit RGB stream profiles so we avoid software color conversions.
        color_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        rgb_profile = None
        try:
            rgb_profile = color_sensor_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            print("[INFO] Using explicit RGB888 640x480 @ 30 fps color profile.")
        except OBError:
            pass

        if rgb_profile is None:
            try:
                rgb_profile = color_sensor_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
                print("[INFO] Using explicit RGB888 1280x720 @ 30 fps color profile.")
            except OBError:
                pass

        if rgb_profile is None:
            rgb_profile = color_sensor_list.get_default_video_stream_profile()
            fmt = rgb_profile.get_format() if hasattr(rgb_profile, 'get_format') else 'unknown'
            print(f"[WARNING] RGB888 profile not available; falling back to default ({fmt}).")

        self.color_profile = rgb_profile

        depth_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_sensor_list.get_default_video_stream_profile()

        device = self.pipeline.get_device()
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            print("[INFO] Color auto-exposure disabled.")
        except OBError:
            print("[WARNING] Could not disable color auto exposure.")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
            print("[INFO] Color exposure set to 100 µs.")
        except OBError:
            print("[WARNING] Could not set color exposure.")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 64)
            print("[INFO] Color gain set to 64.")
        except OBError:
            print("[WARNING] Could not set color gain.")

        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(depth_profile)
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)
        self.config.set_align_mode(OBAlignMode.SW_MODE)

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        self.print_default_camera_settings(device)

        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            raise RuntimeError("Unable to retrieve frames for intrinsics.")

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Unable to retrieve color frame for intrinsics.")

        color_frame = color_frame.as_video_frame()
        color_profile_obj = color_frame.get_stream_profile().as_video_stream_profile()
        intr = color_profile_obj.get_intrinsic()

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
        )
        print(f"[INFO] Camera intrinsics: {intr.width}×{intr.height}  fx={intr.fx:.2f} fy={intr.fy:.2f}  cx={intr.cx:.2f} cy={intr.cy:.2f}")

    def color_frame_to_rgb(self, color_frame) -> np.ndarray:
        """Convert any supported Orbbec color frame format to RGB uint8."""
        vf = color_frame.as_video_frame()
        fmt = vf.get_format()
        w = vf.get_width()
        h = vf.get_height()
        data = np.frombuffer(vf.get_data(), dtype=np.uint8)

        if fmt == OBFormat.RGB:
            img = data.reshape((h, w, 3))
            return img.copy()

        if fmt == OBFormat.BGR:
            img = data.reshape((h, w, 3))
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if fmt in (OBFormat.MJPG, OBFormat.JPEG):
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("MJPEG decode failed")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if fmt in (OBFormat.YUYV, OBFormat.YUY2):
            img = data.reshape((h, w, 2))
            return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_YUYV)

        if fmt == OBFormat.I420:
            img = data.reshape((h * 3 // 2, w))
            return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_I420)

        if fmt == OBFormat.NV12:
            img = data.reshape((h * 3 // 2, w))
            return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_NV12)

        if fmt == OBFormat.NV21:
            img = data.reshape((h * 3 // 2, w))
            return cv2.cvtColor(img, cv2.COLOR_YUV2RGB_NV21)

        print(f"[WARNING] Unsupported color format {fmt}; attempting MJPEG fallback.")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        raise RuntimeError(f"Unsupported color frame format: {fmt}")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            self.consecutive_missed_frames += 1
            print(f"[WARNING] No frames received (missed={self.consecutive_missed_frames}); retrying...")
            frames = self.pipeline.wait_for_frames(2000)
            if not frames:
                self.consecutive_missed_frames += 1
                print(f"[WARNING] Still no frames after 2000 ms (missed={self.consecutive_missed_frames})")
                return None, None

        self.consecutive_missed_frames = 0
        raw_color = frames.get_color_frame() if hasattr(frames, 'get_color_frame') else None
        raw_depth = frames.get_depth_frame() if hasattr(frames, 'get_depth_frame') else None
        self.total_frames_seen += 1
        if raw_color is not None and raw_depth is not None:
            self.total_frame_sets_with_depth += 1

        def frame_info(frame):
            if frame is None:
                return "None"
            parts = []
            if hasattr(frame, 'get_width') and hasattr(frame, 'get_height'):
                parts.append(f"{frame.get_width()}×{frame.get_height()}")
            if hasattr(frame, 'get_format'):
                parts.append(str(frame.get_format()))
            if hasattr(frame, 'get_timestamp'):
                parts.append(f"ts={frame.get_timestamp()}")
            return ",".join(parts)

        if raw_color is not None:
            self.last_color_frame = raw_color
            self.last_color_ts = raw_color.get_timestamp() if hasattr(raw_color, 'get_timestamp') else None
        if raw_depth is not None:
            self.last_depth_frame = raw_depth
            self.last_depth_ts = raw_depth.get_timestamp() if hasattr(raw_depth, 'get_timestamp') else None

        # print(
        #     f"[DEBUG] frames: total={self.total_frames_seen}, "
        #     f"depth_sets={self.total_frame_sets_with_depth}, "
        #     f"color={frame_info(raw_color)}, depth={frame_info(raw_depth)}"
        # )
        aligned = None
        try:
            aligned = self.align_filter.process(frames)
        except Exception as e:
            print(f"[WARNING] Align filter failed: {e}")

        if aligned is None:
            print("[WARNING] Align filter returned None")
            return None, None

        aligned_frames = aligned.as_frame_set() if hasattr(aligned, 'as_frame_set') else aligned
        if aligned_frames is None:
            print("[ERROR] Failed to obtain a usable aligned frame set")
            return None, None

        aligned_color = aligned_frames.get_color_frame() if hasattr(aligned_frames, 'get_color_frame') else None
        aligned_depth = aligned_frames.get_depth_frame() if hasattr(aligned_frames, 'get_depth_frame') else None
        # print(f"[DEBUG] aligned frames: color={bool(aligned_color)}({frame_info(aligned_color)}), depth={bool(aligned_depth)}({frame_info(aligned_depth)})")
        if aligned_color is not None and aligned_depth is not None:
            return aligned_color, aligned_depth

        if raw_color is not None and raw_depth is not None:
            print("[WARNING] Alignment fallback unavailable; returning raw color+depth pair")
            return raw_color, raw_depth

        if self.last_color_frame is not None and self.last_depth_frame is not None:
            dt = None
            if self.last_color_ts is not None and self.last_depth_ts is not None:
                dt = abs(self.last_color_ts - self.last_depth_ts)
            if dt is None or dt <= 2500:
                print(f"[INFO] Using cached color+depth pair (dt={dt})")
                return self.last_color_frame, self.last_depth_frame
            print(f"[WARNING] Cached pair too far apart (dt={dt})")

        print("[WARNING] Missing color or depth frame after aligned fallback")
        return None, None

    def get_intrinsics(self):
        if self.intrinsics is None:
            raise RuntimeError("Intrinsics not initialized. Call setup_streams() first.")
        return self.intrinsics

    def stop(self):
        self.pipeline.stop()
