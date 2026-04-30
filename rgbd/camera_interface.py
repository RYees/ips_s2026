from pyorbbecsdk import *
import open3d as o3d

class CameraInterface:
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
                print(f"{prop_id.name}: {val}")
            except Exception:
                try:
                    val = device.get_bool_property(prop_id)
                    print(f"{prop_id.name}: {val}")
                except Exception:
                    print(f"{prop_id.name}: Not supported")

        print("=== END SETTINGS ===\n")

    def setup_streams(self):
        # Setup color stream (prefer RGB format)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = color_profiles.get_default_video_stream_profile()
        
        # Setup depth stream with default profile
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()

        device = self.pipeline.get_device()
        # self.print_supported_properties(device)
        # self.print_default_camera_settings(device)
        # print("=== OBPropertyID Members ===")
        # for attr in dir(OBPropertyID):
        #    if not attr.startswith("__"):
        #        val = getattr(OBPropertyID, attr)
        #        print(f"{attr}: {val}")


        # --- Configure color stream properties ---
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)  # Disable auto exposure
        except OBError:
            print("[WARNING] Could not disable color auto exposure")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)  # Set manual exposure in microseconds
        except OBError:
            print("[WARNING] Could not set color exposure")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 64)  # Set manual gain
        except OBError:
            print("[WARNING] Could not set color gain")

        print("[INFO] Color stream exposure and gain set.")

        # --- Configure depth stream properties ---
        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, 10000)  # microseconds
        #except OBError:
        #    print("[WARNING] Could not set depth exposure")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, 16)         # gain value
        #except OBError:
        #    print("[WARNING] Could not set depth gain")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_IR_EXPOSURE_INT, 2000)      # microseconds
        #except OBError:
        #    print("[WARNING] Could not set IR exposure")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_IR_GAIN_INT, 24)            # gain value
        #except OBError:
        #    print("[WARNING] Could not set IR gain")

        #print("[INFO] Depth and IR exposure/gain successfully configured.")

        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(depth_profile)
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)

        # Start the pipeline with the configured streams

        # Get the depth sensor to apply hardware alignment settings
        self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        self.print_default_camera_settings(device)

        # Retrieve intrinsics from first frame
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            raise RuntimeError("Unable to retrieve frames for intrinsics.")

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Unable to retrieve color frame for intrinsics.")

        color_frame = color_frame.as_video_frame()
        color_profile = color_frame.get_stream_profile().as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsic()

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_intrinsics.width,
            height=color_intrinsics.height,
            fx=color_intrinsics.fx,
            fy=color_intrinsics.fy,
            cx=color_intrinsics.cx,
            cy=color_intrinsics.cy
        )
        print("[INFO] Camera intrinsics retrieved successfully.")


    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            self.consecutive_missed_frames += 1
            print(f"[WARNING] No frames received from pipeline (missed={self.consecutive_missed_frames}); retrying with longer timeout")
            frames = self.pipeline.wait_for_frames(2000)
            if not frames:
                self.consecutive_missed_frames += 1
                print(f"[WARNING] Still no frames after 2000ms (missed={self.consecutive_missed_frames})")
                return None, None

        self.consecutive_missed_frames = 0
        raw_color = frames.get_color_frame() if hasattr(frames, 'get_color_frame') else None
        raw_depth = frames.get_depth_frame() if hasattr(frames, 'get_depth_frame') else None
        self.total_frames_seen += 1
        if raw_color is not None and raw_depth is not None:
            self.total_frame_sets_with_depth += 1

        def frame_info(frame):
            if frame is None:
                return None
            info = []
            if hasattr(frame, 'get_width') and hasattr(frame, 'get_height'):
                info.append(f"{frame.get_width()}x{frame.get_height()}")
            if hasattr(frame, 'get_format'):
                info.append(str(frame.get_format()))
            if hasattr(frame, 'get_timestamp'):
                info.append(f"ts={frame.get_timestamp()}")
            return ",".join(info)

        if raw_color is not None:
            self.last_color_frame = raw_color
            self.last_color_ts = raw_color.get_timestamp() if hasattr(raw_color, 'get_timestamp') else None
        if raw_depth is not None:
            self.last_depth_frame = raw_depth
            self.last_depth_ts = raw_depth.get_timestamp() if hasattr(raw_depth, 'get_timestamp') else None

        print(
            f"[DEBUG] got frames: total_seen={self.total_frames_seen}, depth_sets={self.total_frame_sets_with_depth}, "
            f"raw_color={bool(raw_color)}({frame_info(raw_color)}), raw_depth={bool(raw_depth)}({frame_info(raw_depth)})"
        )
        if raw_color is not None and raw_depth is not None:
            return raw_color, raw_depth

        if self.last_color_frame is not None and self.last_depth_frame is not None:
            dt = None
            if self.last_color_ts is not None and self.last_depth_ts is not None:
                dt = abs(self.last_color_ts - self.last_depth_ts)
            if dt is None or dt <= 2500:
                print(f"[INFO] using cached color+depth pair, dt={dt}")
                return self.last_color_frame, self.last_depth_frame
            print(f"[WARNING] cached color+depth too far apart, dt={dt}")

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
        print(f"[DEBUG] aligned frames: color={bool(aligned_color)}({frame_info(aligned_color)}), depth={bool(aligned_depth)}({frame_info(aligned_depth)})")
        if aligned_color is not None and aligned_depth is not None:
            return aligned_color, aligned_depth

        print("[WARNING] Missing color or depth frame after aligned fallback")
        return None, None

    def get_intrinsics(self):
        if self.intrinsics is None:
            raise RuntimeError("Intrinsics not initialized. Call setup_streams() first.")
        return self.intrinsics

    def stop(self):
        self.pipeline.stop()
