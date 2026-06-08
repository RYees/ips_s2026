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


class CameraInterfaceAligned:
    """
    Alternative camera interface optimized strictly for hardware-synchronized, 
    matching aspect-ratio 2D/3D stream alignment.
    """

    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.color_profile = None
        self.intrinsics = None
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.consecutive_missed_frames = 0

    def setup_streams(self):
        color_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        # CRITICAL FIX: Force matching resolution (640x480) for both streams.
        # This gives them a shared aspect ratio (4:3), matching your model's 640x640 letterbox.
        try:
            self.color_profile = color_sensor_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            print("[ALIGN INFOTRONIC] Loaded 640x480 RGB profile successfully.")
        except OBError:
            self.color_profile = color_sensor_list.get_default_video_stream_profile()
            print("[ALIGN WARNING] 640x480 RGB profile failed. Loaded default.")

        try:
            depth_profile = depth_sensor_list.get_video_stream_profile(640, 480, OBFormat.Y16, 30)
            print("[ALIGN INFOTRONIC] Loaded matching 640x480 Depth profile.")
        except OBError:
            depth_profile = depth_sensor_list.get_default_video_stream_profile()
            print("[ALIGN WARNING] Fixed 640x480 depth profile unavailable; using default.")

        device = self.pipeline.get_device()
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 64)
            print("[ALIGN INFO] Manual exposures mapped perfectly.")
        except OBError:
            print("[ALIGN WARNING] Custom camera register modifications rejected by hardware properties.")

        # Enable matching array streams
        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(depth_profile)
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)
        
        # Turn on hardware/software translation matrix mode
        self.config.set_align_mode(OBAlignMode.SW_MODE)

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

        # Retrieve intrinsic camera metrics for open3d pipeline integrity
        frames = self.pipeline.wait_for_frames(1500)
        if frames:
            color_frame = frames.get_color_frame()
            if color_frame:
                vf = color_frame.as_video_frame()
                intr = vf.get_stream_profile().as_video_stream_profile().get_intrinsic()
                self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=intr.width, height=intr.height,
                    fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy
                )

    def get_frames(self):
        """Fetches frames directly through the active structural alignment matrix."""
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return None, None

        try:
            aligned = self.align_filter.process(frames)
            if aligned is not None:
                aligned_set = aligned.as_frame_set()
                return aligned_set.get_color_frame(), aligned_set.get_depth_frame()
        except Exception as e:
            print(f"[ALIGN FILTER SUB-ROUTINE CRASH] Matrix calculation error: {e}")

        # Fallback to direct stream pairs if filter yields empty output
        return frames.get_color_frame(), frames.get_depth_frame()

    def stop(self):
        self.pipeline.stop()