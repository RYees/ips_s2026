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
        self.total_frames_seen = 0
        self.total_frame_sets_with_depth = 0

    def setup_streams(self):
        color_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        
        # Keep your exact working resolution preferences
        try:
            self.color_profile = color_sensor_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
            print("[INFO] Aligned Mode: Using explicit RGB888 1280x720 profile.")
        except OBError:
            try:
                self.color_profile = color_sensor_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
                print("[INFO] Aligned Mode: Using explicit RGB888 640x480 profile.")
            except OBError:
                self.color_profile = color_sensor_list.get_default_video_stream_profile()

        depth_sensor_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_sensor_list.get_default_video_stream_profile()

        device = self.pipeline.get_device()
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 64)
        except OBError:
            pass

        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(depth_profile)
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.ANY_SITUATION)
        self.config.set_align_mode(OBAlignMode.SW_MODE)

        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

        frames = self.pipeline.wait_for_frames(1000)
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
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return None, None

        # Force alignment tracking layer execution first
        try:
            aligned = self.align_filter.process(frames)
            if aligned is not None:
                aligned_frames = aligned.as_frame_set()
                return aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
        except Exception as e:
            print(f"[WARNING] Live alignment filter exception: {e}")

        return frames.get_color_frame(), frames.get_depth_frame()

    def stop(self):
        self.pipeline.stop()