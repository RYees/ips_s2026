import argparse
import cv2

from camera.camera_test import CameraInterface as CameraTestInterface
from camera.camera_test2 import CameraInterface as CameraTest2Interface
from camera.camera_interface_fixed import CameraInterface as FixedCameraInterface


def build_camera_interface(name):
    if name == "test":
        return CameraTestInterface()
    if name == "test2":
        return CameraTest2Interface()
    if name == "fixed":
        return FixedCameraInterface()
    raise ValueError(f"Unknown backend: {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run an experimental camera preview backend."
    )
    parser.add_argument(
        "--backend",
        choices=["test", "test2", "fixed"],
        default="test2",
        help="Select the camera backend to preview.",
    )
    args = parser.parse_args()

    cam = build_camera_interface(args.backend)
    cam.setup_streams()

    while True:
        color, depth = cam.get_frames()

        if color is None or depth is None:
            continue

        cv2.imshow("RGB", color)
        cv2.imshow("Depth (Stable)", depth)

        if cv2.waitKey(1) == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
