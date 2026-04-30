import cv2
from camera_test2 import CameraInterface

def main():
    cam = CameraInterface()
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