import numpy as np
import cv2
from pyorbbecsdk import OBFormat

def frame_to_bgr_image(frame):
    width = frame.get_width()
    height = frame.get_height()
    format = frame.get_format()
    # print(f"[DEBUG] Frame: width={width}, height={height}, format={format}")

    data = frame.get_data()
    
    if format == OBFormat.RGB:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif format == OBFormat.BGR:
        img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
        return img
    elif format == OBFormat.MJPG:
        # Decode MJPEG
        img = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    else:
        print(f"[ERROR] Unsupported color format: {format}")
        return None
