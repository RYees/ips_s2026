import numpy as np
import cv2

class AnnotationWriter:
    def __init__(self, label_class=0, normalized=True):
        #self.label_class = label_class
        self.normalized = normalized

    def write(self, filepath, mask, img_shape, label_class):
        height, width = img_shape

        # Find external contours (object outlines)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Select the largest contour (assuming single object per image)
        contour = max(contours, key=cv2.contourArea)

        # Flatten and reshape contour to a 2D array of (x, y) coordinates
        contour = contour.squeeze()
        if contour.ndim != 2:
            return False  # no valid contour found

        # Normalize coordinates to [0, 1]
        if self.normalized:
            polygon = [(x / width, y / height) for x, y in contour]
        else:
            polygon = [(x, y) for x, y in contour]

        # Prepare string format
        poly_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in polygon])
        line = f"{label_class} {poly_str}"

        # Write to YOLO segmentation file
        with open(filepath, "w") as f:
            f.write(line + "\n")
        return True