import cv2
import numpy as np
from ultralytics import YOLO
from rgbd.camera.camera_interface import CameraInterface
from rgbd.processing.utils import frame_to_bgr_image

# 1. Load your custom YOLO26 Segmentation weights
# (Update this path to your best model weights)
MODEL_PATH = "/Users/rzapp/Documents/A{sp}A/ips_s2026/offline_case/samples/runs/segment/runs/version_compare/yolov8n-seg/weights/best.pt"
model = YOLO(MODEL_PATH)

# 2. Initialize your Orbbec Hardware Camera wrapper
print("[INFO] Initializing Orbbec Camera Stream Profiles...")
cam = CameraInterface()
cam.setup_streams()

# Define labels and colors to match your two classes clearly
# Class 0: Copper (Bright Vibrant Green) | Class 1: Steel (Industrial Red)
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
CLASS_LABELS = {0: "Copper", 1: "Steel"}

print("\n[INFO] System Ready! Starting Live Industrial Scrap Tracking...")
print("[INFO] Move a piece of metal under the camera field of view.")
print("[INFO] Click on the video window and press 'q' to safely close.")

try:
    while True:
        # Get live synced color and depth frames from the hardware pipeline
        color_frame, depth_frame = cam.get_frames()
        if color_frame is None:
            continue

        # Convert the Orbbec SDK image format to standard BGR for OpenCV rendering
        raw_rgb = frame_to_bgr_image(color_frame)
        
        # =====================================================================
        # APPLICATION CROP ZONE (Matching your dataset collection profile)
        # =====================================================================
        # Your collection app uses: {'top': 0, 'bottom': 0, 'left': 250, 'right': 520}
        h, w = raw_rgb.shape[:2]
        left_crop = 250
        right_crop = 520
        # Crop the sides to center perfectly on your material runway
        display_frame = raw_rgb[:, left_crop : w - right_crop]

        # 3. Run your model inference on the cropped frame
        # Keep conf=0.50+ to ensure background glares do not create messy false shapes
        results = model.predict(source=display_frame, conf=0.55, verbose=False)

        # 4. Check if the model has isolated any contours
        if results[0].masks is not None and len(results[0].boxes) > 0:
            # Custom visualization layer: Thin contours and clear tags
            annotated_frame = results[0].plot(
                boxes=False,       # Hides square bounding boxes for a cleaner look
                masks=True,        # Enables precise contour wrapping
                labels=True,       # Displays Class tag and confidence %
                line_width=2,      # Sharp outline thickness
                mask_alpha=0.3,    # Sleek transparent fill overlay
                colors=CLASS_COLORS
            )
        else:
            # If nothing is detected, display the clean camera workspace
            annotated_frame = display_frame.copy()

        # Display the live tracked workspace window
        cv2.imshow("Industrial Sorting Feed - Live Tracking", annotated_frame)

        # Press 'q' to close the feed safely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"[FATAL ERROR] Inference loop crashed: {e}")

finally:
    # Safely release camera hardware pipelines to prevent environment freezing
    print("[INFO] Releasing camera streams safely...")
    cam.stop()
    cv2.destroyAllWindows()
    print("[INFO] Application closed.")