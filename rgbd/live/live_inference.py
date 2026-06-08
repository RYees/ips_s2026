import sys
from pathlib import Path

# Fix path resolution so the script can see 'camera' and 'processing' folders
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from camera.camera_interface import CameraInterface
from processing.utils import frame_to_bgr_image

def main():
    # 1. Load your custom YOLOv8 Segmentation weights
    MODEL_PATH = "/home/cpsstudent/Documents/ips_s2026/rgbd/live/best.pt"
    model = YOLO(MODEL_PATH)

    # 2. Map custom industrial labels and colors explicitly
    # Class 0 (Copper): Dark Blue -> BGR: (139, 0, 0)
    # Class 1 (Steel): Teal -> BGR: (128, 128, 0)
    CLASS_NAMES = {0: "Copper", 1: "Steel"}
    CLASS_COLORS = {0: (139, 0, 0), 1: (128, 128, 0)}

    # 3. Initialize your Orbbec Hardware Camera wrapper
    print("[INFO] Initializing Orbbec Camera Stream Profiles...")
    cam = CameraInterface()
    cam.setup_streams()

    print("\n[INFO] System Ready! Starting Full-Screen Detection View...")
    print("[INFO] Click on the video window and press 'q' to safely exit.")

    try:
        while True:
            color_frame, _ = cam.get_frames()
            if color_frame is None:
                continue

            # Convert Orbbec frame to BGR OpenCV image array (FULL VIEW)
            raw_rgb = frame_to_bgr_image(color_frame)
            h, w = raw_rgb.shape[:2]
            annotated_frame = raw_rgb.copy()

            # 4. Stream Optimization: stream=True minimizes memory overhead and lags
            # half=True forces FP16 mathematical operations for faster calculations
            results_generator = model.predict(
                source=raw_rgb, 
                conf=0.25, 
                half=True, 
                stream=True, 
                verbose=False
            )

            for result in results_generator:
                if result.masks is not None and len(result.boxes) > 0:
                    # Loop through every tracked piece of scrap metal using normalized shapes (xyn)
                    for mask_norm, box in zip(result.masks.xyn, result.boxes):
                        class_id = int(box.cls[0].item())
                        conf_score = box.conf[0].item()
                        
                        label_text = f"{CLASS_NAMES.get(class_id, 'Unknown')} {conf_score:.4f}"
                        mask_color = CLASS_COLORS.get(class_id, (0, 255, 0))

                        # Denormalize points back to exact raw frame pixel size
                        # This locks the contours strictly over the real objects
                        polygon = (mask_norm * np.array([w, h])).astype(np.int32)
                        
                        if len(polygon) > 0:
                            # Draw transparent mask layer over the material piece
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [polygon], mask_color)
                            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

                            # Draw a sharp edge border line matching the item class color
                            cv2.polylines(annotated_frame, [polygon], isClosed=True, color=mask_color, thickness=2)

                            # Locate top-most edge point for text placement
                            text_x = int(polygon[:, 0].min())
                            text_y = int(polygon[:, 1].min()) - 7
                            text_y = max(15, text_y) # Prevent clipping past top screen bounds

                            # Double Text Rendering:
                            # First layer: Thick black shadow background outline
                            cv2.putText(
                                annotated_frame, 
                                label_text, 
                                (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 0),       # Black outline color
                                3, 
                                cv2.LINE_AA
                            )
                            # Second layer: Crisp White foreground layer
                            cv2.putText(
                                annotated_frame, 
                                label_text, 
                                (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), # White text color
                                1, 
                                cv2.LINE_AA
                            )

                # Render the final composite tracking workspace view
                cv2.imshow("Industrial Sorting Feed - Full Field of View", annotated_frame)

            # Instantly handle keyboard loops to avoid window freezing
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] Inference loop crashed: {e}")
    finally:
        print("[INFO] Releasing camera hardware streams...")
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Application closed safely.")

if __name__ == "__main__":
    main()