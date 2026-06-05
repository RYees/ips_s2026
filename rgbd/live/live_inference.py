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
    MODEL_PATH = "/Users/rzapp/Documents/A{sp}A/ips_s2026/offline_case/samples/runs/segment/runs/version_compare/yolov8n-seg/weights/best.pt"
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

    print("\n[INFO] System Ready! Starting Optimized Sorting Feed...")
    print("[INFO] Click on the video window and press 'q' to safely exit.")

    try:
        while True:
            color_frame, _ = cam.get_frames()
            if color_frame is None:
                continue

            # Convert Orbbec frame to BGR OpenCV image array
            raw_rgb = frame_to_bgr_image(color_frame)
            h, w = raw_rgb.shape[:2]
            
            # Slicing bounds matching your collection app layout
            left_crop = 250
            right_crop = 520
            display_frame = raw_rgb[:, left_crop : w - right_crop].copy()

            # 4. Stream Optimization: stream=True handles memory allocation much faster
            # half=True drops calculation matrix to FP16 to maximize processing speed
            results_generator = model.predict(
                source=display_frame, 
                conf=0.25, 
                half=True, 
                stream=True, 
                verbose=False
            )

            for result in results_generator:
                # Create a clean canvas copy for drawing our aligned overlays
                annotated_frame = display_frame.copy()

                if result.masks is not None and len(result.boxes) > 0:
                    # Loop through every piece of scrap isolated by the model
                    for mask, box in zip(result.masks.xy, result.boxes):
                        class_id = int(box.cls[0].item())
                        conf_score = box.conf[0].item()
                        
                        # Fetch the assigned naming profile and color
                        label_text = f"{CLASS_NAMES.get(class_id, 'Unknown')} {conf_score:.2f}"
                        color = CLASS_COLORS.get(class_id, (0, 255, 0))

                        # Convert polygon coordinates to an integer matrix
                        polygon = mask.astype(np.int32)
                        
                        if len(polygon) > 0:
                            # Draw transparency filled overlay over the scrap piece
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [polygon], color)
                            cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)

                            # Draw a clean, sharp outer edge outline directly matching the object
                            cv2.polylines(annotated_frame, [polygon], isClosed=True, color=color, thickness=2)

                            # Compute the top-most point of the object polygon to anchor the text label
                            text_x = int(polygon[:, 0].min())
                            text_y = int(polygon[:, 1].min()) - 7
                            text_y = max(15, text_y) # Prevent clipping out of screen boundaries

                            # Render the explicit text string (e.g., "Copper 0.89")
                            cv2.putText(
                                annotated_frame, 
                                label_text, 
                                (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                color, 
                                2, 
                                cv2.LINE_AA
                            )

                # Render the final tracking workspace view
                cv2.imshow("Industrial Sorting Feed - Aligned Production View", annotated_frame)

            # Instantly catch keystrokes to prevent loop hang ups
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] Inference loop crashed: {e}")
    finally:
        print("[INFO] Releasing camera hardware streams...")
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")

if __name__ == "__main__":
    main()