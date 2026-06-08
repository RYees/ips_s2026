import sys
import threading
import queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from camera.camera_interface import CameraInterface
from pyorbbecsdk import OBFormat

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH = "/home/cpsstudent/Documents/ips_s2026/rgbd/live/best.pt"

CLASS_NAMES  = {0: "Copper", 1: "Steel"}
CLASS_COLORS = {0: (139, 0, 0), 1: (128, 128, 0)}  # BGR

CONF_THRESHOLD = 0.45
IOU_THRESHOLD  = 0.40
MIN_MASK_AREA  = 0.002   # fraction of frame area — filters ghost detections


# ─────────────────────────────────────────────────────────────
# OWN FRAME CONVERTER (no dependency on processing/utils.py)
# ─────────────────────────────────────────────────────────────
def orbbec_frame_to_bgr(frame) -> np.ndarray | None:
    """Convert any Orbbec color frame directly to a BGR numpy array."""
    w   = frame.get_width()
    h   = frame.get_height()
    fmt = frame.get_format()
    data = frame.get_data()

    if fmt == OBFormat.RGB:
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
    elif fmt == OBFormat.MJPG:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    else:
        print(f"[ERROR] Unsupported color format: {fmt}")
        return None


# ─────────────────────────────────────────────────────────────
# LETTERBOX INVERSE TRANSFORM
#
# YOLO internally pads the input frame to a square (e.g. 640×640)
# before running inference. The mask coordinates in result.masks.xy
# are in that padded space, NOT in your original frame space.
#
# This function computes exactly how YOLO padded your frame and
# then maps every polygon point back to your original pixel coords.
# ─────────────────────────────────────────────────────────────
def unletterbox_polygon(polygon: np.ndarray, orig_h: int, orig_w: int,
                        model_imgsz: int = 640) -> np.ndarray:
    """
    Map polygon points from YOLO's letterboxed inference space back
    to the original frame's pixel coordinates.

    YOLO scales the frame so the longest side = model_imgsz, then
    pads the shorter side equally on both sides with black bars.
    """
    scale  = min(model_imgsz / orig_h, model_imgsz / orig_w)
    pad_x  = (model_imgsz - orig_w * scale) / 2   # horizontal padding added
    pad_y  = (model_imgsz - orig_h * scale) / 2   # vertical padding added

    pts = polygon.astype(np.float32).copy()
    pts[:, 0] = (pts[:, 0] - pad_x) / scale   # x
    pts[:, 1] = (pts[:, 1] - pad_y) / scale   # y

    # Clamp to frame bounds
    pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)

    return pts.astype(np.int32)


# ─────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────
def draw_detection(frame: np.ndarray, polygon: np.ndarray,
                   color: tuple, label: str) -> None:
    """
    Draw a clean outline directly on the object edge.
    No fill — just a sharp colored border + readable label.
    """
    if len(polygon) < 3:
        return

    # Sharp outline only — no fill, no flicker
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=3)

    # Label: black shadow behind white text
    text_x = int(polygon[:, 0].min())
    text_y = max(15, int(polygon[:, 1].min()) - 8)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# BACKGROUND CAPTURE THREAD
# ─────────────────────────────────────────────────────────────
def capture_worker(cam: CameraInterface, frame_q: queue.Queue,
                   stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        color_frame, _ = cam.get_frames()
        if color_frame is None:
            continue
        # Keep only the latest frame — drop stale ones
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(color_frame)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)

    # Read the imgsz the model was trained at (default 640 if not stored)
    model_imgsz = getattr(model, 'imgsz', 640)
    if isinstance(model_imgsz, (list, tuple)):
        model_imgsz = model_imgsz[0]
    print(f"[INFO] Model inference size: {model_imgsz}")

    print("[INFO] Initializing Orbbec Camera...")
    cam = CameraInterface()
    cam.setup_streams()

    stop_evt  = threading.Event()
    frame_q   = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(
        target=capture_worker,
        args=(cam, frame_q, stop_evt),
        daemon=True
    )
    cap_thread.start()

    print("[INFO] System ready. Press 'q' to exit.")

    frame_count = 0

    try:
        while True:
            color_frame = frame_q.get()

            bgr = orbbec_frame_to_bgr(color_frame)
            if bgr is None:
                continue

            orig_h, orig_w = bgr.shape[:2]

            # ── Frame skip: infer every other frame to reduce CPU load ──
            frame_count += 1
            if frame_count % 2 != 0:
                cv2.imshow("Industrial Sorting Feed", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── Inference ───────────────────────────────────────────────
            results = model.predict(
                source=bgr,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                half=False,    # only True with CUDA GPU
                stream=False,
                verbose=False,
            )

            annotated = bgr.copy()

            for result in results:
                if result.masks is None or len(result.boxes) == 0:
                    continue

                for mask_xy, box in zip(result.masks.xy, result.boxes):
                    # ── THE FIX: undo YOLO's internal letterbox padding ──
                    polygon = unletterbox_polygon(
                        mask_xy, orig_h, orig_w, model_imgsz
                    )

                    # Skip tiny / partial frame-edge detections
                    if cv2.contourArea(polygon) / (orig_w * orig_h) < MIN_MASK_AREA:
                        continue

                    class_id   = int(box.cls[0].item())
                    conf_score = box.conf[0].item()
                    label      = f"{CLASS_NAMES.get(class_id, 'Unknown')} {conf_score:.2f}"
                    color      = CLASS_COLORS.get(class_id, (0, 255, 0))

                    draw_detection(annotated, polygon, color, label)

            cv2.imshow("Industrial Sorting Feed", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] Inference loop crashed: {e}")
        raise
    finally:
        print("[INFO] Shutting down...")
        stop_evt.set()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Closed safely.")


if __name__ == "__main__":
    main()