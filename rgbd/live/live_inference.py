import sys
import threading
import queue
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

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = "/Users/rzapp/Documents/A{sp}A/ips_s2026/offline_case/samples/runs/segment/runs/version_compare/yolov8n-seg/weights/best.pt"

CLASS_NAMES  = {0: "Copper", 1: "Steel"}
CLASS_COLORS = {0: (139, 0, 0), 1: (128, 128, 0)}  # BGR

CONF_THRESHOLD  = 0.45   # raise to suppress weak/ghost edge detections
IOU_THRESHOLD   = 0.40   # lower NMS threshold removes overlapping duplicates
MIN_MASK_AREA   = 0.002  # ignore masks smaller than 0.2 % of frame area


# ─────────────────────────────────────────────
# THREADED CAMERA CAPTURE
# Runs in background so inference never waits
# for a new frame to arrive from the hardware.
# Queue size = 1 guarantees we always process
# the LATEST frame and never build up a backlog.
# ─────────────────────────────────────────────
def capture_worker(cam: CameraInterface, frame_q: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        color_frame, _ = cam.get_frames()
        if color_frame is None:
            continue
        # Drop stale frame so the queue always holds only the newest one
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(color_frame)


# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_detection(frame: np.ndarray, polygon: np.ndarray, color: tuple, label: str):
    """
    Draw a glowing contour outline directly on the object.
    No fill — only the edge is drawn so the object underneath
    stays fully visible and the outline sits exactly on it.

    FIX for alignment:
      We use result.masks.xy (already in original-frame pixel coords)
      instead of result.masks.xyn * [w, h].
      Ultralytics scales xyn back to the source resolution internally,
      but floating-point rounding can introduce a sub-pixel offset.
      Using .xy avoids that entirely.
    """
    if len(polygon) < 3:
        return

    # --- Outer glow: thick semi-transparent ring ---
    glow = frame.copy()
    cv2.polylines(glow, [polygon], isClosed=True, color=color, thickness=7)
    cv2.addWeighted(glow, 0.45, frame, 0.55, 0, frame)

    # --- Sharp inner edge on top ---
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=2)

    # --- Label: black shadow + white text ---
    text_x = int(polygon[:, 0].min())
    text_y = max(15, int(polygon[:, 1].min()) - 8)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)

    print("[INFO] Initializing Orbbec Camera...")
    cam = CameraInterface()
    cam.setup_streams()

    # Start background capture thread
    frame_q   = queue.Queue(maxsize=1)
    stop_evt  = threading.Event()
    cap_thread = threading.Thread(
        target=capture_worker,
        args=(cam, frame_q, stop_evt),
        daemon=True
    )
    cap_thread.start()

    print("[INFO] System ready. Press 'q' in the video window to exit.")

    frame_count = 0

    try:
        while True:
            # Block until the capture thread delivers a fresh frame
            color_frame = frame_q.get()
            raw_bgr = frame_to_bgr_image(color_frame)  # already BGR for OpenCV
            h, w = raw_bgr.shape[:2]

            # ── Frame-skip: run inference every other frame ──────────────
            # Comment this block out if your CPU is fast enough.
            frame_count += 1
            if frame_count % 2 != 0:
                cv2.imshow("Industrial Sorting Feed", raw_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── Inference ────────────────────────────────────────────────
            results = model.predict(
                source=raw_bgr,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                half=False,    # FP16 only helps with a CUDA GPU; leave False on CPU
                stream=False,  # single-frame input — stream generator adds overhead here
                verbose=False,
            )

            annotated = raw_bgr.copy()

            for result in results:
                if result.masks is None or len(result.boxes) == 0:
                    continue

                # ── KEY FIX: use .xy not .xyn ────────────────────────────
                # result.masks.xy  → list of (N,2) arrays already in the
                #                    original source-frame pixel coordinates.
                # result.masks.xyn → normalised [0-1]; manual * [w,h] can
                #                    produce a small but visible offset because
                #                    YOLO pads the image internally before
                #                    inference and xyn is relative to that
                #                    padded canvas, not your raw frame.
                for mask_xy, box in zip(result.masks.xy, result.boxes):
                    polygon = mask_xy.astype(np.int32)

                    # Skip tiny / partial detections at frame edges (ghost fix)
                    if cv2.contourArea(polygon) / (w * h) < MIN_MASK_AREA:
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