import sys
import threading
import queue
import time
import os
from datetime import datetime
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
MIN_MASK_AREA  = 0.002

# Output directories
DIR_IMAGES = Path("live-data/images")
DIR_VIDEOS = Path("live-data/videos")
DIR_IMAGES.mkdir(parents=True, exist_ok=True)
DIR_VIDEOS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FRAME CONVERTER
# ─────────────────────────────────────────────────────────────
def orbbec_frame_to_bgr(frame) -> np.ndarray | None:
    w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
    data = frame.get_data()
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(
            np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)),
            cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        print(f"[ERROR] Unsupported color format: {fmt}")
        return None


# ─────────────────────────────────────────────────────────────
# LETTERBOX INVERSE — VERIFIED FOR 1280×720 → 640×640
#
# For a 1280×720 frame with model_imgsz=640:
#   scale  = min(640/1280, 640/720) = min(0.5, 0.8888) = 0.5
#   scaled = 640×360 (fits inside 640×640)
#   pad_x  = (640 - 640) / 2 = 0      ← no horizontal padding
#   pad_y  = (640 - 360) / 2 = 140    ← 140px top AND bottom
#
# So mask points are shifted DOWN by 140px in YOLO space.
# We subtract pad_y then divide by scale to get back to 1280×720.
# ─────────────────────────────────────────────────────────────
def unletterbox(polygon: np.ndarray, orig_h: int, orig_w: int,
                model_imgsz: int = 640) -> np.ndarray:
    scale = min(model_imgsz / orig_w, model_imgsz / orig_h)
    pad_x = (model_imgsz - orig_w * scale) / 2
    pad_y = (model_imgsz - orig_h * scale) / 2

    pts = polygon.astype(np.float64)
    pts[:, 0] = (pts[:, 0] - pad_x) / scale
    pts[:, 1] = (pts[:, 1] - pad_y) / scale
    pts[:, 0] = np.clip(pts[:, 0], 0, orig_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, orig_h - 1)
    return pts.astype(np.int32)


# ─────────────────────────────────────────────────────────────
# DEBUG: print the transform for the first frame so you can
# verify the numbers match the camera resolution in your logs
# ─────────────────────────────────────────────────────────────
def print_letterbox_debug(orig_h, orig_w, model_imgsz=640):
    scale = min(model_imgsz / orig_w, model_imgsz / orig_h)
    pad_x = (model_imgsz - orig_w * scale) / 2
    pad_y = (model_imgsz - orig_h * scale) / 2
    print(f"[LETTERBOX] orig={orig_w}×{orig_h}  model={model_imgsz}")
    print(f"[LETTERBOX] scale={scale:.6f}  pad_x={pad_x:.2f}  pad_y={pad_y:.2f}")


# ─────────────────────────────────────────────────────────────
# DRAWING — solid outline only, zero flicker
# ─────────────────────────────────────────────────────────────
def draw_detection(frame: np.ndarray, polygon: np.ndarray,
                   color: tuple, label: str) -> None:
    if len(polygon) < 3:
        return
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=3)
    tx = int(polygon[:, 0].min())
    ty = max(15, int(polygon[:, 1].min()) - 8)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# HUD OVERLAY
# Shows keybindings and live recording status on screen
# ─────────────────────────────────────────────────────────────
def draw_hud(frame: np.ndarray, recording: bool, video_writer,
             detection_count: int, fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS + detection count
    cv2.putText(frame, f"FPS: {fps:.1f}   Detections: {detection_count}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Recording indicator
    if recording:
        cv2.circle(frame, (w - 20, 18), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    # Key hints bottom bar
    hints = "[S] Screenshot   [R] Record video   [Q] Quit"
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, hints, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)


# ─────────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray, detections: list) -> str:
    """
    Save a single annotated frame as JPEG.
    Filename encodes timestamp + detected class names.
    Returns the saved path string.
    """
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    names = "_".join(sorted(set(detections))) if detections else "no_detection"
    path  = DIR_IMAGES / f"{ts}_{names}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SNAPSHOT] Saved → {path}")
    return str(path)


def make_video_writer(frame: np.ndarray) -> cv2.VideoWriter:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(DIR_VIDEOS / f"{ts}_detection.mp4")
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
    print(f"[RECORD] Started → {path}")
    return writer


# ─────────────────────────────────────────────────────────────
# BACKGROUND CAPTURE THREAD
# ─────────────────────────────────────────────────────────────
def capture_worker(cam, frame_q, stop_event):
    while not stop_event.is_set():
        color_frame, _ = cam.get_frames()
        if color_frame is None:
            continue
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
    model       = YOLO(MODEL_PATH)
    model_imgsz = 640   

    print("[INFO] Initializing Orbbec Camera...")
    cam = CameraInterface()
    cam.setup_streams()

    stop_evt   = threading.Event()
    frame_q    = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(target=capture_worker,
                                  args=(cam, frame_q, stop_evt), daemon=True)
    cap_thread.start()

    print("[INFO] System ready.")
    print("[INFO] S = screenshot   R = start/stop recording   Q = quit")

    # State variables
    frame_count    = 0
    video_writer   = None
    recording      = False
    fps_timer      = time.time()
    fps            = 0.0
    fps_frames     = 0

    # Cache variables to prevent flickering during skipped frames
    cached_polygons = []   # Format: (polygon, color, label)
    detected_classes = []  # For screenshots/HUD

    try:
        while True:
            color_frame = frame_q.get()
            bgr = orbbec_frame_to_bgr(color_frame)
            if bgr is None:
                continue

            orig_h, orig_w = bgr.shape[:2]
            frame_count += 1
            skip = (frame_count % 2 != 0)

            annotated = bgr.copy()

            # Only run heavy AI inference every 2nd frame
            if not skip:
                results = model.predict(
                    source=bgr,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=False,
                    stream=False,
                    verbose=False,
                )

                # Reset cache for new detections
                cached_polygons = []
                detected_classes = []

                for result in results:
                    if result.masks is None or len(result.boxes) == 0:
                        continue
                    
                    # result.masks.xy is ALREADY scaled to orig_w and orig_h by Ultralytics
                    for mask_xy, box in zip(result.masks.xy, result.boxes):
                        polygon = mask_xy.astype(np.int32)

                        # Filter out tiny noise artifacts
                        if cv2.contourArea(polygon) / (orig_w * orig_h) < MIN_MASK_AREA:
                            continue

                        class_id   = int(box.cls[0].item())
                        conf_score = box.conf[0].item()
                        name       = CLASS_NAMES.get(class_id, "Unknown")
                        label      = f"{name} {conf_score:.2f}"
                        color      = CLASS_COLORS.get(class_id, (0, 255, 0))

                        # Store in cache
                        cached_polygons.append((polygon, color, label))
                        detected_classes.append(name)

            # ── DRAWING (Happens EVERY frame using current or cached data) ──
            for polygon, color, label in cached_polygons:
                draw_detection(annotated, polygon, color, label)

            # ── FPS counter ──────────────────────────────────────────
            fps_frames += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps       = fps_frames / elapsed
                fps_timer = time.time()
                fps_frames = 0

            # ── HUD & Display ────────────────────────────────────────
            draw_hud(annotated, recording, video_writer, len(detected_classes), fps)

            if recording and video_writer is not None:
                video_writer.write(annotated)

            cv2.imshow("Industrial Sorting Feed", annotated)

            # ── Keyboard input ───────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_snapshot(annotated, detected_classes)
            elif key == ord('r'):
                if not recording:
                    video_writer = make_video_writer(annotated)
                    recording    = True
                else:
                    video_writer.release()
                    video_writer = None
                    recording    = False
                    print("[RECORD] Stopped.")

    except Exception as e:
        print(f"[ERROR] Inference loop crashed: {e}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        stop_evt.set()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Closed safely.")

if __name__ == "__main__":
    main()