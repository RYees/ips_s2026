import sys
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

# Fix path routing to project root
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

# Safe output directory using User Home Directory
DIR_IMAGES = Path("live-data/images")
DIR_VIDEOS = Path("live-data/videos")
DIR_IMAGES.mkdir(parents=True, exist_ok=True)
DIR_VIDEOS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# OVERIDDEN FRAME STREAMING ENGINE (Fixes the offset bug)
# ─────────────────────────────────────────────────────────────
def get_aligned_live_frame(cam: CameraInterface) -> np.ndarray | None:
    """
    Bypasses the buggy caching order in the original script.
    Forces Orbbec SDK to actively compute the spatial 
    alignment matrix between depth and RGB lenses.
    """
    try:
        frames = cam.pipeline.wait_for_frames(1000)
        if not frames:
            return None
        
        # 1. Force alignment filter computation FIRST
        aligned = cam.align_filter.process(frames)
        if aligned is not None:
            aligned_frames = aligned.as_frame_set()
            color_frame = aligned_frames.get_color_frame()
        else:
            # Fallback directly to raw frame if alignment engine fails
            color_frame = frames.get_color_frame()
            
        if color_frame is None:
            return None
            
        # 2. Parse frame to BGR matrix
        w, h = color_frame.get_width(), color_frame.get_height()
        fmt = color_frame.get_format()
        data = color_frame.get_data()
        
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3)).copy()
        elif fmt in (OBFormat.MJPG, OBFormat.JPEG):
            return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
    except Exception as e:
        print(f"[LIVE ENGINE ERROR] Frame handling failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────
# CAPTURE THREAD WORKER
# ─────────────────────────────────────────────────────────────
def live_capture_worker(cam, frame_q, stop_event):
    while not stop_event.is_set():
        bgr_frame = get_aligned_live_frame(cam)
        if bgr_frame is None:
            continue
        
        # Drop stale frames to keep the queue real-time
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(bgr_frame)


# ─────────────────────────────────────────────────────────────
# UI DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────
def draw_detection(frame: np.ndarray, polygon: np.ndarray, color: tuple, label: str) -> None:
    if len(polygon) < 3:
        return
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=3)
    tx = int(polygon[:, 0].min())
    ty = max(15, int(polygon[:, 1].min()) - 8)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, recording: bool, detection_count: int, fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}   Detections: {detection_count}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if recording:
        cv2.circle(frame, (w - 20, 18), 8, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 60, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    hints = "[S] Screenshot   [R] Record video   [Q] Quit"
    cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, hints, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)


def save_snapshot(frame: np.ndarray, detections: list) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    names = "_".join(sorted(set(detections))) if detections else "no_detection"
    path = DIR_IMAGES / f"{ts}_{names}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SNAPSHOT] Saved → {path}")
    return str(path)


def make_video_writer(frame: np.ndarray) -> cv2.VideoWriter:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(DIR_VIDEOS / f"{ts}_detection.mp4")
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, 20.0, (w, h))


# ─────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
def main():
    print("[INFO] Loading Segmentation Model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Initializing Orbbec Camera via base Interface...")
    cam = CameraInterface()
    cam.setup_streams()

    stop_evt = threading.Event()
    frame_q = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(target=live_capture_worker, args=(cam, frame_q, stop_evt), daemon=True)
    cap_thread.start()

    print("[INFO] Live tracking interface operational.")

    # Application state
    frame_count = 0
    video_writer = None
    recording = False
    fps_timer = time.time()
    fps = 0.0
    fps_frames = 0

    # Persistence cache variables to eliminate drawing stutters
    cached_polygons = []
    detected_classes = []

    try:
        while True:
            bgr = frame_q.get()
            orig_h, orig_w = bgr.shape[:2]

            frame_count += 1
            skip = (frame_count % 2 != 0)
            annotated = bgr.copy()

            if not skip:
                results = model.predict(
                    source=bgr,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    half=False,
                    stream=False,
                    verbose=False,
                )

                cached_polygons = []
                detected_classes = []

                for result in results:
                    if result.masks is None or len(result.boxes) == 0:
                        continue
                    
                    for mask_xy, box in zip(result.masks.xy, result.boxes):
                        polygon = mask_xy.astype(np.int32)

                        if cv2.contourArea(polygon) / (orig_w * orig_h) < MIN_MASK_AREA:
                            continue

                        class_id = int(box.cls[0].item())
                        conf_score = box.conf[0].item()
                        name = CLASS_NAMES.get(class_id, "Unknown")
                        label = f"{name} {conf_score:.2f}"
                        color = CLASS_COLORS.get(class_id, (0, 255, 0))

                        cached_polygons.append((polygon, color, label))
                        detected_classes.append(name)

            # Draw visual tracking tracking layers frame-by-frame smoothly
            for polygon, color, label in cached_polygons:
                draw_detection(annotated, polygon, color, label)

            # Track processing speed
            fps_frames += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_timer = time.time()
                fps_frames = 0

            draw_hud(annotated, recording, len(detected_classes), fps)

            if recording and video_writer is not None:
                video_writer.write(annotated)

            cv2.imshow("Industrial Sorting Feed - Aligned Mode", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_snapshot(annotated, detected_classes)
            elif key == ord('r'):
                if not recording:
                    video_writer = make_video_writer(annotated)
                    recording = True
                    print("[RECORD] Streaming feed recording started.")
                else:
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print("[RECORD] Streaming feed recording stopped.")

    except Exception as e:
        print(f"[SYSTEM CRASH] Error in live loop: {e}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        stop_evt.set()
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] Camera resources released and windows destroyed clean.")


if __name__ == "__main__":
    main()