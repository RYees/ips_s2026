import sys
import threading
import queue
import time
import atexit
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
MODEL_PATH = "/home/cpsstudent/Documents/ips_s2026/rgdb/live/mbest.pt"

CLASS_NAMES = {0: "Copper", 1: "Steel"}
CLASS_COLORS = {0: (255, 0, 0), 1: (180, 180, 0)}  # Copper blue, steel teal-ish

THEMES = {
    "industrial": {
        "name": "Industrial Control Room",
        "bg": (22, 26, 30),
        "panel": (36, 40, 46),
        "panel_alt": (28, 32, 38),
        "text": (245, 245, 245),
        "muted": (178, 186, 196),
        "accent": (0, 179, 199),
        "accent_soft": (0, 110, 124),
        "card": (18, 21, 25),
        "card_line": (255, 255, 255),
    },
    "clean": {
        "name": "Modern Clean Dashboard",
        "bg": (239, 243, 247),
        "panel": (255, 255, 255),
        "panel_alt": (246, 249, 252),
        "text": (28, 36, 45),
        "muted": (90, 102, 117),
        "accent": (235, 118, 52),
        "accent_soft": (208, 85, 25),
        "card": (250, 251, 253),
        "card_line": (220, 226, 233),
    },
    "hud": {
        "name": "High-Tech HUD",
        "bg": (7, 10, 19),
        "panel": (10, 16, 30),
        "panel_alt": (6, 12, 22),
        "text": (231, 248, 255),
        "muted": (141, 194, 227),
        "accent": (0, 245, 255),
        "accent_soft": (130, 90, 255),
        "card": (4, 8, 16),
        "card_line": (0, 245, 255),
    },
}
THEME_KEYS = list(THEMES.keys())
THEME_LABELS = {key: val["name"] for key, val in THEMES.items()}

CONF_THRESHOLD = 0.30  # lowered — new model is more conservative
IOU_THRESHOLD = 0.40
MIN_MASK_AREA = 0.002

# Must match capture_crop in main.py exactly
# img[0:720, 250:760] → 510px wide — the space the model was trained on
CROP_LEFT = 250
CROP_RIGHT = 760

DIR_IMAGES = Path("live-data/images")
DIR_VIDEOS = Path("live-data/videos")
DIR_IMAGES.mkdir(parents=True, exist_ok=True)
DIR_VIDEOS.mkdir(parents=True, exist_ok=True)

cam_instance = None


# ─────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────
def cleanup_camera_hardware():
    global cam_instance
    if cam_instance is not None:
        try:
            cam_instance.stop()
        except Exception:
            pass
        cam_instance = None


atexit.register(cleanup_camera_hardware)


# ─────────────────────────────────────────────────────────────
# FRAME CONVERTER
# ─────────────────────────────────────────────────────────────
def orbbec_frame_to_bgr(frame) -> np.ndarray | None:
    vf = frame.as_video_frame()
    fmt = vf.get_format()
    w, h = vf.get_width(), vf.get_height()
    data = np.frombuffer(vf.get_data(), dtype=np.uint8)

    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        return data.reshape((h, w, 3)).copy()
    elif fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(
            cv2.cvtColor(data.reshape((h, w, 2)), cv2.COLOR_YUV2RGB_YUYV),
            cv2.COLOR_RGB2BGR,
        )
    print(f"[ERROR] Unsupported format: {fmt}")
    return None


# ─────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────
def draw_detection(
    frame: np.ndarray, polygon: np.ndarray, color: tuple, thickness: int = 3
) -> None:
    if len(polygon) < 3:
        return
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)


def draw_legend(frame: np.ndarray, theme: dict) -> None:
    """Draw a persistent legend card so the object outlines stay uncluttered."""
    h, w = frame.shape[:2]
    box_w, box_h = 250, 110
    x1, y1 = 14, 46
    x2, y2 = x1 + box_w, y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme["card"], -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), theme["card_line"], 1)

    cv2.putText(
        frame,
        "Legend",
        (x1 + 10, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        theme["text"],
        1,
        cv2.LINE_AA,
    )

    row_y = y1 + 48
    for cls_id, name in [(0, "Copper"), (1, "Steel")]:
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(frame, (x1 + 10, row_y - 12), (x1 + 32, row_y + 8), color, -1)
        cv2.rectangle(
            frame, (x1 + 10, row_y - 12), (x1 + 32, row_y + 8), theme["text"], 1
        )
        cv2.putText(
            frame,
            f"{name}",
            (x1 + 42, row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            theme["text"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "blue" if cls_id == 0 else "teal",
            (x2 - 62, row_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            theme["muted"],
            1,
            cv2.LINE_AA,
        )
        row_y += 24


def draw_hud(
    frame: np.ndarray,
    recording: bool,
    detection_count: int,
    fps: float,
    theme: dict,
    theme_label: str,
) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), theme["panel"], -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (0, 0), (w, 42), theme["accent_soft"], 1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}   Detections: {detection_count}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        theme["text"],
        1,
    )
    cv2.putText(
        frame,
        theme_label,
        (w - 310, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        theme["muted"],
        1,
    )
    if recording:
        cv2.circle(frame, (w - 20, 18), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 60, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )
    cv2.rectangle(overlay, (0, h - 34), (w, h), theme["panel"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (0, h - 34), (w, h), theme["accent_soft"], 1)
    cv2.putText(
        frame,
        "[S] Screenshot   [R] Record   [T] Theme   [Q] Quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        theme["text"],
        1,
    )


def draw_theme_badge(frame: np.ndarray, theme: dict) -> None:
    badge_w, badge_h = 320, 34
    x1, y1 = 14, 86
    x2, y2 = x1 + badge_w, y1 + badge_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme["panel_alt"], -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), theme["accent"], 1)
    cv2.putText(
        frame,
        f"Theme: {theme['name']}",
        (x1 + 10, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        theme["text"],
        1,
        cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────
# SAVE HELPERS
# ─────────────────────────────────────────────────────────────
def save_snapshot(frame: np.ndarray, detections: list) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    names = "_".join(sorted(set(detections))) if detections else "no_detection"
    path = DIR_IMAGES / f"{ts}_{names}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[SNAPSHOT] → {path}")


def make_video_writer(frame: np.ndarray) -> cv2.VideoWriter:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = str(DIR_VIDEOS / f"{ts}_detection.mp4")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    print(f"[RECORD] Started → {path}")
    return writer


# ─────────────────────────────────────────────────────────────
# BACKGROUND CAPTURE THREAD
# ─────────────────────────────────────────────────────────────
def capture_worker(cam, frame_q, stop_event):
    while not stop_event.is_set():
        color_frame, _ = cam.get_frames()
        if color_frame is None:
            time.sleep(0.001)
            continue
        bgr = orbbec_frame_to_bgr(color_frame)
        if bgr is None:
            continue
        if not frame_q.empty():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(bgr)
        time.sleep(0.005)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    global cam_instance

    print("[INFO] Loading model...")
    model = YOLO(MODEL_PATH)

    print("[INFO] Initializing camera...")
    cam_instance = CameraInterface()
    cam_instance.setup_streams()

    stop_evt = threading.Event()
    frame_q = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(
        target=capture_worker, args=(cam_instance, frame_q, stop_evt), daemon=True
    )
    cap_thread.start()

    print("[INFO] Ready.  S=screenshot  R=record  Q=quit")
    print(
        f"[INFO] Inference crop: x=[{CROP_LEFT}:{CROP_RIGHT}]  "
        f"({CROP_RIGHT - CROP_LEFT}px wide — matches training data)"
    )

    frame_count = 0
    video_writer = None
    recording = False
    theme_idx = 0
    theme_key = THEME_KEYS[theme_idx]
    fps_timer = time.time()
    fps = 0.0
    fps_frames = 0
    cached_polygons = []
    detected_classes = []

    try:
        while True:
            # Always grab the freshest frame
            bgr = None
            while not frame_q.empty():
                try:
                    bgr = frame_q.get_nowait()
                except queue.Empty:
                    break
            if bgr is None:
                bgr = frame_q.get()

            # ── Crop to training coordinate space ────────────────────
            # Model was trained on img[:, 250:760] — feed it the same
            # region so positions learned during training map correctly
            bgr_crop = bgr[:, CROP_LEFT:CROP_RIGHT]
            crop_h, crop_w = bgr_crop.shape[:2]  # 720 × 510

            frame_count += 1
            annotated = bgr.copy()  # draw on the FULL frame for display

            # Run inference every other frame; reuse cache on skipped frames
            if frame_count % 2 == 0:
                results = model.predict(
                    source=bgr_crop,  # 510px crop — same as training
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
                        # Ultralytics returns mask polygons in the original
                        # source-image coordinate system for the crop we pass
                        # into predict(). Do not apply an extra unletterbox step
                        # here, or the polygon will drift left/up.
                        polygon = mask_xy.astype(np.int32)

                        # Filter tiny / ghost detections
                        if cv2.contourArea(polygon) / (crop_w * crop_h) < MIN_MASK_AREA:
                            continue

                        # Shift x from crop space → full frame space for drawing
                        polygon[:, 0] += CROP_LEFT

                        class_id = int(box.cls[0].item())
                        name = CLASS_NAMES.get(class_id, "Unknown")
                        color = CLASS_COLORS.get(class_id, (0, 255, 0))
                        cached_polygons.append((polygon, color))
                        detected_classes.append(name)

            # Draw cached polygons (stable on skipped frames — no flicker)
            for polygon, color in cached_polygons:
                draw_detection(annotated, polygon, color, thickness=4)

            # FPS counter
            fps_frames += 1
            if (elapsed := time.time() - fps_timer) >= 1.0:
                fps = fps_frames / elapsed
                fps_timer = time.time()
                fps_frames = 0

            theme = THEMES[theme_key]
            draw_hud(
                annotated,
                recording,
                len(detected_classes),
                fps,
                theme,
                theme["name"],
            )
            draw_theme_badge(annotated, theme)
            draw_legend(annotated, theme)

            if recording and video_writer is not None:
                video_writer.write(annotated)

            cv2.imshow("Industrial Sorting Feed", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_snapshot(annotated, detected_classes)
            elif key == ord("r"):
                if not recording:
                    video_writer = make_video_writer(annotated)
                    recording = True
                else:
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print("[RECORD] Stopped.")
            elif key == ord("t"):
                theme_idx = (theme_idx + 1) % len(THEME_KEYS)
                theme_key = THEME_KEYS[theme_idx]
                print(f"[THEME] Switched to {THEMES[theme_key]['name']}")
            elif key == ord("1"):
                theme_idx = THEME_KEYS.index("industrial")
                theme_key = "industrial"
                print(f"[THEME] Switched to {THEMES[theme_key]['name']}")
            elif key == ord("2"):
                theme_idx = THEME_KEYS.index("clean")
                theme_key = "clean"
                print(f"[THEME] Switched to {THEMES[theme_key]['name']}")
            elif key == ord("3"):
                theme_idx = THEME_KEYS.index("hud")
                theme_key = "hud"
                print(f"[THEME] Switched to {THEMES[theme_key]['name']}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        stop_evt.set()
        cleanup_camera_hardware()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
