import sys
import threading
import queue
import time
import atexit
import gc
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
MODEL_DIR = Path("/home/cpsstudent/Documents/ips_s2026/rgdb/live")
MODEL_FILES = {
    "m8best": MODEL_DIR / "m8best.pt",
    "m26best": MODEL_DIR / "m26best.pt",
    "m11best": MODEL_DIR / "m11best.pt",
    "s8best": MODEL_DIR / "s8best.pt",
    "s26best": MODEL_DIR / "s26best.pt",
    "s11best": MODEL_DIR / "s11best.pt",
}
MODEL_SWITCH_KEYS = list(MODEL_FILES.keys())
DEFAULT_MODEL_KEY = "m8best"

CLASS_NAMES = {0: "Copper", 1: "Steel"}
CLASS_COLORS = {0: (255, 0, 0), 1: (180, 180, 0)}  # Copper blue, steel teal-ish
WINDOW_NAME = "Industrial Sorting Feed"
INFO_PANEL_W = 360

CONF_THRESHOLD = 0.30  # lowered — new model is more conservative
IOU_THRESHOLD = 0.40
MIN_MASK_AREA = 0.002

# Must match capture_crop in main.py exactly
# img[0:720, 250:760] → 510px wide — the space the model was trained on
CROP_LEFT = 250
CROP_RIGHT = 760

DIR_IMAGES = Path("live-data/images")
DIR_VIDEOS = Path("live-data/videos")
DIR_LOGS = Path("live-data/logs")
DIR_IMAGES.mkdir(parents=True, exist_ok=True)
DIR_VIDEOS.mkdir(parents=True, exist_ok=True)
DIR_LOGS.mkdir(parents=True, exist_ok=True)

cam_instance = None


def model_mode_label(model_key: str) -> str:
    if model_key.startswith("s"):
        return "Single Detection"
    return "Multi Detection"


def model_family_label(model_key: str) -> str:
    if model_key.startswith("m"):
        return "Multi"
    if model_key.startswith("s"):
        return "Single"
    return "Unknown"


def model_backbone_label(model_key: str) -> str:
    if "26" in model_key:
        return "YOLO26"
    if "11" in model_key:
        return "YOLO11"
    if "8" in model_key:
        return "YOLO8"
    return "Unknown"


def create_camera_interface_with_retry(attempts: int = 5, delay_s: float = 0.8):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return CameraInterface()
        except Exception as exc:
            last_exc = exc
            print(f"[WARNING] Camera open failed on attempt {attempt}/{attempts}: {exc}")
            gc.collect()
            if attempt < attempts:
                time.sleep(delay_s)
    raise RuntimeError(
        "Unable to open Orbbec camera after multiple attempts. "
        "Make sure no other process is using the device and try again."
    ) from last_exc


def create_fps_log_file() -> tuple[Path, object]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DIR_LOGS / f"live_fps_{ts}.log"
    fh = path.open("a", encoding="utf-8")
    return path, fh


def log_line(fh, message: str) -> None:
    print(message)
    fh.write(message + "\n")
    fh.flush()


def load_model_by_key(model_key: str):
    if model_key not in MODEL_FILES:
        raise KeyError(f"Unknown model key: {model_key}")
    model_path = MODEL_FILES[model_key]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(str(model_path)), model_path


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


def draw_confidence_tag(frame: np.ndarray, polygon: np.ndarray, confidence: float) -> None:
    if len(polygon) < 3:
        return
    x, y, w, h = cv2.boundingRect(polygon)
    label = f"{confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    tx = max(0, min(x, frame.shape[1] - tw - 8))
    ty = max(th + 6, y - 6)

    cv2.rectangle(
        frame,
        (tx - 3, ty - th - 4),
        (tx + tw + 3, ty + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (tx, ty),
        font,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_navbar(
    frame: np.ndarray,
    recording: bool,
    detection_count: int,
    fps: float,
    info_open: bool,
    model_key: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    top_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.rectangle(frame, (0, 0), (w, top_h), (0, 110, 124), 1)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}   Detections: {detection_count}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
    )

    def chip_width(label: str) -> int:
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        return 16 + 8 + tw + 10

    def chip(x, label, color):
        box_w = 16
        box_h = 16
        y0 = 13
        cv2.rectangle(frame, (x, y0), (x + box_w, y0 + box_h), color, -1)
        cv2.rectangle(frame, (x, y0), (x + box_w, y0 + box_h), (255, 255, 255), 1)
        cv2.putText(
            frame,
            label,
            (x + box_w + 8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    btn_w, btn_h = 64, 22
    btn_x, btn_y = w - btn_w - 10, 10

    model_btn_w, model_btn_h = 72, 22
    model_btn_x = btn_x - 10 - model_btn_w
    model_btn_y = 10

    steel_x = btn_x - 10 - chip_width("Steel")
    copper_x = steel_x - 12 - chip_width("Copper")
    mode_label = model_mode_label(model_key)
    mode_x = copper_x - 12 - chip_width(mode_label)
    chip(mode_x, mode_label, (0, 150, 180))
    chip(copper_x, "Copper", CLASS_COLORS[0])
    chip(steel_x, "Steel", CLASS_COLORS[1])

    if recording:
        rec_x = max(180, mode_x - 76)
        cv2.circle(frame, (rec_x, 21), 6, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "REC",
            (rec_x + 12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    model_overlay = frame.copy()
    cv2.rectangle(
        model_overlay,
        (model_btn_x, model_btn_y),
        (model_btn_x + model_btn_w, model_btn_y + model_btn_h),
        (45, 45, 55),
        -1,
    )
    cv2.addWeighted(model_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(
        frame,
        (model_btn_x, model_btn_y),
        (model_btn_x + model_btn_w, model_btn_y + model_btn_h),
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "Model",
        (model_btn_x + 11, model_btn_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    btn_overlay = frame.copy()
    cv2.rectangle(btn_overlay, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 110, 124), -1)
    cv2.addWeighted(btn_overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), 1)
    cv2.putText(
        frame,
        "Info" if not info_open else "Close",
        (btn_x + 8, btn_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if recording:
        pass
    cv2.rectangle(overlay, (0, h - 34), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (0, h - 34), (w, h), (0, 110, 124), 1)
    cv2.putText(
        frame,
        "[S] Screenshot   [R] Record   [Q] Quit",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        2,
    )
    return (btn_x, btn_y, btn_w, btn_h), (model_btn_x, model_btn_y, model_btn_w, model_btn_h)


def draw_info_panel(frame: np.ndarray, panel_w: int, model_key: str) -> None:
    if panel_w <= 0:
        return
    h, w = frame.shape[:2]
    x1 = w - panel_w
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (w, h), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.rectangle(frame, (x1, 0), (w, h), (255, 255, 255), 1)
    cv2.rectangle(frame, (x1, 0), (w, 42), (0, 110, 124), 1)
    cv2.putText(
        frame,
        "Model Info",
        (x1 + 14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    lines = [
        ("Model", Path(MODEL_FILES.get(model_key, MODEL_FILES[DEFAULT_MODEL_KEY])).name),
        ("Family", model_family_label(model_key)),
        ("Backbone", model_backbone_label(model_key)),
        ("Mode", model_mode_label(model_key)),
        ("Task", "segmentation"),
        ("Crop", f"x=[{CROP_LEFT}:{CROP_RIGHT}]"),
        ("IoU", f"{IOU_THRESHOLD:.2f}"),
        ("Copper", "blue outline"),
        ("Steel", "teal outline"),
        ("View", "outline-only overlay"),
        ("Controls", "1-6 models, S save, R record, Q quit"),
    ]

    y = 72
    for title, value in lines:
        cv2.putText(
            frame,
            f"{title}:",
            (x1 + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (155, 205, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value,
            (x1 + 122, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += 28


def draw_model_menu(frame: np.ndarray, button_rect: tuple[int, int, int, int], open_menu: bool) -> list[tuple[int, int, int, int]]:
    if not open_menu:
        return []

    bx, by, bw, bh = button_rect
    item_w = 132
    item_h = 26
    x = max(10, bx + bw - item_w)
    y = by + bh + 6
    rects = []

    overlay = frame.copy()
    menu_h = item_h * len(MODEL_SWITCH_KEYS) + 6
    cv2.rectangle(overlay, (x, y), (x + item_w, y + menu_h), (18, 20, 28), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    cv2.rectangle(frame, (x, y), (x + item_w, y + menu_h), (255, 255, 255), 1)

    for idx, model_key in enumerate(MODEL_SWITCH_KEYS):
        iy = y + 3 + idx * item_h
        rect = (x + 2, iy, item_w - 4, item_h - 2)
        rects.append(rect)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (35, 39, 50), -1)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (70, 74, 90), 1)
        label = f"{model_backbone_label(model_key)}  {model_family_label(model_key)}"
        cv2.putText(
            frame,
            label,
            (rect[0] + 8, rect[1] + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return rects


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

    active_model_key = DEFAULT_MODEL_KEY
    fps_log_path, fps_log_fh = create_fps_log_file()
    model, active_model_path = load_model_by_key(active_model_key)

    log_line(fps_log_fh, f"[INFO] FPS log: {fps_log_path}")
    log_line(fps_log_fh, f"[INFO] Loading model: {active_model_path.name}")

    log_line(fps_log_fh, "[INFO] Initializing camera...")
    cam_instance = create_camera_interface_with_retry()
    cam_instance.setup_streams()

    stop_evt = threading.Event()
    frame_q = queue.Queue(maxsize=1)
    cap_thread = threading.Thread(
        target=capture_worker, args=(cam_instance, frame_q, stop_evt), daemon=True
    )
    cap_thread.start()

    log_line(fps_log_fh, "[INFO] Ready.  S=screenshot  R=record  Q=quit")
    log_line(
        fps_log_fh,
        f"[INFO] Inference crop: x=[{CROP_LEFT}:{CROP_RIGHT}]  "
        f"({CROP_RIGHT - CROP_LEFT}px wide — matches training data)",
    )

    frame_count = 0
    video_writer = None
    recording = False
    fps_timer = time.time()
    fps = 0.0
    fps_frames = 0
    inference_time_sum = 0.0
    inference_runs = 0
    cached_polygons = []
    detected_classes = []
    ui_state = {
        "info_open": False,
        "info_width": 0,
        "info_btn_rect": (0, 0, 0, 0),
        "model_btn_rect": (0, 0, 0, 0),
        "model_menu_open": False,
        "model_menu_item_rects": [],
        "model_key": active_model_key,
        "model_path": active_model_path,
    }

    def on_mouse(event, x, y, flags, param):
        nonlocal model
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        mx, my, mw, mh = ui_state["model_btn_rect"]
        if mx <= x <= mx + mw and my <= y <= my + mh:
            ui_state["model_menu_open"] = not ui_state["model_menu_open"]
            return

        if ui_state["model_menu_open"]:
            for idx, rect in enumerate(ui_state["model_menu_item_rects"]):
                rx, ry, rw, rh = rect
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    next_key = MODEL_SWITCH_KEYS[idx]
                    ui_state["model_key"] = next_key
                    ui_state["model_path"] = MODEL_FILES[next_key]
                    log_line(fps_log_fh, f"[INFO] Switching model to {ui_state['model_path'].name}...")
                    model, _ = load_model_by_key(next_key)
                    cached_polygons.clear()
                    detected_classes.clear()
                    ui_state["model_menu_open"] = False
                    log_line(fps_log_fh, f"[INFO] Model loaded: {ui_state['model_path'].name}")
                    return

        bx, by, bw, bh = ui_state["info_btn_rect"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            ui_state["info_open"] = not ui_state["info_open"]
            print(f"[UI] Info panel {'opened' if ui_state['info_open'] else 'closed'}.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

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

            # Run inference on every frame so stale detections clear quickly
            infer_start = time.perf_counter()
            results = model.predict(
                source=bgr_crop,  # 510px crop — same as training
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                half=False,
                stream=False,
                verbose=False,
            )
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            inference_time_sum += infer_ms
            inference_runs += 1

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
                    confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
                    cached_polygons.append((polygon, color, confidence))
                    detected_classes.append(name)

            # Draw cached polygons (stable on skipped frames — no flicker)
            for polygon, color, confidence in cached_polygons:
                draw_detection(annotated, polygon, color, thickness=4)
                draw_confidence_tag(annotated, polygon, confidence)

            # FPS counter
            fps_frames += 1
            if (elapsed := time.time() - fps_timer) >= 1.0:
                fps = fps_frames / elapsed
                avg_infer_ms = (inference_time_sum / inference_runs) if inference_runs else 0.0
                live_msg = (
                    f"[FPS] live={fps:.2f} fps | infer_avg={avg_infer_ms:.1f} ms "
                    f"| detections={len(detected_classes)} | model={ui_state['model_path'].name}"
                )
                log_line(fps_log_fh, live_msg)
                fps_timer = time.time()
                fps_frames = 0
                inference_time_sum = 0.0
                inference_runs = 0

            target_w = INFO_PANEL_W if ui_state["info_open"] else 0
            if ui_state["info_width"] < target_w:
                ui_state["info_width"] = min(target_w, ui_state["info_width"] + 28)
            elif ui_state["info_width"] > target_w:
                ui_state["info_width"] = max(target_w, ui_state["info_width"] - 28)

            draw_info_panel(annotated, ui_state["info_width"], ui_state["model_key"])
            ui_state["info_btn_rect"], ui_state["model_btn_rect"] = draw_navbar(
                annotated,
                recording,
                len(detected_classes),
                fps,
                ui_state["info_open"],
                ui_state["model_key"],
            )
            ui_state["model_menu_item_rects"] = draw_model_menu(
                annotated, ui_state["model_btn_rect"], ui_state["model_menu_open"]
            )

            if recording and video_writer is not None:
                video_writer.write(annotated)

            cv2.imshow(WINDOW_NAME, annotated)

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

    except KeyboardInterrupt:
        log_line(fps_log_fh, "\n[INFO] Stopped by user.")
    except Exception as e:
        log_line(fps_log_fh, f"[ERROR] {e}")
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        stop_evt.set()
        cleanup_camera_hardware()
        try:
            fps_log_fh.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
