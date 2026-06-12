import cv2
import numpy as np
from pathlib import Path
import re


def test_metadata_alignment(base_dir: Path, asset_name: str):
    # 1. Point directly to your images subdirectory
    img_path = base_dir / "images" / f"{asset_name}.png"

    if not img_path.exists():
        print(f"[ERROR] Could not find the image at: {img_path}")
        return

    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    print(f"\n==================================================")
    print(f"Loaded Realtime Image Resolution: {w}x{h} px")

    # 2. Find and parse the matching metadata file
    txt_path = base_dir / "info" / f"{asset_name}.txt"
    if not txt_path.exists():
        txt_path = base_dir / f"{asset_name}.txt"

    if not txt_path.exists():
        print(f"[ERROR] Missing metadata configuration file at: {txt_path}")
        return

    lines = txt_path.read_text().splitlines()
    cx, cy = None, None
    crop_left, crop_top = 0, 0

    for i, line in enumerate(lines):
        if "Intrinsics matrix:" in line:
            r0 = [float(x) for x in re.findall(r"-?[\d]+\.[\d]+", lines[i + 1])]
            r1 = [float(x) for x in re.findall(r"-?[\d]+\.[\d]+", lines[i + 2])]
            cx = r0[2]
            cy = r1[2]
        elif "Crop:" in line or "crop:" in line:
            v = [int(x) for x in re.findall(r"\d+", line)]
            if len(v) >= 4:
                crop_top, crop_left = v[0], v[2]

    print(
        f"Parsed Config -> cx: {cx}, cy: {cy} | crop_left: {crop_left}, crop_top: {crop_top}"
    )

    if cx is None or cy is None:
        print("[ERROR] Pinhole matrix parameters completely missing from txt file.")
        return

    # Create diagnostic view copy
    diagnostic_view = img.copy()

    # 3. Draw Image Visual Center (GREEN Crosshair)
    canvas_cx = w // 2
    canvas_cy = h // 2
    cv2.line(diagnostic_view, (canvas_cx, 0), (canvas_cx, h), (0, 255, 0), 2)
    cv2.line(diagnostic_view, (0, canvas_cy), (w, canvas_cy), (0, 255, 0), 2)
    cv2.putText(
        diagnostic_view,
        f"Image Visual Center ({canvas_cx}, {canvas_cy})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # 4. Draw Camera Mathematical Projected Center (RED Crosshair)
    math_cx = int(cx - crop_left)
    math_cy = int(cy - crop_top)

    cv2.line(diagnostic_view, (math_cx, 0), (math_cx, h), (0, 0, 255), 2)
    cv2.line(diagnostic_view, (0, math_cy), (w, math_cy), (0, 0, 255), 2)
    cv2.circle(diagnostic_view, (math_cx, math_cy), 10, (0, 0, 255), -1)
    cv2.putText(
        diagnostic_view,
        f"Matrix Projected Center ({math_cx}, {math_cy})",
        (20, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # Save output back into the images folder for easy finding
    output_path = base_dir / "images" / f"{asset_name}_alignment_diagnostics.png"
    cv2.imwrite(str(output_path), diagnostic_view)
    print(f"[SUCCESS] Alignment diagnostic map saved to: {output_path}")
    print(f"==================================================\n")


if __name__ == "__main__":
    samples_directory = Path(
        "/Users/rzapp/Documents/A{sp}A/ips_s2026/offline_case/samples"
    )
    test_metadata_alignment(base_dir=samples_directory, asset_name="img0002")
