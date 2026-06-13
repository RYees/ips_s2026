import cv2
import numpy as np
from pathlib import Path


def verify_mask_alignment(img_name):
    dataset_path = Path("/home/cpsstudent/Documents/ips_s2026/rgbd/dataset")

    rgb_path = dataset_path / "cropped_rgb" / f"{img_name}.png"
    mask_path = dataset_path / "masks" / f"{img_name}.png"

    if not rgb_path.exists() or not mask_path.exists():
        print("[ERROR] Missing file(s) for mask alignment check.")
        print(rgb_path)
        print(mask_path)
        return

    # Load images
    img = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("[ERROR] Failed to load RGB or mask image.")
        print(rgb_path)
        print(mask_path)
        return

    # Check shape equality
    if img.shape[:2] != mask.shape:
        print(
            f"[MISMATCH] Dimension failure! RGB: {img.shape[:2]} vs Mask: {mask.shape}"
        )
        return

    print(f"[SUCCESS] Dimensions match perfectly: {img.shape[:2]}")

    # Create a semi-transparent colored overlay (Green) over the object
    overlay = img.copy()
    overlay[mask > 0] = [0, 255, 0]  # Color mask pixels bright green

    # Blend the overlay with the original image (alpha transparency)
    verification_blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Draw the contour line around the mask boundary for a sharper visual cue
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(
            verification_blend, contours, -1, (0, 0, 255), 2
        )  # Red outline

    debug_dir = dataset_path / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{img_name}_mask_overlay.png"
    cv2.imwrite(str(out_path), verification_blend)
    print(f"[SAVED] Overlay written to: {out_path}")

    # Display the result to manually inspect alignment when running locally.
    window_name = f"Mask Verification: {img_name}"
    cv2.imshow(window_name, verification_blend)
    print("Press any key on the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    verify_mask_alignment("img1458")
