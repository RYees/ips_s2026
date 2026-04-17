# Orbbec RGB-D App Fix Notes

## Summary
This document records the recent fix for the Orbbec RGB-D capture app in `pyorbbec-rgbd/examples/rgbd`.
The application was crashing on startup, the camera stream was dropping frames, and the label-saving behavior needed verification.

## Problem
- The GUI app failed with a fatal error: `RGBDCollectorApp object has no attribute 'Q'`.
- The `main.py` startup code called `self.Q()` but the method was missing.
- The camera was being accessed simultaneously by `OrbbecViewer` and the RGB-D app, and the USB cable was unreliable.
- That combination caused skipped frames, dropped depth or color frames, and intermittent frame sync failures.
- A crop helper could produce invalid or empty crops, causing unstable capture/segmentation behavior.
- The dataset label file format and saved class value needed verification.

## Fixes Applied
### 1. Fix startup loop method
- Added `def Q(self): self.update_video()` to `main.py`.
- This restores the periodic GUI refresh and prevents the startup crash from `self.Q()` being undefined.

### 2. Improve frame handling when the camera is unstable
- Updated `camera_interface.py` to handle pipeline frame drops more gracefully.
- `get_frames()` now retries `pipeline.wait_for_frames()` with a longer timeout when no frames arrive.
- It logs warnings instead of failing immediately when frames are missing.
- Cached the last successful color and depth frames and reused them when timestamps were still close.
- Added a fallback that safely processes alignment output and checks for missing aligned color/depth frames.
- This helps the app survive skipped frames caused by a bad cable or device contention from `OrbbecViewer`.

### 3. Make image cropping safe
- Updated `crop_manual()` in `main.py` to return the original image when crop bounds are invalid.
- Guarded against `top + bottom >= height` or `left + right >= width`.
- This avoids empty arrays and downstream segmentation issues.

### 4. Validate label saving behavior
- Verified that saved label files in `dataset/labels/` begin with the class value from the UI.
- Example: `img0000.txt` starts with `0`, which corresponds to the selected label value.
- Confirmed the annotation writer writes `label_class` followed by normalized polygon coordinates.

## Files Changed
- `pyorbbec-rgbd/examples/rgbd/main.py`
  - Added `Q()` method.
  - Added crop-bound safety in `crop_manual()`.
- `pyorbbec-rgbd/examples/rgbd/camera_interface.py`
  - Added frame retry logic, cached frame fallback, and safer alignment handling.
- `pyorbbec-rgbd/examples/rgbd/annotation_writer.py`
  - Verified label format and data writing logic.

## Notes
- The app now starts successfully and displays the live RGB-D feed.
- The save sequence writes:
  - `dataset/images/imgXXXX.png`
  - `dataset/depth/imgXXXX.png`
  - `dataset/labels/imgXXXX.txt`
  - `dataset/pointcloud/imgXXXX.ply`
  - `dataset/info/imgXXXX.txt`
- If the label mapping needs to be changed, update the UI text and the selected values in `main.py`.

## Recommended next step
- Test several captures to confirm `0` and `1` labels are being stored as expected.
- If needed, adjust the dropdown mapping in `main.py` to swap Copper/Steel label semantics.
