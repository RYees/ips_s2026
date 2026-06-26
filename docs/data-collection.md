# Data Collection

## Purpose

The data collection UI is the capture workflow that saves a full package per sample:

- cropped RGB
- uncropped RGB
- depth image
- mask
- YOLO label
- point cloud
- metadata text
- debug artifacts

This workflow uses the shared `offline_case/` masking helpers from the full CPS workspace checkout.

## Run It

From the `rgbd/` folder:

```bash
make run
```

## Save Workflow

1. Capture a frame.
2. Review the preview panels.
3. Choose the class label.
4. Save the snapshot.

The app only commits the package if every required artifact is valid. If the mask contour or point cloud is missing, the save is aborted so you do not end up with half-written samples.

## Output Tree

The collector writes into `rgbd/data/`:

```text
rgbd/data/
  images/
  uncropped_rgb/
  depth/
  masks/
  labels/
  info/
  pointcloud/
  debug/
```

## Notes

- `images/` contains the cropped RGB images used for training.
- `uncropped_rgb/` keeps the raw full-frame capture for recovery and review.
- `info/` stores the per-sample text summary.
- `debug/` stores the live log files and mask inspection images.

See `ui-controls.md` for the exact buttons and hotkeys.
