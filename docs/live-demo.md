# Live Demo

## Purpose

The live demo is the always-on inference UI for the conveyor belt stream. It draws the segmentation overlays on the live image and shows the current model status in the top bar.

## Run It

From the `rgbd/` folder:

```bash
make demo
```

## Default Model

The default model is `m26best.pt`.

The live UI includes a model selector so you can switch between:

- `m8best.pt`
- `m26best.pt`
- `m11best.pt`
- `s8best.pt`
- `s26best.pt`
- `s11best.pt`

## What the UI Shows

- live FPS
- number of detections
- current model name
- a model-mode indicator for single vs multiple detection
- an info drawer with model details
- object overlays with confidence labels next to each segmentation

## Color Meaning

The demo uses the object colors as the legend:

- blue = copper
- teal = steel

The object text labels are intentionally minimized so the color mapping stays clean on the screen.

## Useful Keys

- `S` - save a snapshot
- `R` - start/stop recording
- `Q` - quit the demo

The UI also exposes mouse-click controls for the model button and info button.
