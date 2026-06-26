# Models

## Current Live Demo Weights

The live demo currently supports six selectable weights:

- `m8best.pt`
- `m26best.pt`
- `m11best.pt`
- `s8best.pt`
- `s26best.pt`
- `s11best.pt`

The default is `m26best.pt`.

## Naming for GitHub Release Assets

If you publish the final weights as GitHub Release assets, a cleaner naming scheme is:

- `best_single.pt`
- `best_multiple.pt`
- `best_multiple_depth.pt`

If you use those names, update the model map in `rgbd/live/live_inference.py` so the demo can load them.

## Where to Place Them

Keep the model files next to the live demo code or update the model directory path in the live demo configuration.

The important thing is that the filenames in the code match the filenames on disk.
