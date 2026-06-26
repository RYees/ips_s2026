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

The live demo expects these `.pt` files to be available in `rgbd/live/models/`.

## Naming for GitHub Release Assets

If you publish the final weights on GitHub, place them in GitHub Releases and use a cleaner naming scheme of:

- `best_single.pt`
- `best_multiple.pt`
- `best_multiple_depth.pt`

If you use those names, update the model map in `rgbd/live/live_inference.py` so the demo can load them.

## Where to Place Them

Keep the model files next to the live demo code or upload them as GitHub Release assets and document the download location in the README.

The important thing is that the filenames in the code match the filenames on disk.
