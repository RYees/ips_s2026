# Live Demo Model Files

The live demo loads every `.pt` file found in `rgdb/live/models/` and shows the file stem in the dropdown.

Current model names:

| File | Purpose |
| --- | --- |
| `best_multiple_26.pt` | Multi-object detection / segmentation model |
| `best_single_26.pt` | Single-object detection / segmentation model |
| `m11best.pt` | Legacy model kept in the folder |
| `m8best.pt` | Legacy model kept in the folder |
| `s11best.pt` | Legacy model kept in the folder |
| `s8best.pt` | Legacy model kept in the folder |

The live demo keeps the selected model name visible in the navbar so operators can confirm which weight file is active.
