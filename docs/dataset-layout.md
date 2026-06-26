# Dataset Layout

## CPS Server Locations

The prepared datasets live on the CPS server at:

```text
/mnt/cps_persistent1_shared/ips_s2026/data-finals/dataset-singles-seg
/mnt/cps_persistent1_shared/ips_s2026/data-finals/dataset-multiples-seg
/mnt/cps_persistent1_shared/ips_s2026/data-finals/dataset-singles-cls
```

## What Each Split Is For

- `dataset-singles-seg` - single-object segmentation training
- `dataset-multiples-seg` - multiple-object segmentation training
- `dataset-singles-cls` - single-object classification training

## Collected Data Structure

The live collector stores each capture under `rgbd/data/` with consistent names so the dataset can be rebuilt or audited later.

Use `docs/data-collection.md` for the save tree and `docs/models.md` for the model naming conventions.
