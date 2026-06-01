"""
Batch binary-mask to YOLO label export for a dataset tree.

This script reads the already-generated binary mask PNG for each sample and
converts it into a YOLO segmentation label using the existing annotation writer.

Expected dataset layout:
    <dataset_root>/
        images/
        pointcloud/
        info/
        masks/
        labels/

Usage:
    python3 offline_case/batch_mask_to_yolo.py --dataset-root offline_case/dataset
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_stems(dataset_root: Path) -> tuple[list[str], dict[str, set[str]]]:
    images = {p.stem for p in (dataset_root / "images").glob("*.png")}
    pointclouds = {p.stem for p in (dataset_root / "pointcloud").glob("*.ply")}
    infos = {p.stem for p in (dataset_root / "info").glob("*.txt")}
    stems = sorted(images & pointclouds & infos)
    return stems, {"images": images, "pointcloud": pointclouds, "info": infos}


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_empty_label(label_path: Path) -> None:
    label_path.write_text("", encoding="utf-8")


def parse_selected_class(info_path: Path, default: int = 0) -> int:
    if not info_path.exists():
        return default

    text = info_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^\s*Selected class:\s*(\d+)\s*$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return default


def load_image_shape(image_path: Path) -> tuple[int, int] | None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return height, width


def process_one(
    dataset_root: Path,
    stem: str,
    image_dir: Path,
    info_dir: Path,
    mask_dir: Path,
    label_dir: Path,
    writer,
    default_class: int,
) -> bool:
    import cv2

    image_path = image_dir / f"{stem}.png"
    info_path = info_dir / f"{stem}.txt"
    mask_path = mask_dir / f"{stem}.png"
    label_path = label_dir / f"{stem}.txt"

    img_shape = load_image_shape(image_path)
    if img_shape is None:
        print(f"[SKIP] {stem}: could not load image size from {image_path}", flush=True)
        write_empty_label(label_path)
        return False

    label_class = parse_selected_class(info_path, default=default_class)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[SKIP] {stem}: could not load mask PNG from {mask_path}", flush=True)
        write_empty_label(label_path)
        return False

    wrote_label = writer.write(str(label_path), mask, img_shape, label_class)
    if not wrote_label:
        write_empty_label(label_path)

    fg_px = int((mask > 0).sum())
    print(
        f"[DONE] {stem}: mask={mask_path.name}, label={label_path.name}, foreground_px={fg_px:,}",
        flush=True,
    )
    return True


def process_all(dataset_root: Path, default_class: int) -> int:
    from rgbd.processing.annotation_writer import AnnotationWriter

    mask_dir = dataset_root / "masks"
    label_dir = dataset_root / "labels"
    image_dir = dataset_root / "images"
    info_dir = dataset_root / "info"
    ensure_dirs(mask_dir, label_dir)

    stems, availability = discover_stems(dataset_root)
    print(
        "[DATASET] "
        f"images={len(availability['images'])}, "
        f"pointcloud={len(availability['pointcloud'])}, "
        f"info={len(availability['info'])}, "
        f"matched={len(stems)}",
        flush=True,
    )

    missing = {
        kind: sorted(items - set(stems))
        for kind, items in availability.items()
        if items - set(stems)
    }
    for kind, names in missing.items():
        print(f"[WARN] {kind} without full match: {', '.join(names[:10])}", flush=True)

    writer = AnnotationWriter(normalized=True)
    ok_count = 0
    for idx, stem in enumerate(stems, start=1):
        print(f"\n[{idx}/{len(stems)}] Processing {stem}", flush=True)
        if process_one(
            dataset_root,
            stem,
            image_dir,
            info_dir,
            mask_dir,
            label_dir,
            writer,
            default_class,
        ):
            ok_count += 1

    print(
        f"\n[SUMMARY] completed={ok_count}/{len(stems)} samples, "
        f"masks_dir={mask_dir}, labels_dir={label_dir}",
        flush=True,
    )
    return ok_count


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert binary mask PNGs into YOLO polygon labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("offline_case/dataset"),
        help="Dataset root containing images/, pointcloud/, info/, masks/, labels/",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id to write into each label file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    process_all(dataset_root, args.class_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
