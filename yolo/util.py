import os
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# =========================
# CLASS MAPPING
# dataset → YOLO
# 1 = crop → 0
# 2 = weed → 1
# =========================
CLASS_MAP = {1: 0, 2: 1}


def mask_to_yolo_polygons(mask, class_map=None):
    """Convert a semantic mask to YOLO polygon label strings.

    Args:
        mask: 2D numpy array of class ids.
        class_map: Mapping from dataset class id to YOLO class id.

    Returns:
        A list of strings ready to write into a YOLO polygon label file.
    """
    if class_map is None:
        class_map = CLASS_MAP

    h, w = mask.shape
    lines = []

    for dataset_cls, yolo_cls in class_map.items():
        binary_mask = (mask == dataset_cls).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue

            contour = contour.squeeze()
            if contour.ndim != 2:
                continue

            contour = contour.astype(np.float32)
            contour[:, 0] /= w
            contour[:, 1] /= h
            poly = contour.reshape(-1)
            lines.append("{} {}".format(yolo_cls, " ".join(map(str, poly))))

    return lines


def convert_semantic_masks_to_yolo_labels(
    dataset_dir="dataset", splits=("train", "val"), class_map=None
):
    """Convert semantic PNG masks into YOLO polygon label files.

    Args:
        dataset_dir: Root dataset folder containing split/semantics.
        splits: Which dataset splits to process.
        class_map: Mapping from dataset class id to YOLO class id.
    """
    if class_map is None:
        class_map = CLASS_MAP

    for split in splits:
        print(f"\n🔄 Processing {split} set...")
        mask_dir = os.path.join(dataset_dir, split, "semantics")
        label_dir = os.path.join(dataset_dir, split, "labels")
        os.makedirs(label_dir, exist_ok=True)

        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")]
        for mask_name in tqdm(mask_files, desc=f"{split} masks", unit="file"):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            label_path = os.path.join(label_dir, mask_name.replace(".png", ".txt"))
            lines = mask_to_yolo_polygons(mask, class_map=class_map)
            with open(label_path, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

        print(f"✅ Done {split}")


def copy_files_for_split(file_list, source_images, source_labels, output, split):
    os.makedirs(os.path.join(output, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output, split, "labels"), exist_ok=True)

    for img_file in file_list:
        label_file = img_file.rsplit(".", 1)[0] + ".txt"
        shutil.copy(
            os.path.join(source_images, img_file),
            os.path.join(output, split, "images", img_file),
        )

        src_label_path = os.path.join(source_labels, label_file)
        if os.path.exists(src_label_path):
            shutil.copy(
                src_label_path, os.path.join(output, split, "labels", label_file)
            )


def split_dataset(
    source_images="data/images",
    source_labels="data/labels",
    output="dataset_split",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=None,
):
    """Split a dataset into train/val/test folders and copy image/label files.

    Args:
        source_images: Directory with source images.
        source_labels: Directory with source YOLO labels.
        output: Output root folder.
        train_ratio: Fraction used for training.
        val_ratio: Fraction used for validation.
        test_ratio: Fraction used for testing.
        seed: Optional random seed.
    """
    if seed is not None:
        random.seed(seed)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output, split, "labels"), exist_ok=True)

    images = [
        f
        for f in os.listdir(source_images)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(images)

    n = len(images)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    copy_files_for_split(
        images[:train_end], source_images, source_labels, output, "train"
    )
    copy_files_for_split(
        images[train_end:val_end], source_images, source_labels, output, "val"
    )
    copy_files_for_split(images[val_end:], source_images, source_labels, output, "test")

    print("✅ Dataset split complete!")


if __name__ == "__main__":
    convert_semantic_masks_to_yolo_labels()
    split_dataset()
