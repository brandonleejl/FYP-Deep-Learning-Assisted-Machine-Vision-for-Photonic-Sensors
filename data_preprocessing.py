import csv
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def list_image_files(image_dir: str) -> List[str]:
    """Return sorted image files from a directory."""
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files: List[str] = []
    for ext in exts:
        files.extend(tf.io.gfile.glob(os.path.join(image_dir, ext)))
    return sorted(files)


def create_labels_template(image_dir: str, output_csv: str) -> int:
    """Create a CSV template with columns: filename,ph."""
    image_paths = list_image_files(image_dir)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ph"])
        for path in image_paths:
            writer.writerow([os.path.basename(path), ""])
    return len(image_paths)


def preprocess_images(
    source_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
) -> int:
    """Resize with padding and write preprocessed PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = list_image_files(source_dir)

    for path in image_paths:
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
        img = tf.image.resize_with_pad(img, target_size[0], target_size[1])

        out_path = os.path.join(output_dir, os.path.basename(path))
        png = tf.image.encode_png(tf.cast(img * 255.0, tf.uint8))
        tf.io.write_file(out_path, png)

    return len(image_paths)


def load_labels_csv(label_csv: str) -> Dict[str, float]:
    """Read labels.csv with columns: filename,ph. Invalid/empty rows are skipped."""
    labels: Dict[str, float] = {}
    with open(label_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get("filename") or "").strip()
            ph_text = (row.get("ph") or "").strip()
            if not filename or not ph_text:
                continue
            try:
                labels[filename] = float(ph_text)
            except ValueError:
                # Keep training robust if one row has malformed numeric data.
                continue
    return labels


def get_mask_path(image_path: str, mask_dir: str) -> str:
    """Resolve the best available mask filename for an image."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    candidates = [
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}_mask.PNG"),
        os.path.join(mask_dir, f"{stem}.png"),
        os.path.join(mask_dir, f"{stem}.PNG"),
    ]

    for candidate in candidates:
        if tf.io.gfile.exists(candidate):
            return candidate

    # Default expected path when mask does not exist yet.
    return candidates[0]


def split_train_val(items: Sequence, val_split: float = 0.2, seed: int = SEED):
    """Random train/validation split with sanity checks."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1 (exclusive).")
    if len(items) < 2:
        raise ValueError("Need at least 2 samples to create train/validation split.")

    idx = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_val = max(1, int(len(items) * val_split))
    n_val = min(n_val, len(items) - 1)
    val_set = set(idx[:n_val])
    train_items = [items[i] for i in range(len(items)) if i not in val_set]
    val_items = [items[i] for i in range(len(items)) if i in val_set]
    return train_items, val_items


def read_image(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Decode RGB image and resize."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method="bilinear")
    return img


def read_mask(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    """Decode binary mask and resize with nearest interpolation."""
    mask = tf.io.read_file(path)
    mask = tf.io.decode_image(mask, channels=1, expand_animations=False)
    mask.set_shape([None, None, 1])
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, image_size, method="nearest")
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def augment_image_and_mask(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply identical geometric augmentations to image and mask."""
    seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed)

    seed = tf.random.uniform([2], maxval=10000, dtype=tf.int32)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed)

    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def augment_regression_image(image: tf.Tensor) -> tf.Tensor:
    """Photometric augmentation for regression model."""
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


if __name__ == "__main__":
    src = "train"
    dst = "train_preprocessed"
    labels_template = "labels_template.csv"

    n = preprocess_images(src, dst, target_size=(512, 512))
    print(f"Preprocessed {n} images from '{src}' to '{dst}'.")

    m = create_labels_template(src, labels_template)
    print(f"Created '{labels_template}' with {m} rows. Fill in pH values before training.")
