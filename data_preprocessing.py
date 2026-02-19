import csv
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PH_MIN = 3.0
PH_MAX = 8.0
PH_STEP = 0.1
NUM_CLASSES = int(round((PH_MAX - PH_MIN) / PH_STEP)) + 1


def ph_to_class_idx(ph: float) -> int:
    idx = int(round((ph - PH_MIN) / PH_STEP))
    return max(0, min(NUM_CLASSES - 1, idx))


def class_idx_to_ph(idx: int) -> float:
    return PH_MIN + float(idx) * PH_STEP


def get_ph_values_tf() -> tf.Tensor:
    """Return pH support values for expected-value decoding."""
    return tf.linspace(PH_MIN, PH_MAX, NUM_CLASSES)


def list_image_files(image_dir: str) -> List[str]:
    """Return sorted image files from a directory."""
    exts = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
    files: List[str] = []
    for ext in exts:
        files.extend(tf.io.gfile.glob(os.path.join(image_dir, ext)))

    # Windows can match the same file across uppercase/lowercase glob patterns.
    # Keep only unique absolute paths in a case-insensitive way.
    unique: Dict[str, str] = {}
    for path in files:
        abs_path = os.path.abspath(path)
        key = os.path.normcase(abs_path)
        if key not in unique:
            unique[key] = abs_path

    return sorted(unique.values())


def create_labels_template(image_dir: str, output_csv: str) -> int:
    """
    Create or update labels CSV with columns: filename,ph.

    If the CSV already exists, keep previously entered pH values and only add
    missing filenames.
    """
    image_paths = list_image_files(image_dir)
    existing_ph: Dict[str, str] = {}

    if os.path.exists(output_csv):
        with open(output_csv, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                ph_value = (row.get("ph") or "").strip()
                if filename:
                    key = filename.lower()
                    # Keep the first non-empty pH if duplicates exist.
                    if key not in existing_ph or (not existing_ph[key] and ph_value):
                        existing_ph[key] = ph_value

    # Always update the same CSV file path in place to avoid duplicate files.
    output_csv = os.path.abspath(output_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ph"])
        seen_filenames = set()
        for path in image_paths:
            filename = os.path.basename(path)
            key = filename.lower()
            if key in seen_filenames:
                continue
            seen_filenames.add(key)
            writer.writerow([filename, existing_ph.get(key, "")])

    return len(seen_filenames)


def preprocess_images(
    source_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
) -> int:
    """Resize with padding and write preprocessed PNG files for DeepLab input."""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = list_image_files(source_dir)
    if not image_paths:
        return 0

    # Create a list of output paths corresponding to image paths
    out_paths = [os.path.join(output_dir, os.path.basename(p)) for p in image_paths]

    # Use tf.data.Dataset for parallel processing
    ds = tf.data.Dataset.from_tensor_slices((image_paths, out_paths))

    def _process_image(image_path: tf.Tensor, out_path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(image_path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0, 1]
        img = tf.image.resize_with_pad(img, target_size[0], target_size[1])

        # Encode to PNG and save to disk
        png = tf.image.encode_png(tf.cast(img * 255.0, tf.uint8))
        tf.io.write_file(out_path, png)
        return image_path

    # Parallelize the map with AUTOTUNE
    ds = ds.map(_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    # Prefetch for better performance
    ds = ds.prefetch(tf.data.AUTOTUNE)

    # Iterate through the dataset to trigger processing
    count = 0
    for _ in ds:
        count += 1

    return count


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
        os.path.join(mask_dir, f"{stem}.json"),
        os.path.join(mask_dir, f"{stem}_mask.json"),
        os.path.join(mask_dir, f"{stem}_mask.png"),
        os.path.join(mask_dir, f"{stem}_mask.PNG"),
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


def _read_mask_py(path, target_h, target_w, class_name):
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for mask loading. Install with: pip install pillow"
        ) from exc

    path_str = path.numpy().decode("utf-8")
    h = int(target_h.numpy())
    w = int(target_w.numpy())
    label = class_name.numpy().decode("utf-8")
    ext = os.path.splitext(path_str)[1].lower()

    if ext == ".json":
        with open(path_str, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not isinstance(img_w, int) or not isinstance(img_h, int):
            raise ValueError(f"Labelme JSON missing imageWidth/imageHeight: {path_str}")

        pil_mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(pil_mask)

        shapes = data.get("shapes", [])
        labeled_shapes = [s for s in shapes if s.get("label") == label]
        draw_shapes = labeled_shapes if labeled_shapes else shapes

        for shape in draw_shapes:
            pts = shape.get("points", [])
            if len(pts) < 3:
                continue
            polygon = [(float(p[0]), float(p[1])) for p in pts]
            draw.polygon(polygon, fill=255)
    else:
        pil_mask = Image.open(path_str).convert("L")

    pil_mask = pil_mask.resize((w, h), resample=Image.NEAREST)
    arr = np.asarray(pil_mask, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr


def read_mask(path: tf.Tensor, image_size: Tuple[int, int], class_name: str = "hydrogel") -> tf.Tensor:
    """Decode binary mask from PNG/JPG or Labelme JSON and resize."""
    target_h, target_w = image_size
    mask = tf.py_function(
        _read_mask_py,
        inp=[path, target_h, target_w, tf.constant(class_name)],
        Tout=tf.float32,
    )
    mask.set_shape([target_h, target_w, 1])
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return mask


def make_ph_dataset(
    image_paths: Sequence[str],
    labels: Dict[str, float],
    image_size: Tuple[int, int],
    batch_size: int,
    training: bool = False,
    augment_fn=None,
    seed: int = SEED,
) -> tf.data.Dataset:
    """
    Build pH dataset returning image, (ph_float, ph_idx_int).
    Keeps labels.csv format unchanged (filename,ph).
    """
    samples = []
    for path in image_paths:
        filename = os.path.basename(path)
        if filename not in labels:
            continue
        ph_float = float(labels[filename])
        ph_idx = ph_to_class_idx(ph_float)
        samples.append((path, ph_float, ph_idx))

    if not samples:
        raise ValueError("No labeled samples available to build pH dataset.")

    path_arr = [s[0] for s in samples]
    ph_float_arr = np.asarray([s[1] for s in samples], dtype=np.float32)
    ph_idx_arr = np.asarray([s[2] for s in samples], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((path_arr, ph_float_arr, ph_idx_arr))

    def _map_fn(path, ph_float, ph_idx):
        image = read_image(path, image_size)
        if training and augment_fn is not None:
            image = augment_fn(image)
        return image, (ph_float, tf.cast(ph_idx, tf.int32))

    if training:
        ds = ds.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(base_dir, "images")
    dst = os.path.join(base_dir, "images_preprocessed")
    labels_csv = os.path.join(base_dir, "labels.csv")

    n = preprocess_images(src, dst, target_size=(512, 512))
    print(f"Preprocessed {n} images from '{src}' to '{dst}'.")

    m = create_labels_template(src, labels_csv)
    print(f"Updated '{labels_csv}' with {m} unique filenames. Fill in/update pH values as needed.")
