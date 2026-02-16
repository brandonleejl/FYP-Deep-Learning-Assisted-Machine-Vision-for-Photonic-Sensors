import argparse
import csv
import math
import os
import random
import re
from datetime import datetime
from typing import List, Sequence, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from data_augmentation import augment_image_and_mask, augment_regression_image
from data_preprocessing import (
    NUM_CLASSES,
    class_idx_to_ph,
    get_ph_values_tf,
    list_image_files,
    load_labels_csv,
    ph_to_class_idx,
    read_image,
    read_mask,
    split_train_val,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE


# -------------------------
# GPU config (safe default)
# -------------------------
def configure_gpu(gpu_only: bool = False) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        msg = "No GPU detected. Using CPU."
        if gpu_only:
            raise RuntimeError("No GPU detected but --gpu-only was enabled.")
        print(msg)
        return

    # Allow TF to allocate memory gradually
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("GPU detected:", [g.name for g in gpus])


# -------------------------
# Paths and config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_CSV = os.path.join(BASE_DIR, "labels.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MASK_DIR_EXTERNAL = r"C:\Users\brand\Downloads\FYP Code\masks"
MASK_DIR_DEFAULT = os.path.join(BASE_DIR, "masks")
MASK_DIR_ALT = os.path.join(BASE_DIR, "mask")
if os.path.isdir(MASK_DIR_EXTERNAL):
    MASK_DIR = MASK_DIR_EXTERNAL
elif os.path.isdir(MASK_DIR_DEFAULT):
    MASK_DIR = MASK_DIR_DEFAULT
elif os.path.isdir(MASK_DIR_ALT):
    MASK_DIR = MASK_DIR_ALT
else:
    MASK_DIR = os.path.join(BASE_DIR, "masks")

SEG_IMAGE_SIZE = (512, 512)
REG_IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4
SEG_EPOCHS = 40
CLS_EPOCHS_HEAD = 25      # train classifier head first
CLS_EPOCHS_FINETUNE = 25  # then finetune
VAL_SPLIT = 0.2


# -------------------------
# Mask matching helpers
# -------------------------
def list_mask_candidates(mask_dir: str) -> List[str]:
    patterns = ["*.json", "*.JSON", "*_mask.png", "*_mask.PNG", "*.png", "*.PNG"]
    files: List[str] = []
    for pattern in patterns:
        files.extend(tf.io.gfile.glob(os.path.join(mask_dir, pattern)))
    return sorted(set(files))


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+", text)


def resolve_mask_path_for_image(image_path: str, mask_candidates: Sequence[str]) -> str:
    """
    Match an image to a mask file even if filenames differ.
    Priority:
    1) exact stem / stem_mask
    2) same numeric id
    3) partial containment
    """
    image_stem = _stem(image_path).lower()
    image_nums = _extract_numbers(image_stem)

    exact = []
    numeric = []
    partial = []

    for mask_path in mask_candidates:
        mask_stem = _stem(mask_path).lower()
        mask_stem_base = mask_stem[:-5] if mask_stem.endswith("_mask") else mask_stem

        if mask_stem == image_stem or mask_stem_base == image_stem:
            exact.append(mask_path)
            continue

        if image_nums:
            mask_nums = _extract_numbers(mask_stem)
            if any(n in mask_nums for n in image_nums):
                numeric.append(mask_path)
                continue

        if image_stem in mask_stem or mask_stem in image_stem:
            partial.append(mask_path)

    if exact:
        return sorted(exact)[0]
    if numeric:
        return sorted(numeric)[0]
    if partial:
        return sorted(partial)[0]
    return ""


# -------------------------
# Segmentation tf.data
# -------------------------
def make_seg_dataset(
    pairs: Sequence[Tuple[str, str]],
    training: bool,
    class_name: str,
) -> tf.data.Dataset:
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    def _map_fn(image_path, mask_path):
        image = read_image(image_path, SEG_IMAGE_SIZE)              # float32 [0..1]
        mask = read_mask(mask_path, SEG_IMAGE_SIZE, class_name)     # expects your read_mask supports json label
        if training:
            image, mask = augment_image_and_mask(image, mask)
        return image, mask

    if training:
        ds = ds.shuffle(len(pairs), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# -------------------------
# DeepLabV3+ (MobileNetV2)
# -------------------------
def conv_bn_relu(x: tf.Tensor, filters: int, kernel_size: int, dilation_rate: int = 1) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        dilation_rate=dilation_rate,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_deeplabv3plus(input_shape=(512, 512, 3)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    # Prefer ImageNet weights, fallback if download blocked
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet",
        )
        print("Seg backbone: MobileNetV2 (ImageNet weights)")
    except Exception as e:
        print("WARNING: Could not load ImageNet weights for MobileNetV2. Falling back to weights=None.")
        print("Reason:", str(e)[:200], "...")
        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights=None,
        )

    low_level = backbone.get_layer("block_3_expand_relu").output
    high_level = backbone.get_layer("block_13_expand_relu").output

    # ASPP
    b0 = conv_bn_relu(high_level, 256, 1)
    b1 = conv_bn_relu(high_level, 256, 3, dilation_rate=6)
    b2 = conv_bn_relu(high_level, 256, 3, dilation_rate=12)
    b3 = conv_bn_relu(high_level, 256, 3, dilation_rate=18)

    b4 = layers.GlobalAveragePooling2D()(high_level)
    b4 = layers.Reshape((1, 1, int(high_level.shape[-1])))(b4)
    b4 = conv_bn_relu(b4, 256, 1)
    b4 = layers.UpSampling2D(
        size=(int(high_level.shape[1]), int(high_level.shape[2])),
        interpolation="bilinear",
    )(b4)

    x = layers.Concatenate()([b0, b1, b2, b3, b4])
    x = conv_bn_relu(x, 256, 1)

    # Decoder
    low_level = conv_bn_relu(low_level, 48, 1)
    x = layers.UpSampling2D(
        size=(
            int(low_level.shape[1]) // int(x.shape[1]),
            int(low_level.shape[2]) // int(x.shape[2]),
        ),
        interpolation="bilinear",
    )(x)
    x = layers.Concatenate()([x, low_level])
    x = conv_bn_relu(x, 256, 3)
    x = conv_bn_relu(x, 256, 3)

    x = layers.UpSampling2D(
        size=(
            input_shape[0] // int(x.shape[1]),
            input_shape[1] // int(x.shape[2]),
        ),
        interpolation="bilinear",
    )(x)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="deeplabv3plus_segmentation")


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth=1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# -------------------------
# ROI cropping utilities
# -------------------------
def crop_bbox_from_mask(image: np.ndarray, mask01: np.ndarray, pad_frac: float = 0.05) -> np.ndarray:
    """
    image: (H,W,3), mask01: (H,W) in {0,1}
    Crop to bounding box of mask, with optional padding.
    """
    ys, xs = np.where(mask01 > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return image  # fallback if empty mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    h, w = mask01.shape[:2]
    pad_x = int((x_max - x_min + 1) * pad_frac)
    pad_y = int((y_max - y_min + 1) * pad_frac)

    x_min = max(0, x_min - pad_x)
    x_max = min(w - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(h - 1, y_max + pad_y)

    return image[y_min:y_max + 1, x_min:x_max + 1, :]


def resize_with_pad_np(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize with padding (keeps aspect ratio).
    """
    t = tf.convert_to_tensor(img, dtype=tf.float32)
    t = tf.image.resize_with_pad(t, size[0], size[1])
    return t.numpy().astype(np.float32)


# -------------------------
# Build masked/cropped arrays for pH classifier
# -------------------------
def build_classifier_arrays(
    image_paths: Sequence[str],
    ph_indices: Sequence[int],
    seg_model: keras.Model,
    mask_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.int32] = []

    for path, y_idx in zip(image_paths, ph_indices):
        # Read full image at seg size for consistent mask prediction
        image_tf = read_image(tf.constant(path), SEG_IMAGE_SIZE)  # tf.Tensor
        pred_mask = seg_model.predict(tf.expand_dims(image_tf, 0), verbose=0)[0, ..., 0]
        mask01 = (pred_mask > mask_threshold).astype(np.float32)

        image_np = image_tf.numpy().astype(np.float32)
        roi = crop_bbox_from_mask(image_np, mask01, pad_frac=0.05)
        roi = resize_with_pad_np(roi, REG_IMAGE_SIZE)

        xs.append(roi)
        ys.append(np.int32(y_idx))

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


def make_cls_dataset(x: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b: (augment_regression_image(a), b), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# -------------------------
# pH classifier (transfer learning)
# -------------------------
def build_ph_classifier(num_classes: int, input_shape=(224, 224, 3)) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    try:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
        print("Classifier backbone: MobileNetV2 (ImageNet weights)")
    except Exception as e:
        print("WARNING: Could not load ImageNet weights for classifier backbone; using weights=None.")
        print("Reason:", str(e)[:200], "...")
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_tensor=inputs,
        )

    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)  # logits

    model = keras.Model(inputs, outputs, name="ph_classifier")
    return model


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


# -------------------------
# Save results
# -------------------------
def save_results_csv(results: dict, run_timestamp: str, results_dir: str = RESULTS_DIR) -> str:
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{run_timestamp}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
    return out_path


def save_predictions_csv(
    image_paths: Sequence[str],
    actual_ph: Sequence[float],
    actual_idx: Sequence[int],
    pred_idx: Sequence[int],
    pred_ph_class: Sequence[float],
    pred_ph_expected: Sequence[float],
    confidence: Sequence[float],
    mae: float,
    rmse: float,
    r2: float,
    run_timestamp: str,
    results_dir: str = RESULTS_DIR,
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{run_timestamp}_predictions.csv")
    run_date = run_timestamp.split("_")[0]

    fieldnames = [
        # New schema
        "run_timestamp",
        "run_date",
        "image_path",
        "filename",
        "actual_ph",
        "actual_idx",
        "pred_idx",
        "pred_ph_class",
        "pred_ph_expected",
        "abs_error_expected",
        # Legacy-compatible schema for Excel/report consumers
        "timestamp",
        "predicted_ph_expected",
        "predicted_ph_class",
        "confidence",
        "abs_error",
        "mae",
        "rmse",
        "r2",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for p, yt_ph, yt_idx, yp_idx, yp_class, yp_exp, conf in zip(
            image_paths, actual_ph, actual_idx, pred_idx, pred_ph_class, pred_ph_expected, confidence
        ):
            err = abs(float(yp_exp) - float(yt_ph))
            writer.writerow(
                {
                    "run_timestamp": run_timestamp,
                    "run_date": run_date,
                    "image_path": str(p),
                    "filename": os.path.basename(p),
                    "actual_ph": float(yt_ph),
                    "actual_idx": int(yt_idx),
                    "pred_idx": int(yp_idx),
                    "pred_ph_class": float(yp_class),
                    "pred_ph_expected": float(yp_exp),
                    "abs_error_expected": float(err),
                    "timestamp": run_timestamp,
                    "predicted_ph_expected": float(yp_exp),
                    "predicted_ph_class": float(yp_class),
                    "confidence": float(conf),
                    "abs_error": float(err),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                }
            )
    return out_path


def save_results_excel(csv_path: str, run_timestamp: str, results_dir: str = RESULTS_DIR) -> str:
    """Save summary row as Excel with timestamped filename."""
    if pd is None:
        raise ModuleNotFoundError(
            "Pandas is required for Excel export. Install with: pip install pandas openpyxl"
        )

    out_path = os.path.join(results_dir, f"{run_timestamp}.xlsx")
    df = pd.read_csv(csv_path)
    df.to_excel(out_path, index=False)
    return out_path


def save_predictions_excel(csv_path: str, run_timestamp: str, results_dir: str = RESULTS_DIR) -> str:
    """Save predictions as Excel with a timestamped filename."""
    if pd is None:
        raise ModuleNotFoundError(
            "Pandas is required for Excel export. Install with: pip install pandas openpyxl"
        )

    out_path = os.path.join(results_dir, f"{run_timestamp}_predictions.xlsx")
    df = pd.read_csv(csv_path)
    df.to_excel(out_path, index=False)
    return out_path


def save_report_bundle(
    actual_ph: np.ndarray,
    pred_ph_expected: np.ndarray,
    abs_err: np.ndarray,
    mae: float,
    rmse: float,
    r2: float,
    acc_01: float,
    acc_03: float,
    acc_05: float,
    run_timestamp: str,
    results_dir: str = RESULTS_DIR,
) -> Dict[str, str]:
    """Export metrics text + publication-ready figures."""
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"{run_timestamp}_metrics.txt")

    metrics_text = "\n".join(
        [
            f"Run timestamp: {run_timestamp}",
            f"Samples: {int(actual_ph.shape[0])}",
            f"MAE: {mae:.6f}",
            f"RMSE: {rmse:.6f}",
            f"R^2: {r2:.6f}",
            f"Accuracy within +/-0.1: {acc_01:.4f}",
            f"Accuracy within +/-0.3: {acc_03:.4f}",
            f"Accuracy within +/-0.5: {acc_05:.4f}",
        ]
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")

    plt.rcParams.update({"font.size": 11})

    # FIG1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual_ph, pred_ph_expected, alpha=0.8)
    lo = float(min(np.min(actual_ph), np.min(pred_ph_expected)))
    hi = float(max(np.max(actual_ph), np.max(pred_ph_expected)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title("Actual vs Predicted pH")
    ax.set_xlabel("Actual pH")
    ax.set_ylabel("Predicted pH (expected)")
    ax.text(
        0.03,
        0.97,
        f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR^2: {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig1_png = os.path.join(results_dir, f"{run_timestamp}_fig1_actual_vs_pred.png")
    fig1_pdf = os.path.join(results_dir, f"{run_timestamp}_fig1_actual_vs_pred.pdf")
    fig.savefig(fig1_png, dpi=300)
    fig.savefig(fig1_pdf)
    plt.close(fig)

    # FIG2
    residual = pred_ph_expected - actual_ph
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(actual_ph, residual, alpha=0.8)
    ax.axhline(0.0, linestyle="--")
    ax.set_title("Residuals vs Actual pH")
    ax.set_xlabel("Actual pH")
    ax.set_ylabel("Residual (Predicted - Actual)")
    fig.tight_layout()
    fig2_png = os.path.join(results_dir, f"{run_timestamp}_fig2_residuals.png")
    fig2_pdf = os.path.join(results_dir, f"{run_timestamp}_fig2_residuals.pdf")
    fig.savefig(fig2_png, dpi=300)
    fig.savefig(fig2_pdf)
    plt.close(fig)

    # FIG3
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(abs_err, bins=20)
    ax.axvline(0.1, linestyle="--")
    ax.axvline(0.3, linestyle="--")
    ax.axvline(0.5, linestyle="--")
    ax.set_title("Histogram of Absolute Error")
    ax.set_xlabel("Absolute Error |Predicted - Actual|")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig3_png = os.path.join(results_dir, f"{run_timestamp}_fig3_abs_error_hist.png")
    fig3_pdf = os.path.join(results_dir, f"{run_timestamp}_fig3_abs_error_hist.pdf")
    fig.savefig(fig3_png, dpi=300)
    fig.savefig(fig3_pdf)
    plt.close(fig)

    # FIG4
    tol = np.linspace(0.0, 1.0, 101)
    acc = np.array([np.mean(abs_err <= t) for t in tol], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tol, acc)
    ax.set_title("Tolerance Accuracy Curve")
    ax.set_xlabel("Tolerance")
    ax.set_ylabel("Accuracy: P(|error| <= tolerance)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig4_png = os.path.join(results_dir, f"{run_timestamp}_fig4_tolerance_curve.png")
    fig4_pdf = os.path.join(results_dir, f"{run_timestamp}_fig4_tolerance_curve.pdf")
    fig.savefig(fig4_png, dpi=300)
    fig.savefig(fig4_pdf)
    plt.close(fig)

    return {
        "metrics": metrics_path,
        "fig1_png": fig1_png,
        "fig1_pdf": fig1_pdf,
        "fig2_png": fig2_png,
        "fig2_pdf": fig2_pdf,
        "fig3_png": fig3_png,
        "fig3_pdf": fig3_pdf,
        "fig4_png": fig4_png,
        "fig4_pdf": fig4_pdf,
    }


# -------------------------
# MAIN
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-only", action="store_true", help="Crash if no GPU detected.")
    parser.add_argument("--labelme-class", default="hydrogel", help="Labelme polygon label name for hydrogel.")
    parser.add_argument("--disable-seg-early-stopping", action="store_true", help="Disable segmentation EarlyStopping.")
    parser.add_argument("--disable-cls-early-stopping", action="store_true", help="Disable classifier EarlyStopping (both head and finetune).")
    parser.add_argument("--seg-es-patience", type=int, default=10, help="Segmentation EarlyStopping patience.")
    parser.add_argument("--seg-es-min-delta", type=float, default=1e-4, help="Segmentation EarlyStopping min_delta.")
    parser.add_argument("--cls-es-patience", type=int, default=8, help="Classifier EarlyStopping patience.")
    parser.add_argument("--cls-es-min-delta", type=float, default=1e-4, help="Classifier EarlyStopping min_delta.")
    args = parser.parse_args()

    configure_gpu(gpu_only=args.gpu_only)

    # 1) Collect images
    image_paths_all = list_image_files(IMAGE_DIR)
    if not image_paths_all:
        raise RuntimeError(f"No images found in '{IMAGE_DIR}'.")

    # 2) Load pH labels (discrete values)
    if not tf.io.gfile.exists(LABEL_CSV):
        raise RuntimeError("Missing labels.csv with columns: filename,ph")

    labels = load_labels_csv(LABEL_CSV)  # Dict[filename] -> float pH

    # 3) Match masks for all images (seg data)
    mask_candidates = list_mask_candidates(MASK_DIR)
    if not mask_candidates:
        raise RuntimeError(f"No masks found in '{MASK_DIR}'. Expected LabelMe .json or mask png.")

    image_to_mask: Dict[str, str] = {}
    missing_masks = []
    non_json = []
    for img in image_paths_all:
        m = resolve_mask_path_for_image(img, mask_candidates)
        if not m:
            missing_masks.append(os.path.basename(img))
            continue
        image_to_mask[img] = m
        if not m.lower().endswith(".json"):
            non_json.append(os.path.basename(m))

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise RuntimeError(f"Missing masks for {len(missing_masks)} images. Examples: {preview}")

    # If you're using LabelMe JSON, keep this strict:
    if non_json:
        preview = ", ".join(non_json[:10])
        raise RuntimeError(
            f"Some matched masks are not .json (but you said you use LabelMe). "
            f"Count: {len(non_json)}. Examples: {preview}"
        )

    # 4) Build list of images that have BOTH masks and pH labels (for pH stage)
    eligible = []
    for img in image_paths_all:
        fname = os.path.basename(img)
        if fname in labels and img in image_to_mask:
            eligible.append(img)

    if len(eligible) < 10:
        raise RuntimeError(f"Not enough labeled samples for pH. Found only {len(eligible)}.")

    print(f"Total images: {len(image_paths_all)}")
    print(f"Eligible (mask + pH): {len(eligible)}")

    # 5) Single split (no leakage)
    # Split based on eligible images. Seg and pH training use same split.
    train_imgs, val_imgs = split_train_val(eligible, val_split=VAL_SPLIT)
    train_set = set(train_imgs)
    val_set = set(val_imgs)

    # Segmentation pairs:
    # - Train: all images NOT in val (so val never seen by seg)
    # - Val: only val images
    seg_train_pairs = [(img, image_to_mask[img]) for img in image_paths_all if img not in val_set]
    seg_val_pairs = [(img, image_to_mask[img]) for img in val_imgs]

    print(f"Seg train pairs: {len(seg_train_pairs)} | Seg val pairs: {len(seg_val_pairs)}")
    print(f"pH train samples: {len(train_imgs)} | pH val samples: {len(val_imgs)}")

    seg_train_ds = make_seg_dataset(seg_train_pairs, training=True, class_name=args.labelme_class)
    seg_val_ds = make_seg_dataset(seg_val_pairs, training=False, class_name=args.labelme_class)

    # 6) Train segmentation model
    seg_model = build_deeplabv3plus(input_shape=(SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 3))
    seg_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=[
            dice_coef,
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
        ],
    )

    seg_callbacks = [
        keras.callbacks.ModelCheckpoint("best_segmentation_deeplabv3plus.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    if not args.disable_seg_early_stopping:
        seg_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.seg_es_patience,
                min_delta=args.seg_es_min_delta,
                restore_best_weights=True,
            )
        )

    seg_model.fit(
        seg_train_ds,
        validation_data=seg_val_ds,
        epochs=SEG_EPOCHS,
        callbacks=seg_callbacks,
        verbose=1,
    )

    seg_eval = seg_model.evaluate(seg_val_ds, verbose=0)
    print("Segmentation eval [loss, dice, iou]:", seg_eval)

    # 7) Fixed pH classes over 0.1 steps from 3.0 to 8.0
    ph_values_tf = get_ph_values_tf()
    ph_values_arr = ph_values_tf.numpy().astype(np.float32)
    print(f"NUM_CLASSES: {NUM_CLASSES}")
    print(f"pH support first/last: {float(ph_values_arr[0]):.1f}, {float(ph_values_arr[-1]):.1f}")

    # Prepare train/val pH targets
    train_ph = [float(labels[os.path.basename(p)]) for p in train_imgs]
    val_ph = [float(labels[os.path.basename(p)]) for p in val_imgs]
    y_train_idx = [ph_to_class_idx(p) for p in train_ph]
    y_val_idx = [ph_to_class_idx(p) for p in val_ph]

    # Sanity check class index ranges.
    for idx in y_train_idx[:5] + y_val_idx[:5]:
        assert 0 <= int(idx) < NUM_CLASSES, f"Label index out of range: {idx}"

    # Quick mapping verification on 5 samples.
    print("Mapping sanity (first 5 train samples):")
    for path, ph, idx in list(zip(train_imgs, train_ph, y_train_idx))[:5]:
        decoded = class_idx_to_ph(int(idx))
        print(f"  {os.path.basename(path)} | ph_float={ph:.3f} | ph_idx={int(idx)} | decoded_ph={decoded:.3f}")

    # 8) Convert images to ROI arrays using seg predictions
    x_train, y_train = build_classifier_arrays(train_imgs, y_train_idx, seg_model, mask_threshold=0.5)
    x_val, y_val = build_classifier_arrays(val_imgs, y_val_idx, seg_model, mask_threshold=0.5)

    cls_train_ds = make_cls_dataset(x_train, y_train, training=True)
    cls_val_ds = make_cls_dataset(x_val, y_val, training=False)

    # 9) Train pH classifier (head first, then fine-tune)
    cls_model = build_ph_classifier(num_classes=NUM_CLASSES, input_shape=(REG_IMAGE_SIZE[0], REG_IMAGE_SIZE[1], 3))

    # Freeze backbone initially
    backbone = cls_model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2" in cls_model.layers[1].name.lower() else None
    # safer: freeze all except last Dense layers
    for layer in cls_model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = False

    cls_model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    cls_head_callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    if not args.disable_cls_early_stopping:
        cls_head_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.cls_es_patience,
                min_delta=args.cls_es_min_delta,
                restore_best_weights=True,
            )
        )

    cls_model.fit(
        cls_train_ds,
        validation_data=cls_val_ds,
        epochs=CLS_EPOCHS_HEAD,
        callbacks=cls_head_callbacks,
        verbose=1,
    )

    # Fine-tune: unfreeze last part of MobileNetV2 if available
    for layer in cls_model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True

    cls_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    cls_ft_callbacks = [
        keras.callbacks.ModelCheckpoint("best_ph_classifier.keras", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    if not args.disable_cls_early_stopping:
        cls_ft_callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.cls_es_patience,
                min_delta=args.cls_es_min_delta,
                restore_best_weights=True,
            )
        )

    cls_model.fit(
        cls_train_ds,
        validation_data=cls_val_ds,
        epochs=CLS_EPOCHS_FINETUNE,
        callbacks=cls_ft_callbacks,
        verbose=1,
    )

    # 10) Evaluate pH with expected-value decoding from softmax(logits).
    print("Evaluation preprocessing: deterministic only (no augmentation).")
    logits = cls_model.predict(x_val, verbose=0)  # (N, C)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    pred_idx = np.argmax(probs, axis=1).astype(np.int32)
    confidence = np.max(probs, axis=1).astype(np.float32)
    pred_ph_class = np.asarray([class_idx_to_ph(int(i)) for i in pred_idx], dtype=np.float32)
    pred_ph_expected = (probs * ph_values_arr[None, :]).sum(axis=1).astype(np.float32)

    y_true_ph = np.array(val_ph, dtype=np.float32)
    y_true_idx = np.array(y_val_idx, dtype=np.int32)

    mae = float(np.mean(np.abs(pred_ph_expected - y_true_ph)))
    rmse = float(math.sqrt(np.mean((pred_ph_expected - y_true_ph) ** 2)))
    r2 = r2_score(y_true_ph, pred_ph_expected)
    acc_idx = float(np.mean(pred_idx == y_true_idx))

    abs_err = np.abs(pred_ph_expected - y_true_ph)
    acc_01 = float(np.mean(abs_err <= 0.1))
    acc_03 = float(np.mean(abs_err <= 0.3))
    acc_05 = float(np.mean(abs_err <= 0.5))

    print(f"pH Classification Accuracy (index): {acc_idx:.4f}")
    print(f"pH Expected-Value MAE:             {mae:.4f}")
    print(f"pH Expected-Value RMSE:            {rmse:.4f}")
    print(f"pH Expected-Value R^2:             {r2:.4f}")
    print(f"Tolerance Accuracy |error|<=0.1:   {acc_01:.4f}")
    print(f"Tolerance Accuracy |error|<=0.3:   {acc_03:.4f}")
    print(f"Tolerance Accuracy |error|<=0.5:   {acc_05:.4f}")

    # Histogram/counts to detect class collapse.
    actual_counts = np.bincount(y_true_idx, minlength=NUM_CLASSES)
    pred_counts = np.bincount(pred_idx, minlength=NUM_CLASSES)
    actual_nonzero = {int(i): int(c) for i, c in enumerate(actual_counts) if c > 0}
    pred_nonzero = {int(i): int(c) for i, c in enumerate(pred_counts) if c > 0}
    print("Actual idx counts (non-zero):", actual_nonzero)
    print("Pred idx counts (non-zero):", pred_nonzero)

    if len(y_true_ph) > 0:
        print(
            "Eval example:",
            f"ph_float={float(y_true_ph[0]):.3f},",
            f"pred_ph_class={float(pred_ph_class[0]):.3f},",
            f"ph_expected={float(pred_ph_expected[0]):.3f}",
        )

    # 11) Save results and report artifacts with one run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": run_timestamp,
        "run_date": run_timestamp.split("_")[0],
        "image_dir": IMAGE_DIR,
        "mask_dir": MASK_DIR,
        "label_csv": LABEL_CSV,
        "labelme_class": args.labelme_class,
        "seg_model": "deeplabv3plus_segmentation",
        "seg_loss": float(seg_eval[0]),
        "seg_dice": float(seg_eval[1]),
        "seg_iou": float(seg_eval[2]),
        "ph_num_classes": NUM_CLASSES,
        "ph_cls_acc_idx": acc_idx,
        "ph_cls_acc": acc_idx,
        "ph_mae": mae,
        "ph_rmse": rmse,
        "ph_r2": r2,
        "ph_acc_tol_01": acc_01,
        "ph_acc_tol_03": acc_03,
        "ph_acc_tol_05": acc_05,
        "num_total_images": len(image_paths_all),
        "num_eligible_ph": len(eligible),
        "num_ph_train": len(train_imgs),
        "num_ph_val": len(val_imgs),
        # Legacy-compatible aliases for existing spreadsheets.
        "num_total": len(image_paths_all),
        "num_eligible": len(eligible),
    }

    results_csv = save_results_csv(results, run_timestamp=run_timestamp)
    results_xlsx = ""
    try:
        results_xlsx = save_results_excel(results_csv, run_timestamp=run_timestamp)
    except Exception as exc:
        print(f"Summary Excel export skipped: {exc}")

    preds_csv = save_predictions_csv(
        image_paths=val_imgs,
        actual_ph=y_true_ph,
        actual_idx=y_true_idx,
        pred_idx=pred_idx,
        pred_ph_class=pred_ph_class,
        pred_ph_expected=pred_ph_expected,
        confidence=confidence,
        mae=mae,
        rmse=rmse,
        r2=r2,
        run_timestamp=run_timestamp,
    )

    preds_xlsx = ""
    try:
        preds_xlsx = save_predictions_excel(preds_csv, run_timestamp=run_timestamp)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")

    report_paths = save_report_bundle(
        actual_ph=y_true_ph,
        pred_ph_expected=pred_ph_expected,
        abs_err=abs_err,
        mae=mae,
        rmse=rmse,
        r2=r2,
        acc_01=acc_01,
        acc_03=acc_03,
        acc_05=acc_05,
        run_timestamp=run_timestamp,
        results_dir=RESULTS_DIR,
    )

    print(f"Saved results CSV: {results_csv}")
    if results_xlsx:
        print(f"Saved results Excel: {results_xlsx}")
    print(f"Saved predictions CSV: {preds_csv}")
    if preds_xlsx:
        print(f"Saved predictions Excel: {preds_xlsx}")
    print(f"Saved metrics report: {report_paths['metrics']}")
    print(f"Saved figure: {report_paths['fig1_png']}")
    print(f"Saved figure: {report_paths['fig2_png']}")
    print(f"Saved figure: {report_paths['fig3_png']}")
    print(f"Saved figure: {report_paths['fig4_png']}")
    print("Saved models: best_segmentation_deeplabv3plus.keras, best_ph_classifier.keras")


if __name__ == "__main__":
    main()
