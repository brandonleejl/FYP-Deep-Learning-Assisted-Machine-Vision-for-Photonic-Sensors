import csv
import math
import os
import random
import re
from datetime import datetime
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_augmentation import augment_image_and_mask, augment_regression_image
from data_preprocessing import (
    list_image_files,
    load_labels_csv,
    read_image,
    read_mask,
    split_train_val,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def configure_gpu_only() -> None:
    """
    Enforce GPU-only execution.
    Raises an error if no GPU is available so the run never falls back to CPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No GPU detected. GPU-only mode is enabled; CPU fallback is disabled. "
            "Install CUDA/cuDNN-compatible TensorFlow GPU environment and try again."
        )

    # Restrict TensorFlow to GPU devices only.
    tf.config.set_visible_devices(gpus, "GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpu_names = [gpu.name for gpu in gpus]
    print(f"GPU-only mode enabled. Visible GPUs: {gpu_names}")


configure_gpu_only()
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# Paths and training config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
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
    MASK_DIR = IMAGE_DIR
LABEL_CSV = os.path.join(BASE_DIR, "labels.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SEG_IMAGE_SIZE = (512, 512)
REG_IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4
SEG_EPOCHS = 40
REG_EPOCHS = 60
VAL_SPLIT = 0.2
REQUIRE_JSON_MASKS = True


def list_mask_candidates(mask_dir: str) -> List[str]:
    """List possible mask files (JSON/PNG) from mask_dir."""
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
    Match an image to a mask file even when filenames differ.
    Priority:
    1) exact stem / stem_mask
    2) same numeric id (e.g., IMG_6042 <-> any mask containing 6042)
    3) partial stem containment
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


def make_seg_dataset(pairs: Sequence[Tuple[str, str]], training: bool) -> tf.data.Dataset:
    """Create segmentation tf.data pipeline."""
    image_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    def _map_fn(image_path, mask_path):
        image = read_image(image_path, SEG_IMAGE_SIZE)
        mask = read_mask(mask_path, SEG_IMAGE_SIZE)
        if training:
            image, mask = augment_image_and_mask(image, mask)
        return image, mask

    if training:
        ds = ds.shuffle(len(pairs), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


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
    """DeepLabV3+ style segmentation model with MobileNetV2 backbone."""
    inputs = keras.Input(shape=input_shape)

    # `weights=None` avoids runtime model download requirements.
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=None,
    )

    low_level = backbone.get_layer("block_3_expand_relu").output
    high_level = backbone.get_layer("block_13_expand_relu").output

    # ASPP branch
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

    # Decoder branch
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


def build_regressor(input_shape=(224, 224, 3)) -> keras.Model:
    """CNN regressor for pH prediction."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs, name="ph_regressor")


def build_regression_arrays(
    image_paths: Sequence[str],
    ph_values: Sequence[float],
    seg_model: keras.Model,
    mask_threshold: float = 0.5,
):
    """Generate masked images (hydrogel-only) for regression."""
    xs: List[np.ndarray] = []
    ys: List[np.float32] = []

    for path, ph in zip(image_paths, ph_values):
        image = read_image(tf.constant(path), SEG_IMAGE_SIZE)
        pred_mask = seg_model.predict(tf.expand_dims(image, 0), verbose=0)[0, ..., 0]
        mask = (pred_mask > mask_threshold).astype(np.float32)

        masked_image = image.numpy() * np.expand_dims(mask, axis=-1)
        masked_image = tf.image.resize(masked_image, REG_IMAGE_SIZE).numpy().astype(np.float32)

        xs.append(masked_image)
        ys.append(np.float32(ph))

    return np.array(xs), np.array(ys)


def make_reg_dataset(x: np.ndarray, y: np.ndarray, training: bool) -> tf.data.Dataset:
    """Create regression tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b: (augment_regression_image(a), b), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-12)))


def save_results_csv(results: dict, results_dir: str = RESULTS_DIR) -> str:
    """Save one-row experiment metrics to timestamped CSV."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{timestamp}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        writer.writeheader()
        writer.writerow(results)
    return out_path


def save_test_predictions_csv(
    test_image_paths: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    summary: dict,
    results_dir: str = RESULTS_DIR,
) -> str:
    """Save per-sample test predictions and global metrics to timestamped CSV."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{timestamp}_test_predictions.csv")

    fieldnames = [
        "timestamp",
        "filename",
        "actual_ph",
        "predicted_ph",
        "abs_error",
        "squared_error",
        "mae",
        "rmse",
        "r2",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path, yt, yp in zip(test_image_paths, y_true, y_pred):
            err = float(yp - yt)
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "filename": os.path.basename(path),
                    "actual_ph": float(yt),
                    "predicted_ph": float(yp),
                    "abs_error": abs(err),
                    "squared_error": err * err,
                    "mae": float(summary["reg_mae"]),
                    "rmse": float(summary["reg_rmse"]),
                    "r2": float(summary["reg_r2"]),
                }
            )
    return out_path


def save_test_predictions_excel(
    test_image_paths: Sequence[str],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    summary: dict,
    results_dir: str = RESULTS_DIR,
) -> str:
    """
    Save test predictions and Excel charts.
    Requires `xlsxwriter` package.
    """
    try:
        import xlsxwriter
    except ModuleNotFoundError:
        print("Excel export skipped: install xlsxwriter (`pip install xlsxwriter`).")
        return ""

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"{timestamp}_test_predictions.xlsx")

    workbook = xlsxwriter.Workbook(out_path)
    ws = workbook.add_worksheet("test_predictions")
    summary_ws = workbook.add_worksheet("summary")

    headers = [
        "sample_index",
        "filename",
        "actual_ph",
        "predicted_ph",
        "residual",
        "abs_error",
        "squared_error",
    ]

    for col, h in enumerate(headers):
        ws.write(0, col, h)

    for i, (path, yt, yp) in enumerate(zip(test_image_paths, y_true, y_pred), start=1):
        residual = float(yp - yt)
        ws.write(i, 0, i)
        ws.write(i, 1, os.path.basename(path))
        ws.write(i, 2, float(yt))
        ws.write(i, 3, float(yp))
        ws.write(i, 4, residual)
        ws.write(i, 5, abs(residual))
        ws.write(i, 6, residual * residual)

    n = len(test_image_paths)

    summary_ws.write_row(0, 0, ["metric", "value"])
    summary_ws.write_row(1, 0, ["MAE", float(summary["reg_mae"])])
    summary_ws.write_row(2, 0, ["RMSE", float(summary["reg_rmse"])])
    summary_ws.write_row(3, 0, ["R2", float(summary["reg_r2"])])

    # Chart 1: Actual vs Predicted pH by sample index
    chart_line = workbook.add_chart({"type": "line"})
    chart_line.add_series(
        {
            "name": "Actual pH",
            "categories": ["test_predictions", 1, 0, n, 0],
            "values": ["test_predictions", 1, 2, n, 2],
        }
    )
    chart_line.add_series(
        {
            "name": "Predicted pH",
            "categories": ["test_predictions", 1, 0, n, 0],
            "values": ["test_predictions", 1, 3, n, 3],
        }
    )
    chart_line.set_title({"name": "Actual vs Predicted pH"})
    chart_line.set_x_axis({"name": "Sample Index"})
    chart_line.set_y_axis({"name": "pH"})
    summary_ws.insert_chart("D2", chart_line, {"x_scale": 1.3, "y_scale": 1.2})

    # Chart 2: Residual plot
    chart_residual = workbook.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
    chart_residual.add_series(
        {
            "name": "Residual (Pred - Actual)",
            "categories": ["test_predictions", 1, 0, n, 0],
            "values": ["test_predictions", 1, 4, n, 4],
        }
    )
    chart_residual.set_title({"name": "Residuals by Sample"})
    chart_residual.set_x_axis({"name": "Sample Index"})
    chart_residual.set_y_axis({"name": "Residual"})
    summary_ws.insert_chart("D20", chart_residual, {"x_scale": 1.3, "y_scale": 1.2})

    # Chart 3: Absolute error by sample
    chart_error = workbook.add_chart({"type": "column"})
    chart_error.add_series(
        {
            "name": "Absolute Error",
            "categories": ["test_predictions", 1, 0, n, 0],
            "values": ["test_predictions", 1, 5, n, 5],
        }
    )
    chart_error.set_title({"name": "Absolute Error by Sample"})
    chart_error.set_x_axis({"name": "Sample Index"})
    chart_error.set_y_axis({"name": "Absolute Error"})
    summary_ws.insert_chart("D38", chart_error, {"x_scale": 1.3, "y_scale": 1.2})

    workbook.close()
    return out_path


def main() -> None:
    # 1) Collect image files.
    image_paths = list_image_files(IMAGE_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in '{IMAGE_DIR}'.")

    # 2) Build segmentation pairs (image + mask).
    mask_candidates = list_mask_candidates(MASK_DIR)
    if not mask_candidates:
        raise RuntimeError(
            f"No mask files found in '{MASK_DIR}'. Add Labelme JSON files or mask PNG files."
        )

    seg_pairs = []
    missing_masks = []
    non_json_matches = []

    for path in image_paths:
        mask_path = resolve_mask_path_for_image(path, mask_candidates)
        if mask_path:
            seg_pairs.append((path, mask_path))
            if REQUIRE_JSON_MASKS and not mask_path.lower().endswith(".json"):
                non_json_matches.append(os.path.basename(mask_path))
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            missing_masks.append(stem)

    if REQUIRE_JSON_MASKS and non_json_matches:
        preview = ", ".join(non_json_matches[:10])
        raise RuntimeError(
            f"Matched some masks but they are not JSON in '{MASK_DIR}'. "
            f"Count: {len(non_json_matches)}. Examples: {preview}"
        )

    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise RuntimeError(
            f"Could not match masks for all images in '{MASK_DIR}'. "
            f"Missing count: {len(missing_masks)}. "
            f"Image stems not matched (examples): {preview}"
        )

    print(f"Segmentation samples: {len(seg_pairs)}")

    # 3) Train/val split and data pipeline for segmentation.
    seg_train, seg_val = split_train_val(seg_pairs, val_split=VAL_SPLIT)
    seg_train_ds = make_seg_dataset(seg_train, training=True)
    seg_val_ds = make_seg_dataset(seg_val, training=False)

    # 4) Train segmentation model.
    seg_model = build_deeplabv3plus(input_shape=(SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 3))
    seg_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=bce_dice_loss,
        metrics=[
            dice_coef,
            keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
        ],
    )

    seg_model.fit(
        seg_train_ds,
        validation_data=seg_val_ds,
        epochs=SEG_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint("best_segmentation_deeplabv3plus.keras", save_best_only=True),
        ],
        verbose=1,
    )

    seg_eval = seg_model.evaluate(seg_val_ds, verbose=0)
    print("Segmentation eval [loss, dice, iou]:", seg_eval)

    # 5) Load pH labels for regression.
    if not tf.io.gfile.exists(LABEL_CSV):
        raise RuntimeError("Missing labels.csv with columns: filename,ph")
    labels = load_labels_csv(LABEL_CSV)

    reg_image_paths = []
    reg_ph = []
    for path in image_paths:
        name = os.path.basename(path)
        if name in labels:
            reg_image_paths.append(path)
            reg_ph.append(labels[name])

    if len(reg_image_paths) < 4:
        raise RuntimeError("Not enough labeled samples for regression.")

    # 6) Split regression samples.
    reg_items = list(zip(reg_image_paths, reg_ph))
    reg_train, reg_val = split_train_val(reg_items, val_split=VAL_SPLIT)

    train_paths = [x[0] for x in reg_train]
    train_ph = [x[1] for x in reg_train]
    val_paths = [x[0] for x in reg_val]
    val_ph = [x[1] for x in reg_val]

    # 7) Create masked hydrogel images using segmentation predictions.
    x_train, y_train = build_regression_arrays(train_paths, train_ph, seg_model)
    x_val, y_val = build_regression_arrays(val_paths, val_ph, seg_model)

    reg_train_ds = make_reg_dataset(x_train, y_train, training=True)
    reg_val_ds = make_reg_dataset(x_val, y_val, training=False)

    # 8) Train pH regressor.
    reg_model = build_regressor(input_shape=(REG_IMAGE_SIZE[0], REG_IMAGE_SIZE[1], 3))
    reg_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    reg_model.fit(
        reg_train_ds,
        validation_data=reg_val_ds,
        epochs=REG_EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint("best_ph_regressor.keras", save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ],
        verbose=1,
    )

    # 9) Evaluate regression metrics.
    preds = reg_model.predict(x_val, verbose=0).reshape(-1)
    mae = float(np.mean(np.abs(preds - y_val)))
    rmse = float(math.sqrt(np.mean((preds - y_val) ** 2)))
    r2 = r2_score(y_val, preds)

    print(f"Regression MAE:  {mae:.4f}")
    print(f"Regression RMSE: {rmse:.4f}")
    print(f"Regression R^2:  {r2:.4f}")

    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_dir": IMAGE_DIR,
        "mask_dir": MASK_DIR,
        "label_csv": LABEL_CSV,
        "seg_model": "deeplabv3plus_segmentation",
        "seg_loss": float(seg_eval[0]),
        "seg_dice": float(seg_eval[1]),
        "seg_iou": float(seg_eval[2]),
        "reg_mae": mae,
        "reg_rmse": rmse,
        "reg_r2": r2,
        "num_seg_samples": len(seg_pairs),
        "num_reg_train": len(train_paths),
        "num_reg_val": len(val_paths),
    }
    results_csv = save_results_csv(results)
    test_results_csv = save_test_predictions_csv(val_paths, y_val, preds, results)
    test_results_xlsx = save_test_predictions_excel(val_paths, y_val, preds, results)
    print(f"Saved results CSV: {results_csv}")
    print(f"Saved test predictions CSV: {test_results_csv}")
    if test_results_xlsx:
        print(f"Saved test predictions Excel: {test_results_xlsx}")


if __name__ == "__main__":
    main()
