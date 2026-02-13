import math
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data_preprocessing import (
    augment_image_and_mask,
    augment_regression_image,
    get_mask_path,
    list_image_files,
    load_labels_csv,
    read_image,
    read_mask,
    split_train_val,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

# Paths and training config
IMAGE_DIR = "train"
MASK_DIR = "masks"
LABEL_CSV = "labels.csv"

SEG_IMAGE_SIZE = (512, 512)
REG_IMAGE_SIZE = (224, 224)

BATCH_SIZE = 4
SEG_EPOCHS = 40
REG_EPOCHS = 60
VAL_SPLIT = 0.2


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


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_unet(input_shape=(512, 512, 3)) -> keras.Model:
    """U-Net segmentation model."""
    inputs = keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)

    u4 = layers.UpSampling2D()(bn)
    u4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(u4, 256)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128)

    u2 = layers.UpSampling2D()(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64)

    u1 = layers.UpSampling2D()(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)
    return keras.Model(inputs, outputs, name="unet_segmentation")


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


def main() -> None:
    # 1) Collect image files.
    image_paths = list_image_files(IMAGE_DIR)
    if not image_paths:
        raise RuntimeError(f"No images found in '{IMAGE_DIR}'.")

    # 2) Build segmentation pairs (image + mask).
    seg_pairs = []
    missing_masks = []
    for path in image_paths:
        mask_path = get_mask_path(path, MASK_DIR)
        if tf.io.gfile.exists(mask_path):
            seg_pairs.append((path, mask_path))
        else:
            missing_masks.append(mask_path)

    if not seg_pairs:
        raise RuntimeError(
            f"No masks found in '{MASK_DIR}'. Expected names like IMG_6054_mask.png."
        )

    print(f"Segmentation samples: {len(seg_pairs)}")
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images skipped due to missing masks.")

    # 3) Train/val split and data pipeline for segmentation.
    seg_train, seg_val = split_train_val(seg_pairs, val_split=VAL_SPLIT)
    seg_train_ds = make_seg_dataset(seg_train, training=True)
    seg_val_ds = make_seg_dataset(seg_val, training=False)

    # 4) Train segmentation model.
    seg_model = build_unet(input_shape=(SEG_IMAGE_SIZE[0], SEG_IMAGE_SIZE[1], 3))
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
            keras.callbacks.ModelCheckpoint("best_segmentation_unet.keras", save_best_only=True),
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


if __name__ == "__main__":
    main()
