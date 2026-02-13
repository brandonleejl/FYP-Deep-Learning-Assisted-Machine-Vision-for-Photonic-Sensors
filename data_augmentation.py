from typing import Tuple

import tensorflow as tf


def configure_gpu_only() -> None:
    """
    Enforce GPU-only execution for this module.
    Raises if no GPU is detected to avoid CPU fallback.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "No GPU detected. GPU-only mode is enabled; CPU fallback is disabled."
        )
    tf.config.set_visible_devices(gpus, "GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


configure_gpu_only()


def augment_image_and_mask(image: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply identical geometric augmentations to image and mask for segmentation."""
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
    """Photometric augmentation for pH regression model."""
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image
