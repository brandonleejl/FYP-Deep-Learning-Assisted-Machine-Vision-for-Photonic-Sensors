from typing import Tuple

import tensorflow as tf


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
