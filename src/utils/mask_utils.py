import tensorflow as tf
from typing import Tuple

"""mask_utils.py
Utility helpers for segmentation mask manipulation and metrics.
All functions operate on TensorFlow tensors for easy tf.data / Keras use.
"""

__all__ = [
    "one_hot_encode",
    "decode_one_hot",
    "resize_mask",
    "compute_dice",
    "compute_iou",
]


def one_hot_encode(mask: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Convert H×W integer mask → one-hot encoded H×W×C tensor."""
    mask = tf.cast(mask, tf.int32)
    return tf.one_hot(mask, depth=num_classes, dtype=tf.float32)


def decode_one_hot(mask: tf.Tensor) -> tf.Tensor:
    """Argmax channel dimension to obtain H×W class ids."""
    return tf.argmax(mask, axis=-1, output_type=tf.int32)


def resize_mask(mask: tf.Tensor, size: Tuple[int, int]) -> tf.Tensor:
    """Nearest-neighbour resize for masks preserving class ids."""
    return tf.image.resize(mask, size, method="nearest")


def compute_dice(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Dice coefficient for one-hot encoded masks."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def compute_iou(y_true: tf.Tensor, y_pred: tf.Tensor, num_classes: int) -> tf.Tensor:
    """Mean IoU for one-hot masks."""
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes)
    mean_iou.update_state(y_true, y_pred)
    return mean_iou.result()