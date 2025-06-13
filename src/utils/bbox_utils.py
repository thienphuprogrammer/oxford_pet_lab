import tensorflow as tf
from typing import Tuple

"""bbox_utils.py
Utility functions for bounding–box manipulations and metrics used by the
object–detection pipeline.

NOTE: All public helpers operate on TensorFlow tensors to avoid unnecessary
conversion in the tf.data pipeline / model graph.
"""

__all__ = [
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "normalize_bbox",
    "denormalize_bbox",
    "compute_iou",
    "non_max_suppression",
]

# -----------------------------------------------------------------------------
# Conversion helpers
# -----------------------------------------------------------------------------

def xyxy_to_xywh(bbox: tf.Tensor) -> tf.Tensor:
    """Convert [x_min, y_min, x_max, y_max] → [x_center, y_center, w, h].

    Args:
        bbox: tensor of shape (..., 4) in absolute pixel coords.
    Returns:
        Tensor in the same dtype/shape as *bbox* with converted format.
    """
    x_min, y_min, x_max, y_max = tf.split(tf.cast(bbox, tf.float32), 4, axis=-1)
    w = x_max - x_min
    h = y_max - y_min
    x_c = x_min + w / 2.0
    y_c = y_min + h / 2.0
    return tf.concat([x_c, y_c, w, h], axis=-1)


def xywh_to_xyxy(bbox: tf.Tensor) -> tf.Tensor:
    """Inverse of :pyfunc:`xyxy_to_xywh`."""
    x_c, y_c, w, h = tf.split(tf.cast(bbox, tf.float32), 4, axis=-1)
    half_w = w / 2.0
    half_h = h / 2.0
    x_min = x_c - half_w
    y_min = y_c - half_h
    x_max = x_c + half_w
    y_max = y_c + half_h
    return tf.concat([x_min, y_min, x_max, y_max], axis=-1)


# -----------------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------------

def normalize_bbox(bbox: tf.Tensor, img_shape: Tuple[int, int]) -> tf.Tensor:
    """Normalise absolute bbox using *img_shape* `(height, width)` to [0-1]."""
    h, w = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)
    x_min, y_min, x_max, y_max = tf.split(tf.cast(bbox, tf.float32), 4, axis=-1)
    return tf.concat([x_min / w, y_min / h, x_max / w, y_max / h], axis=-1)


def denormalize_bbox(bbox: tf.Tensor, img_shape: Tuple[int, int]) -> tf.Tensor:
    """Inverse of :pyfunc:`normalize_bbox`."""
    h, w = tf.cast(img_shape[0], tf.float32), tf.cast(img_shape[1], tf.float32)
    x_min, y_min, x_max, y_max = tf.split(tf.cast(bbox, tf.float32), 4, axis=-1)
    return tf.concat([x_min * w, y_min * h, x_max * w, y_max * h], axis=-1)


# -----------------------------------------------------------------------------
# Metrics / post-processing
# -----------------------------------------------------------------------------

def compute_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """Compute IoU between two sets of bboxes.

    Args:
        boxes1: [..., 4] in *xyxy* format
        boxes2: [..., 4] broadcast-compatible with *boxes1*
    Returns:
        IoU tensor broadcast over leading dims.
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)

    x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = tf.maximum(x2 - x1, 0.0)
    inter_h = tf.maximum(y2 - y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter_area + 1e-8
    return inter_area / union


def non_max_suppression(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    max_output_size: int = 100,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
):
    """TensorFlow NMS wrapper returning filtered indices."""
    return tf.image.non_max_suppression(
        boxes=tf.cast(boxes, tf.float32),
        scores=tf.cast(scores, tf.float32),
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        name="nms",
    )