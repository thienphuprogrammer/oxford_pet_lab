from __future__ import annotations

"""evaluator.py
General-purpose Evaluation helper used across detection, segmentation and
multitask pipelines. The evaluator is intentionally light-weight so it can be
used both inside notebooks and scripts.
"""

from typing import Dict, Any
import tensorflow as tf
from tqdm import tqdm

from src.training.metrics import (
    SOTAMetrics,
)

__all__ = ["Evaluator"]


class Evaluator:
    """Run model evaluation on a *tf.data.Dataset* using specified task type.

    Parameters
    ----------
    model
        A compiled ``tf.keras.Model`` with the same output signature as used
        during training.
    task_type
        One of ``{"detection", "segmentation", "multitask"}``.
    num_classes
        Number of segmentation classes (only needed for segmentation / multitask).
    """

    def __init__(self, model: tf.keras.Model, task_type: str, num_classes: int | None = None):
        self.model = model
        self.task_type = task_type.lower()

        if self.task_type == "detection":
            self.metric = SOTAMetrics.get_detection_metrics_structured(model, num_classes=num_classes or 37)
        elif self.task_type == "segmentation":
            if num_classes is None:
                raise ValueError("num_classes must be provided for segmentation task")
            self.metric = SOTAMetrics.get_segmentation_metrics(num_classes=num_classes)
        elif self.task_type == "multitask":
            if num_classes is None:
                raise ValueError("num_classes must be provided for multitask")
            self.metric = SOTAMetrics.get_multitask_metrics(
                num_detection_classes=num_classes,
                num_segmentation_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def evaluate(self, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Iterate over *dataset* and compute metrics."""
        self.metric.reset_state()

        for batch in tqdm(dataset, desc="Evaluating", unit="batch"):
            if self.task_type in {"detection", "segmentation"}:
                images, targets = batch
                preds = self.model(images, training=False)
                self.metric.update_state(targets, preds)
            else:  # multitask expects tuple of dicts
                images, targets = batch
                detection_targets, segmentation_targets = targets
                detection_preds, segmentation_preds = self.model(images, training=False)

                self.metric.detection_iou.update_state(detection_targets["bbox"], detection_preds["bbox"])
                self.metric.detection_mae.update_state(detection_targets["bbox"], detection_preds["bbox"])
                self.metric.seg_dice.update_state(segmentation_targets, segmentation_preds)
                self.metric.seg_pixel_acc.update_state(segmentation_targets, segmentation_preds)

        return self.metric.result()
