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
    """Evaluator for model evaluation across different task types."""
    
    def __init__(self, model: tf.keras.Model, task_type: str, num_classes: int = None):
        """Initialize evaluator with model and task type."""
        self.model = model
        self.task_type = task_type
        self.num_classes = num_classes
        
        # Initialize metrics based on task type
        if task_type == 'detection':
            self.metrics = SOTAMetrics.get_detection_metrics_structured(model, num_classes=num_classes or 37)
        elif task_type == 'segmentation':
            if num_classes is None:
                raise ValueError("num_classes must be provided for segmentation task")
            self.metrics = SOTAMetrics.get_segmentation_metrics_structured(model, num_classes=num_classes)
        elif task_type == 'multitask':
            if num_classes is None:
                raise ValueError("num_classes must be provided for multitask evaluation")
            self.metrics = SOTAMetrics.get_multitask_metrics_structured(
                model,
                num_detection_classes=num_classes,
                num_segmentation_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def evaluate(self, dataset: tf.data.Dataset) -> dict[str, Any]:
        """Evaluate model on dataset and return metrics."""
        # Reset all metrics
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                metric.reset_state()
        else:
            self.metrics.reset_state()
        
        # Evaluate on dataset
        for batch in tqdm(dataset, desc="Evaluating", unit="batch"):
            images, targets = batch
            
            # Get model predictions
            predictions = self.model(images, training=False)
            
            # Update metrics
            if isinstance(self.metrics, list):
                for metric in self.metrics:
                    metric.update_state(targets, predictions)
            else:
                self.metrics.update_state(targets, predictions)
        
        # Get metric results
        if isinstance(self.metrics, list):
            results = {}
            for metric in self.metrics:
                results[metric.name] = metric.result().numpy()
            return results
        else:
            return {self.metrics.name: self.metrics.result().numpy()}
