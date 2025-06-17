import tensorflow as tf
from tensorflow.keras import metrics
from typing import List, Dict, Any

class SOTAMetrics:
    """State-of-the-art metrics for object detection and segmentation."""
    
    @staticmethod
    def get_detection_metrics_structured(model: tf.keras.Model, num_classes: int = 37) -> List[metrics.Metric]:
        """Get structured detection metrics."""
        return [
            metrics.MeanIoU(num_classes=num_classes, name='detection_iou'),
            metrics.MeanAbsoluteError(name='detection_mae'),
            metrics.CategoricalAccuracy(name='detection_accuracy')
        ]
    
    @staticmethod
    def get_segmentation_metrics_structured(model: tf.keras.Model, num_classes: int) -> List[metrics.Metric]:
        """Get structured segmentation metrics."""
        return [
            metrics.MeanIoU(num_classes=num_classes, name='segmentation_iou'),
            metrics.CategoricalAccuracy(name='segmentation_accuracy'),
            metrics.Precision(name='segmentation_precision'),
            metrics.Recall(name='segmentation_recall')
        ]
    
    @staticmethod
    def get_multitask_metrics_structured(
        model: tf.keras.Model,
        num_detection_classes: int = 37,
        num_segmentation_classes: int = 2
    ) -> List[metrics.Metric]:
        """Get structured multitask metrics."""
        return [
            # Detection metrics
            metrics.MeanIoU(num_classes=num_detection_classes, name='detection_iou'),
            metrics.MeanAbsoluteError(name='detection_mae'),
            metrics.CategoricalAccuracy(name='detection_accuracy'),
            
            # Segmentation metrics
            metrics.MeanIoU(num_classes=num_segmentation_classes, name='segmentation_iou'),
            metrics.CategoricalAccuracy(name='segmentation_accuracy'),
            metrics.Precision(name='segmentation_precision'),
            metrics.Recall(name='segmentation_recall')
        ] 