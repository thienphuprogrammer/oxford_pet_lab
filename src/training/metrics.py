"""
Custom metrics for object detection and semantic segmentation
"""
import tensorflow as tf
from typing import List

class IoUMetric(tf.keras.metrics.Metric):
    """Intersection over Union metric for bounding boxes"""
    
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: [batch_size, 4] in format [x1, y1, x2, y2]
        """
        # Calculate intersection
        x1_max = tf.maximum(y_true[..., 0], y_pred[..., 0])
        y1_max = tf.maximum(y_true[..., 1], y_pred[..., 1])
        x2_min = tf.minimum(y_true[..., 2], y_pred[..., 2])
        y2_min = tf.minimum(y_true[..., 3], y_pred[..., 3])
        
        intersection_area = tf.maximum(0.0, x2_min - x1_max) * tf.maximum(0.0, y2_min - y1_max)
        
        # Calculate areas
        true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
        union_area = true_area + pred_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / (union_area + 1e-8)
        
        # Update state
        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.size(iou), tf.float32))
        
    def result(self):
        return self.total_iou / self.count
        
    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


class MeanAverageError(tf.keras.metrics.Metric):
    """Mean Average Error for bounding box regression"""
    
    def __init__(self, name='mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
        self.total_error.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.shape(error)[0], tf.float32))
        
    def result(self):
        return self.total_error / self.count
        
    def reset_state(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)


class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient for semantic segmentation"""
    
    def __init__(self, smooth=1.0, name='dice_coef', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.total_dice = self.add_weight(name='total_dice', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to binary if needed
        if y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
            
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        
        # Calculate intersection and union for each sample
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        
        # Calculate Dice coefficient for each sample
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Update state
        self.total_dice.assign_add(tf.reduce_sum(dice_coef))
        self.count.assign_add(tf.cast(tf.shape(dice_coef)[0], tf.float32))
        
    def result(self):
        return self.total_dice / self.count
        
    def reset_state(self):
        self.total_dice.assign(0.0)
        self.count.assign(0.0)


class PixelAccuracy(tf.keras.metrics.Metric):
    """Pixel accuracy for semantic segmentation"""
    
    def __init__(self, name='pixel_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class predictions
        if y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
            
        # Calculate correct predictions
        correct = tf.cast(tf.equal(y_true, y_pred), tf.float32)
        
        # Update state
        self.total_correct.assign_add(tf.reduce_sum(correct))
        self.total_pixels.assign_add(tf.cast(tf.size(correct), tf.float32))
        
    def result(self):
        return self.total_correct / self.total_pixels
        
    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_pixels.assign(0.0)


class MeanIoU(tf.keras.metrics.Metric):
    """Mean IoU for semantic segmentation"""
    
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name='total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class predictions
        if y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
            
        # Flatten predictions
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate confusion matrix
        cm = tf.math.confusion_matrix(
            y_true_flat, y_pred_flat, num_classes=self.num_classes
        )
        
        # Update total confusion matrix
        self.total_cm.assign_add(tf.cast(cm, tf.float32))
        
    def result(self):
        # Calculate IoU for each class
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        true_positives = tf.linalg.diag_part(self.total_cm)
        
        # IoU = TP / (TP + FP + FN)
        denominator = sum_over_row + sum_over_col - true_positives
        iou = true_positives / (denominator + 1e-8)
        
        # Return mean IoU
        return tf.reduce_mean(iou)
        
    def reset_state(self):
        self.total_cm.assign(tf.zeros_like(self.total_cm))


class PrecisionAtIoU(tf.keras.metrics.Metric):
    """Precision at specific IoU threshold for object detection"""
    
    def __init__(self, iou_threshold=0.5, name='precision_at_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate IoU
        iou = self._calculate_iou(y_true, y_pred)
        
        # Determine true positives and false positives
        tp = tf.reduce_sum(tf.cast(iou >= self.iou_threshold, tf.float32))
        fp = tf.reduce_sum(tf.cast(iou < self.iou_threshold, tf.float32))
        
        # Update state
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        
    def _calculate_iou(self, y_true, y_pred):
        """Calculate IoU between predicted and true bounding boxes"""
        # Implementation similar to IoUMetric
        x1_max = tf.maximum(y_true[..., 0], y_pred[..., 0])
        y1_max = tf.maximum(y_true[..., 1], y_pred[..., 1])
        x2_min = tf.minimum(y_true[..., 2], y_pred[..., 2])
        y2_min = tf.minimum(y_true[..., 3], y_pred[..., 3])
        
        intersection_area = tf.maximum(0.0, x2_min - x1_max) * tf.maximum(0.0, y2_min - y1_max)
        
        true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
        union_area = true_area + pred_area - intersection_area
        
        return intersection_area / (union_area + 1e-8)
        
    def result(self):
        return self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)


class RecallAtIoU(tf.keras.metrics.Metric):
    """Recall at specific IoU threshold for object detection"""
    
    def __init__(self, iou_threshold=0.5, name='recall_at_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate IoU
        iou = self._calculate_iou(y_true, y_pred)
        
        # Determine true positives and false negatives
        tp = tf.reduce_sum(tf.cast(iou >= self.iou_threshold, tf.float32))
        total_positives = tf.cast(tf.shape(y_true)[0], tf.float32)
        fn = total_positives - tp
        
        # Update state
        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)
        
    def _calculate_iou(self, y_true, y_pred):
        """Calculate IoU between predicted and true bounding boxes"""
        x1_max = tf.maximum(y_true[..., 0], y_pred[..., 0])
        y1_max = tf.maximum(y_true[..., 1], y_pred[..., 1])
        x2_min = tf.minimum(y_true[..., 2], y_pred[..., 2])
        y2_min = tf.minimum(y_true[..., 3], y_pred[..., 3])
        
        intersection_area = tf.maximum(0.0, x2_min - x1_max) * tf.maximum(0.0, y2_min - y1_max)
        
        true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
        union_area = true_area + pred_area - intersection_area
        
        return intersection_area / (union_area + 1e-8)
        
    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        
    def reset_state(self):
        self.true_positives.assign(0.0)
        self.false_negatives.assign(0.0)


class MultiTaskMetrics(tf.keras.metrics.Metric):
    """Combined metrics for multitask learning"""
    
    def __init__(self, num_classes: int = None, name='multitask_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

        # Detection metrics
        self.detection_iou = IoUMetric(name='detection_iou')
        self.detection_mae = MeanAverageError(name='detection_mae')
        
        # Segmentation metrics
        self.seg_dice = DiceCoefficient(name='seg_dice')
        self.seg_pixel_acc = PixelAccuracy(name='seg_pixel_acc')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update detection metrics
        self.detection_iou.update_state(y_true['bbox'], y_pred['bbox'])
        self.detection_mae.update_state(y_true['bbox'], y_pred['bbox'])
        
        # Update segmentation metrics
        self.seg_dice.update_state(y_true['mask'], y_pred['mask'])
        self.seg_pixel_acc.update_state(y_true['mask'], y_pred['mask'])
        
    def result(self):
        return {
            'detection_iou': self.detection_iou.result(),
            'detection_mae': self.detection_mae.result(),
            'seg_dice': self.seg_dice.result(),
            'seg_pixel_acc': self.seg_pixel_acc.result()
        }
        
    def reset_state(self):
        self.detection_iou.reset_state()
        self.detection_mae.reset_state()
        self.seg_dice.reset_state()
        self.seg_pixel_acc.reset_state()


class DetectionMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int = None, name='detection_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou = IoUMetric()
        self.mae = MeanAverageError()
        self.precision = PrecisionAtIoU(iou_threshold=0.5)
        self.recall = RecallAtIoU(iou_threshold=0.5)
        self.classification_acc = tf.keras.metrics.CategoricalAccuracy(name='classification_acc')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.iou.update_state(y_true['bbox'], y_pred['bbox'])
        self.mae.update_state(y_true['bbox'], y_pred['bbox'])
        self.precision.update_state(y_true['bbox'], y_pred['bbox'])
        self.recall.update_state(y_true['bbox'], y_pred['bbox'])
        self.classification_acc.update_state(y_true['cls'], y_pred['cls'])
    
    def result(self):
        return {
            'iou': self.iou.result(),
            'mae': self.mae.result(),
            'precision': self.precision.result(),
            'recall': self.recall.result(),
            'classification_acc': self.classification_acc.result()
        }
    
    def reset_state(self):
        self.iou.reset_state()
        self.mae.reset_state()
        self.precision.reset_state()
        self.recall.reset_state()
        self.classification_acc.reset_state()


class SegmentationMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes: int = None, name='segmentation_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dice = DiceCoefficient()
        self.pixel_acc = PixelAccuracy()
        self.mean_iou = MeanIoU(num_classes=num_classes) if num_classes is not None else None
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.dice.update_state(y_true, y_pred)
        self.pixel_acc.update_state(y_true, y_pred)
        if self.mean_iou is not None:
            self.mean_iou.update_state(y_true, y_pred)
    
    def result(self):
        return {
            'dice': self.dice.result(),
            'pixel_acc': self.pixel_acc.result(),
            'mean_iou': self.mean_iou.result() if self.mean_iou is not None else None
        }
    
    def reset_state(self):
        self.dice.reset_state()
        self.pixel_acc.reset_state()
        if self.mean_iou is not None:
            self.mean_iou.reset_state()


def get_metrics(task_type='detection', num_classes=None) -> List[tf.keras.metrics.Metric]:
    """Factory function to get metrics based on task type"""
    if task_type == 'detection':
        return [DetectionMetrics(num_classes=num_classes)]
    elif task_type == 'segmentation':
        return [SegmentationMetrics(num_classes=num_classes)]
    elif task_type == 'multitask':
        return [MultiTaskMetrics(num_classes=num_classes)]
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Metric configuration presets
DETECTION_METRICS_CONFIG = {
    'iou_thresholds': [0.3, 0.5, 0.7],
    'metrics': ['iou', 'mae', 'precision', 'recall', 'accuracy']
}

SEGMENTATION_METRICS_CONFIG = {
    'metrics': ['dice', 'pixel_accuracy', 'mean_iou', 'categorical_accuracy']
}

MULTITASK_METRICS_CONFIG = {
    'detection_weight': 0.5,
    'segmentation_weight': 0.5,
    'metrics': ['combined_iou', 'combined_accuracy', 'task_balance']
}