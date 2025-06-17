"""
Optimized custom metrics for object detection and semantic segmentation
"""
import tensorflow as tf
from typing import List, Dict, Union


class IoUMetric(tf.keras.metrics.Metric):
    """Intersection over Union metric for bounding boxes"""
    
    def __init__(self, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        """y_true, y_pred: [batch_size, 4] in format [x1, y1, x2, y2]"""
        # ------------------------------------------------------------------
        # Ensure both tensors share the same dtype (float32) to avoid mixed
        # precision ops mismatch when the model runs in float16. Casting to
        # float32 is safe for metric computation and prevents the TypeError
        # raised by tf.minimum / tf.maximum.
        # ------------------------------------------------------------------
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate intersection
        inter_area = tf.maximum(0.0, 
            tf.minimum(y_true[..., 2], y_pred[..., 2]) - tf.maximum(y_true[..., 0], y_pred[..., 0])
        ) * tf.maximum(0.0, 
            tf.minimum(y_true[..., 3], y_pred[..., 3]) - tf.maximum(y_true[..., 1], y_pred[..., 1])
        )
        
        # Calculate union
        true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
        union_area = true_area + pred_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + tf.keras.backend.epsilon())
        
        self.iou_sum.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.size(iou), tf.float32))
        
    def result(self):
        return self.iou_sum / (self.count + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)


class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient for semantic segmentation"""
    
    def __init__(self, smooth=1e-6, name='dice_coef', **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to binary if multi-class
        if y_pred.shape[-1] > 1:
            y_pred = tf.argmax(y_pred, axis=-1)
            y_true = tf.argmax(y_true, axis=-1)
            
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten for batch-wise calculation
        y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        union = tf.reduce_sum(y_true_flat + y_pred_flat, axis=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum.assign_add(tf.reduce_sum(dice))
        self.count.assign_add(tf.cast(tf.shape(dice)[0], tf.float32))
        
    def result(self):
        return self.dice_sum / (self.count + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)


class DetectionMetrics(tf.keras.metrics.Metric):
    """Combined metrics for object detection tasks"""
    
    def __init__(self, iou_threshold=0.5, name='detection_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_threshold = iou_threshold
        
        # Use built-in metrics where possible
        self.bbox_mae = tf.keras.metrics.MeanAbsoluteError(name='bbox_mae')
        self.cls_accuracy = tf.keras.metrics.CategoricalAccuracy(name='cls_accuracy')
        self.iou_metric = IoUMetric(name='iou')
        
        # Precision/Recall tracking
        self.tp = self.add_weight(name='true_positives', initializer='zeros')
        self.fp = self.add_weight(name='false_positives', initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', initializer='zeros')
    
    def _calculate_iou(self, y_true, y_pred):
        """Helper to calculate IoU"""
        # Cast to float32 to avoid dtype mismatches when mixed precision (float16)
        # is enabled in the model.
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        inter_area = tf.maximum(0.0, 
            tf.minimum(y_true[..., 2], y_pred[..., 2]) - tf.maximum(y_true[..., 0], y_pred[..., 0])
        ) * tf.maximum(0.0, 
            tf.minimum(y_true[..., 3], y_pred[..., 3]) - tf.maximum(y_true[..., 1], y_pred[..., 1])
        )
        
        true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
        pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])
        union_area = true_area + pred_area - inter_area
        
        return inter_area / (union_area + tf.keras.backend.epsilon())
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update all detection metrics"""
        # Helper function to get tensor by multiple possible keys
        def _get_tensor(data_dict, keys):
            for key in keys:
                if key in data_dict:
                    return data_dict[key]
            return None
        
        # Handle different input formats
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            # Multi-output format
            bbox_true = _get_tensor(y_true, ['bbox', 'bbox_output', 'head_bbox', 'bounding_box'])
            bbox_pred = _get_tensor(y_pred, ['bbox', 'bbox_output', 'head_bbox', 'bounding_box'])
            cls_true = _get_tensor(y_true, ['class', 'cls', 'label', 'species', 'class_output'])
            cls_pred = _get_tensor(y_pred, ['class', 'cls', 'label', 'species', 'class_output'])
        else:
            # Single tensor format - assume bbox only
            bbox_true, bbox_pred = y_true, y_pred
            cls_true = cls_pred = None
        
        # Update bbox metrics
        if bbox_true is not None and bbox_pred is not None:
            self.bbox_mae.update_state(bbox_true, bbox_pred)
            self.iou_metric.update_state(bbox_true, bbox_pred)
            
            # Update precision/recall
            iou_scores = self._calculate_iou(bbox_true, bbox_pred)
            tp = tf.reduce_sum(tf.cast(iou_scores >= self.iou_threshold, tf.float32))
            fp = tf.reduce_sum(tf.cast(iou_scores < self.iou_threshold, tf.float32))
            fn = tf.cast(tf.shape(bbox_true)[0], tf.float32) - tp
            
            self.tp.assign_add(tp)
            self.fp.assign_add(fp)
            self.fn.assign_add(fn)
        
        # Update classification metrics
        if cls_true is not None and cls_pred is not None:
            # Handle label format conversion
            if cls_true.shape.rank == 2 and cls_true.shape[-1] == 1:
                cls_true = tf.squeeze(cls_true, axis=-1)
            
            if cls_true.shape.rank == 1:
                num_classes = tf.shape(cls_pred)[-1]
                cls_true = tf.one_hot(tf.cast(cls_true, tf.int32), depth=num_classes)
            
            self.cls_accuracy.update_state(cls_true, cls_pred)
    
    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1_score = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        
        return {
            'iou': self.iou_metric.result(),
            'bbox_mae': self.bbox_mae.result(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'cls_accuracy': self.cls_accuracy.result()
        }
    
    def reset_state(self):
        self.iou_metric.reset_state()
        self.bbox_mae.reset_state()
        self.cls_accuracy.reset_state()
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class SegmentationMetrics(tf.keras.metrics.Metric):
    """Combined metrics for semantic segmentation tasks"""
    
    def __init__(self, num_classes=None, name='segmentation_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        
        # Use built-in metrics
        self.pixel_accuracy = tf.keras.metrics.CategoricalAccuracy(name='pixel_accuracy')
        self.dice_coef = DiceCoefficient(name='dice_coef')
        
        # Use built-in MeanIoU if available
        if num_classes:
            self.mean_iou = tf.keras.metrics.MeanIoU(num_classes=num_classes, name='mean_iou')
        else:
            self.mean_iou = None
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.pixel_accuracy.update_state(y_true, y_pred)
        self.dice_coef.update_state(y_true, y_pred)
        
        if self.mean_iou:
            # Convert to class predictions for MeanIoU
            if y_pred.shape[-1] > 1:
                y_pred_classes = tf.argmax(y_pred, axis=-1)
                y_true_classes = tf.argmax(y_true, axis=-1) if y_true.shape[-1] > 1 else y_true
            else:
                y_pred_classes = tf.cast(y_pred > 0.5, tf.int32)
                y_true_classes = tf.cast(y_true, tf.int32)
            
            self.mean_iou.update_state(y_true_classes, y_pred_classes)
    
    def result(self):
        results = {
            'pixel_accuracy': self.pixel_accuracy.result(),
            'dice_coef': self.dice_coef.result()
        }
        
        if self.mean_iou:
            results['mean_iou'] = self.mean_iou.result()
        
        return results
    
    def reset_state(self):
        self.pixel_accuracy.reset_state()
        self.dice_coef.reset_state()
        if self.mean_iou:
            self.mean_iou.reset_state()


def get_metrics(task_type: str, num_classes: int = None, **kwargs) -> List[tf.keras.metrics.Metric]:
    """Factory function to get optimized metrics based on task type"""
    
    if task_type == 'detection':
        return [DetectionMetrics(**kwargs)]
    
    elif task_type == 'segmentation':
        return [SegmentationMetrics(num_classes=num_classes, **kwargs)]
    
    elif task_type == 'classification':
        # Use built-in metrics for pure classification
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        return metrics
    
    elif task_type == 'multitask':
        # Return both detection and segmentation metrics
        return [
            DetectionMetrics(name='detection', **kwargs),
            SegmentationMetrics(num_classes=num_classes, name='segmentation', **kwargs)
        ]
    
    else:
        raise ValueError(f"Unknown task type: {task_type}. "
                        f"Supported types: 'detection', 'segmentation', 'classification', 'multitask'")
