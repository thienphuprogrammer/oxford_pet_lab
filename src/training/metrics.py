import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from ..config.config import Config

class DetectionMetrics:
    """Metrics for object detection."""
    
    @staticmethod
    def iou(y_true, y_pred):
        """Calculate Intersection over Union (IoU) for bounding boxes."""
        # Convert predictions to [y1, x1, y2, x2] format
        pred_boxes = tf.stack([
            y_pred[:, 1], y_pred[:, 0],
            y_pred[:, 3], y_pred[:, 2]
        ], axis=-1)
        
        # Convert true boxes to [y1, x1, y2, x2] format
        true_boxes = tf.stack([
            y_true[:, 1], y_true[:, 0],
            y_true[:, 3], y_true[:, 2]
        ], axis=-1)
        
        # Calculate intersection
        intersect_mins = tf.maximum(pred_boxes[:, :2], true_boxes[:, :2])
        intersect_maxes = tf.minimum(pred_boxes[:, 2:], true_boxes[:, 2:])
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.0)
        intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * \
                   (pred_boxes[:, 3] - pred_boxes[:, 1])
        true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * \
                   (true_boxes[:, 3] - true_boxes[:, 1])
        union_area = pred_area + true_area - intersect_area
        
        # Calculate IoU
        iou = intersect_area / (union_area + K.epsilon())
        return tf.reduce_mean(iou)
    
    @staticmethod
    def mean_iou(y_true, y_pred):
        """Calculate mean IoU over all samples."""
        iou = DetectionMetrics.iou(y_true, y_pred)
        return tf.reduce_mean(iou)
    
    @staticmethod
    def bbox_mae(y_true, y_pred):
        """Calculate Mean Absolute Error for bounding boxes."""
        return tf.reduce_mean(tf.abs(y_true - y_pred))

class SegmentationMetrics:
    """Metrics for semantic segmentation."""
    
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Calculate Dice coefficient."""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    @staticmethod
    def mean_iou(y_true, y_pred, num_classes=3):
        """Calculate mean IoU for segmentation."""
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        iou = []
        for i in range(num_classes):
            intersection = K.sum(
                K.cast(y_true == i, 'float32') * 
                K.cast(y_pred == i, 'float32')
            )
            union = K.sum(
                K.cast(y_true == i, 'float32') + 
                K.cast(y_pred == i, 'float32')
            )
            iou.append(intersection / (union - intersection + K.epsilon()))
        
        return K.mean(iou)
    
    @staticmethod
    def pixel_accuracy(y_true, y_pred):
        """Calculate pixel-wise accuracy."""
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        return K.mean(K.equal(y_true, y_pred))

class MultiTaskMetrics:
    """Metrics for multi-task learning."""
    
    @staticmethod
    def combined_metric(y_true, y_pred):
        """Calculate combined metric for multi-task learning."""
        # Split predictions
        bbox_pred = y_pred[0]
        class_pred = y_pred[1]
        seg_pred = y_pred[2]
        
        # Split true values
        bbox_true = y_true[0]
        class_true = y_true[1]
        seg_true = y_true[2]
        
        # Calculate individual metrics
        bbox_iou = DetectionMetrics.mean_iou(bbox_true, bbox_pred)
        class_acc = tf.keras.metrics.sparse_categorical_accuracy(class_true, class_pred)
        seg_iou = SegmentationMetrics.mean_iou(seg_true, seg_pred)
        
        # Combine metrics with weights
        return 0.3 * bbox_iou + 0.3 * class_acc + 0.4 * seg_iou
