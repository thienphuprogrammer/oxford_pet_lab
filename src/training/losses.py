"""
Custom loss functions for object detection and semantic segmentation
"""
import tensorflow as tf
import numpy as np


class SmoothL1Loss(tf.keras.losses.Loss):
    """Smooth L1 loss for bounding box regression"""
    
    def __init__(self, beta=1.0, reduction=tf.keras.losses.Reduction.AUTO, name='smooth_l1_loss'):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        
    def call(self, y_true, y_pred):
        diff = tf.abs(y_true - y_pred)
        loss = tf.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        return tf.reduce_mean(loss, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config


class GIoULoss(tf.keras.losses.Loss):
    """Generalized IoU loss for bounding box regression"""
    
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='giou_loss'):
        super().__init__(reduction=reduction, name=name)
        
    def call(self, y_true, y_pred):
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
        
        # Calculate enclosing box
        x1_min = tf.minimum(y_true[..., 0], y_pred[..., 0])
        y1_min = tf.minimum(y_true[..., 1], y_pred[..., 1])
        x2_max = tf.maximum(y_true[..., 2], y_pred[..., 2])
        y2_max = tf.maximum(y_true[..., 3], y_pred[..., 3])
        
        enclosing_area = (x2_max - x1_min) * (y2_max - y1_min)
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
        
        # Return loss (1 - GIoU)
        return 1.0 - giou


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, 
                 reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
            
        # Calculate cross entropy
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        
        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class DiceLoss(tf.keras.losses.Loss):
    """Dice loss for semantic segmentation"""
    
    def __init__(self, smooth=1.0, reduction=tf.keras.losses.Reduction.AUTO, name='dice_loss'):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice_coef
    
    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config


class TverskyLoss(tf.keras.losses.Loss):
    """Tversky loss for handling class imbalance in segmentation"""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, 
                 reduction=tf.keras.losses.Reduction.AUTO, name='tversky_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def call(self, y_true, y_pred):
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate True Positives, False Positives, False Negatives
        tp = tf.reduce_sum(y_true_flat * y_pred_flat)
        fp = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
        fn = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
        
        # Calculate Tversky index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Return Tversky loss
        return 1.0 - tversky_index
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'smooth': self.smooth
        })
        return config


class CombinedSegmentationLoss(tf.keras.losses.Loss):
    """Combined loss for segmentation (CrossEntropy + Dice)"""
    
    def __init__(self, ce_weight=0.5, dice_weight=0.5, 
                 reduction=tf.keras.losses.Reduction.AUTO, name='combined_seg_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy()
        self.dice_loss = DiceLoss()
        
    def call(self, y_true, y_pred):
        ce_loss = self.ce_loss(y_true, y_pred)
        dice_loss = self.dice_loss(y_true, y_pred)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'ce_weight': self.ce_weight,
            'dice_weight': self.dice_weight
        })
        return config


class MultiTaskLoss(tf.keras.losses.Loss):
    """Multi-task loss for joint detection and segmentation"""
    
    def __init__(self, detection_weight=1.0, segmentation_weight=1.0,
                 bbox_loss='smooth_l1', cls_loss='focal', seg_loss='combined',
                 reduction=tf.keras.losses.Reduction.AUTO, name='multitask_loss'):
        super().__init__(reduction=reduction, name=name)
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        
        # Initialize losses
        if bbox_loss == 'smooth_l1':
            self.bbox_loss = SmoothL1Loss()
        elif bbox_loss == 'giou':
            self.bbox_loss = GIoULoss()
        else:
            self.bbox_loss = tf.keras.losses.MeanSquaredError()
            
        if cls_loss == 'focal':
            self.cls_loss = FocalLoss()
        else:
            self.cls_loss = tf.keras.losses.CategoricalCrossentropy()
            
        if seg_loss == 'combined':
            self.seg_loss = CombinedSegmentationLoss()
        elif seg_loss == 'dice':
            self.seg_loss = DiceLoss()
        else:
            self.seg_loss = tf.keras.losses.CategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        """
        y_true: dict with keys 'bbox', 'class', 'mask'
        y_pred: dict with keys 'bbox', 'class', 'mask'
        """
        # Detection losses
        bbox_loss = self.bbox_loss(y_true['bbox'], y_pred['bbox'])
        cls_loss = self.cls_loss(y_true['class'], y_pred['class'])
        detection_loss = bbox_loss + cls_loss
        
        # Segmentation loss
        seg_loss = self.seg_loss(y_true['mask'], y_pred['mask'])
        
        # Combined loss
        total_loss = (self.detection_weight * detection_loss + 
                     self.segmentation_weight * seg_loss)
        
        return total_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'detection_weight': self.detection_weight,
            'segmentation_weight': self.segmentation_weight
        })
        return config


class IoULoss(tf.keras.losses.Loss):
    """IoU loss for bounding box regression"""
    
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='iou_loss'):
        super().__init__(reduction=reduction, name=name)
        
    def call(self, y_true, y_pred):
        """Calculate IoU loss between predicted and true bounding boxes"""
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
        
        # Return loss (1 - IoU)
        return 1.0 - iou

class DetectionLoss(tf.keras.losses.Loss):
    def __init__(self, bbox_loss_weight=1.0, cls_loss_weight=1.0, iou_loss_weight=1.0, reduction=tf.keras.losses.Reduction.AUTO, name='detection_loss'):
        super().__init__(reduction=reduction, name=name)
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.iou_loss_weight = iou_loss_weight
        
    def call(self, y_true, y_pred):
        bbox_loss = tf.keras.losses.MeanSquaredError()(y_true['bbox'], y_pred['bbox'])
        cls_loss = tf.keras.losses.CategoricalCrossentropy()(y_true['class'], y_pred['class'])
        iou_loss = IoULoss()(y_true['bbox'], y_pred['bbox'])
        
        return self.bbox_loss_weight * bbox_loss + self.cls_loss_weight * cls_loss + self.iou_loss_weight * iou_loss


class SegmentationLoss(tf.keras.losses.Loss):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, focal_weight=0.0, reduction=tf.keras.losses.Reduction.AUTO, name='segmentation_loss'):
        super().__init__(reduction=reduction, name=name)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def call(self, y_true, y_pred):
        ce_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
        dice_loss = DiceLoss()(y_true, y_pred)
        focal_loss = FocalLoss()(y_true, y_pred)
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss

def get_loss_function(loss_config):
    """Factory function to get loss function based on configuration"""
    loss_type = loss_config.get('type', 'mse')
    
    if loss_type == 'smooth_l1':
        return SmoothL1Loss(beta=loss_config.get('beta', 1.0))
    elif loss_type == 'giou':
        return GIoULoss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'dice':
        return DiceLoss(smooth=loss_config.get('smooth', 1.0))
    elif loss_type == 'tversky':
        return TverskyLoss(
            alpha=loss_config.get('alpha', 0.7),
            beta=loss_config.get('beta', 0.3)
        )
    elif loss_type == 'combined_seg':
        return CombinedSegmentationLoss(
            ce_weight=loss_config.get('ce_weight', 0.5),
            dice_weight=loss_config.get('dice_weight', 0.5)
        )
    elif loss_type == 'multitask':
        return MultiTaskLoss(
            detection_weight=loss_config.get('detection_weight', 1.0),
            segmentation_weight=loss_config.get('segmentation_weight', 1.0)
        )
    elif loss_type == 'iou':
        return IoULoss()
    elif loss_type == 'categorical_crossentropy':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss_type == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_type == 'mse':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Loss weights for different tasks
DETECTION_LOSS_WEIGHTS = {
    'bbox_regression': 1.0,
    'classification': 1.0,
    'objectness': 1.0
}

SEGMENTATION_LOSS_WEIGHTS = {
    'pixel_classification': 1.0,
    'boundary_loss': 0.5
}

MULTITASK_LOSS_WEIGHTS = {
    'detection': 1.0,
    'segmentation': 1.0,
    'shared_features': 0.1
}