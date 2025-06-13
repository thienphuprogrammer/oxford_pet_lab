import tensorflow as tf
from tensorflow import keras
import numpy as np

class SmoothL1Loss(keras.losses.Loss):
    """Smooth L1 loss for bounding box regression"""
    
    def __init__(self, beta=1.0, name='smooth_l1_loss'):
        super().__init__(name=name)
        self.beta = beta
    
    def call(self, y_true, y_pred):
        """
        Compute Smooth L1 loss
        
        Args:
            y_true: Ground truth bounding boxes [batch_size, 4]
            y_pred: Predicted bounding boxes [batch_size, 4]
        """
        diff = tf.abs(y_true - y_pred)
        
        # Smooth L1 loss
        loss = tf.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        return tf.reduce_mean(loss)

class GIoULoss(keras.losses.Loss):
    """Generalized IoU loss for bounding box regression"""
    
    def __init__(self, name='giou_loss'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        """
        Compute Generalized IoU loss
        
        Args:
            y_true: Ground truth bounding boxes [batch_size, 4] (y_min, x_min, y_max, x_max)
            y_pred: Predicted bounding boxes [batch_size, 4] (y_min, x_min, y_max, x_max)
        """
        # Extract coordinates
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y1_true, x1_true, y2_true, x2_true = tf.split(y_true, 4, axis=-1)
        y1_pred, x1_pred, y2_pred, x2_pred = tf.split(y_pred, 4, axis=-1)
        
        # Calculate intersection
        x1_inter = tf.maximum(x1_true, x1_pred)
        y1_inter = tf.maximum(y1_true, y1_pred)
        x2_inter = tf.minimum(x2_true, x2_pred)
        y2_inter = tf.minimum(y2_true, y2_pred)
        
        inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)
        
        # Calculate union
        true_area = (x2_true - x1_true) * (y2_true - y1_true)
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        union_area = true_area + pred_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-8)
        
        # Calculate enclosing box
        x1_enclosing = tf.minimum(x1_true, x1_pred)
        y1_enclosing = tf.minimum(y1_true, y1_pred)
        x2_enclosing = tf.maximum(x2_true, x2_pred)
        y2_enclosing = tf.maximum(y2_true, y2_pred)
        
        enclosing_area = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
        
        # GIoU loss
        loss = 1.0 - giou
        
        return tf.reduce_mean(loss)

class DiceLoss(keras.losses.Loss):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6, name='dice_loss'):
        super().__init__(name=name)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        """
        Compute Dice loss
        
        Args:
            y_true: Ground truth masks [batch_size, height, width, num_classes]
            y_pred: Predicted masks [batch_size, height, width, num_classes]
        """
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate Dice coefficient
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice

class FocalLoss(keras.losses.Loss):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        """
        Compute Focal loss
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
        """
        # Clip predictions to prevent numerical instability
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)

class CombinedDetectionLoss(keras.losses.Loss):
    """Combined loss for object detection (classification + localization)"""
    
    def __init__(self, 
                 bbox_loss_weight=1.0, 
                 class_loss_weight=1.0,
                 bbox_loss_type='smooth_l1',
                 name='combined_detection_loss'):
        super().__init__(name=name)
        self.bbox_loss_weight = bbox_loss_weight
        self.class_loss_weight = class_loss_weight
        
        # Initialize bbox loss
        if bbox_loss_type == 'smooth_l1':
            self.bbox_loss = SmoothL1Loss()
        elif bbox_loss_type == 'giou':
            self.bbox_loss = GIoULoss()
        else:
            raise ValueError(f"Unknown bbox loss type: {bbox_loss_type}")
        
        # Classification loss
        self.class_loss = keras.losses.CategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        """
        Compute combined detection loss
        
        Args:
            y_true: Dictionary with 'bbox' and 'class' ground truth
            y_pred: Dictionary with 'bbox_output' and 'class_output' predictions
        """
        # Bbox loss
        bbox_loss = self.bbox_loss(y_true['bbox'], y_pred['bbox_output'])
        
        # Classification loss
        class_loss = self.class_loss(y_true['class'], y_pred['class_output'])
        
        # Combined loss
        total_loss = (self.bbox_loss_weight * bbox_loss + 
                     self.class_loss_weight * class_loss)
        
        return total_loss

class CombinedSegmentationLoss(keras.losses.Loss):
    """Combined loss for segmentation (cross-entropy + dice)"""
    
    def __init__(self, 
                 ce_weight=0.5, 
                 dice_weight=0.5,
                 name='combined_segmentation_loss'):
        super().__init__(name=name)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # Initialize losses
        self.ce_loss = keras.losses.CategoricalCrossentropy()
        self.dice_loss = DiceLoss()
    
    def call(self, y_true, y_pred):
        """
        Compute combined segmentation loss
        
        Args:
            y_true: Ground truth segmentation masks
            y_pred: Predicted segmentation masks
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(y_true, y_pred)
        
        # Dice loss
        dice_loss = self.dice_loss(y_true, y_pred)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return total_loss

class IoULoss(keras.losses.Loss):
    """IoU loss for bounding box regression"""
    
    def __init__(self, name='iou_loss'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred):
        """
        Compute IoU loss
        
        Args:
            y_true: Ground truth bounding boxes [batch_size, 4]
            y_pred: Predicted bounding boxes [batch_size, 4]
        """
        # Extract coordinates
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y1_true, x1_true, y2_true, x2_true = tf.split(y_true, 4, axis=-1)
        y1_pred, x1_pred, y2_pred, x2_pred = tf.split(y_pred, 4, axis=-1)
        
        # Calculate intersection
        x1_inter = tf.maximum(x1_true, x1_pred)
        y1_inter = tf.maximum(y1_true, y1_pred)
        x2_inter = tf.minimum(x2_true, x2_pred)
        y2_inter = tf.minimum(y2_true, y2_pred)
        
        inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)
        
        # Calculate union
        true_area = (x2_true - x1_true) * (y2_true - y1_true)
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        union_area = true_area + pred_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-8)
        
        # IoU loss
        loss = 1.0 - iou
        
        return tf.reduce_mean(loss)
