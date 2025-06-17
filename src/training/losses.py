"""
Optimized and simplified loss functions for object detection and semantic segmentation
Focus on essential losses with maximum use of TensorFlow built-in functions
"""
import tensorflow as tf
from tensorflow.keras import losses
import numpy as np

# Core utility functions
def safe_divide(num, denom, eps=1e-7):
    """Safe division avoiding zero division."""
    return num / (denom + eps)

def bbox_areas(boxes):
    """Compute bbox areas efficiently."""
    return tf.maximum(1e-7, (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]))

def bbox_intersection(boxes1, boxes2):
    """Compute intersection area between bboxes."""
    lt = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    rb = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
    wh = tf.maximum(0.0, rb - lt)
    return wh[..., 0] * wh[..., 1]


class IoULoss(losses.Loss):
    """Unified IoU Loss supporting IoU, GIoU, DIoU, CIoU variants."""
    
    def __init__(self, mode='ciou', reduction='sum_over_batch_size', name=None):
        super().__init__(reduction=reduction, name=name or f'{mode}_loss')
        self.mode = mode.lower()
        
    @tf.function
    def call(self, y_true, y_pred):
        # Basic IoU calculation
        intersection = bbox_intersection(y_true, y_pred)
        area1, area2 = bbox_areas(y_true), bbox_areas(y_pred)
        union = area1 + area2 - intersection
        iou = safe_divide(intersection, union)
        
        if self.mode == 'iou':
            return 1.0 - iou
            
        # Enclosing box for GIoU/DIoU/CIoU
        enclose_lt = tf.minimum(y_true[..., :2], y_pred[..., :2])
        enclose_rb = tf.maximum(y_true[..., 2:], y_pred[..., 2:])
        enclose_wh = tf.maximum(1e-7, enclose_rb - enclose_lt)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        
        if self.mode == 'giou':
            giou = iou - safe_divide(enclose_area - union, enclose_area)
            return 1.0 - giou
            
        # Center coordinates for DIoU/CIoU
        center1 = (y_true[..., :2] + y_true[..., 2:]) * 0.5
        center2 = (y_pred[..., :2] + y_pred[..., 2:]) * 0.5
        center_dist = tf.reduce_sum(tf.square(center1 - center2), -1)
        diagonal = tf.reduce_sum(tf.square(enclose_wh), -1) + 1e-7
        
        if self.mode == 'diou':
            diou = iou - safe_divide(center_dist, diagonal)
            return 1.0 - diou
            
        if self.mode == 'ciou':
            # Aspect ratio penalty
            wh1 = tf.maximum(1e-7, y_true[..., 2:] - y_true[..., :2])
            wh2 = tf.maximum(1e-7, y_pred[..., 2:] - y_pred[..., :2])
            v = (4 / np.pi**2) * tf.square(
                tf.atan(safe_divide(wh1[..., 0], wh1[..., 1])) -
                tf.atan(safe_divide(wh2[..., 0], wh2[..., 1]))
            )
            alpha = tf.stop_gradient(safe_divide(v, 1 - iou + v))
            ciou = iou - safe_divide(center_dist, diagonal) - alpha * v
            return 1.0 - ciou


class FocalLoss(losses.Loss):
    """Optimized Focal Loss using built-in cross entropy."""
    
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, 
                 reduction='sum_over_batch_size', name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    @tf.function
    def call(self, y_true, y_pred):
        # Use built-in cross entropy
        ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) if self.from_logits else \
             tf.keras.backend.binary_crossentropy(y_true, y_pred)
             
        # Compute p_t efficiently
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        if self.from_logits:
            p_t = tf.nn.sigmoid(p_t)
            
        # Focal weight
        alpha_t = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        return focal_weight * ce


class DiceLoss(losses.Loss):
    """Simplified Dice Loss using TensorFlow ops."""
    
    def __init__(self, smooth=1.0, reduction='sum_over_batch_size', name='dice_loss'):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        
    @tf.function
    def call(self, y_true, y_pred):
        # Flatten for easier computation
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice = safe_divide(2 * intersection + self.smooth, 
                          tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + self.smooth)
        return 1.0 - dice


class TverskyLoss(losses.Loss):
    """Tversky Loss - generalization of Dice loss."""
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0, 
                 reduction='sum_over_batch_size', name='tversky_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta  
        self.smooth = smooth
        
    @tf.function
    def call(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        tp = tf.reduce_sum(y_true_f * y_pred_f)
        fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
        fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
        
        tversky = safe_divide(tp + self.smooth, 
                             tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CombinedLoss(losses.Loss):
    """Flexible combined loss for detection/segmentation tasks."""
    
    def __init__(self, losses_config, reduction='sum_over_batch_size', name='combined_loss'):
        """
        losses_config: dict like {'focal': 1.0, 'dice': 0.5, 'ciou': 2.0}
        """
        super().__init__(reduction=reduction, name=name)
        self.losses_config = losses_config
        self.loss_fns = self._build_losses()
        
    def _build_losses(self):
        """Build loss functions from config."""
        loss_map = {
            'focal': FocalLoss(),
            'dice': DiceLoss(),
            'tversky': TverskyLoss(),
            'iou': IoULoss('iou'),
            'giou': IoULoss('giou'), 
            'diou': IoULoss('diou'),
            'ciou': IoULoss('ciou'),
            'mse': losses.MeanSquaredError(reduction='none'),
            'mae': losses.MeanAbsoluteError(reduction='none'),
            'bce': losses.BinaryCrossentropy(reduction='none'),
            'cce': losses.CategoricalCrossentropy(reduction='none'),
        }
        
        return {name: loss_map[name] for name in self.losses_config.keys() 
                if name in loss_map}
    
    @tf.function
    def call(self, y_true, y_pred):
        total_loss = 0.0
        
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            # Handle multi-output case
            for loss_name, weight in self.losses_config.items():
                if loss_name in self.loss_fns:
                    # Try to match keys automatically
                    for true_key in y_true.keys():
                        if true_key in y_pred:
                            loss_val = self.loss_fns[loss_name](y_true[true_key], y_pred[true_key])
                            total_loss += weight * tf.reduce_mean(loss_val)
                            break
        else:
            # Single output case - apply first loss function
            first_loss = list(self.loss_fns.values())[0]
            total_loss = first_loss(y_true, y_pred)
            
        return total_loss


# Simplified factory function
def get_loss(loss_type, **kwargs):
    """Factory function for loss creation."""
    loss_registry = {
        'iou': lambda: IoULoss('iou', **kwargs),
        'giou': lambda: IoULoss('giou', **kwargs), 
        'diou': lambda: IoULoss('diou', **kwargs),
        'ciou': lambda: IoULoss('ciou', **kwargs),
        'focal': lambda: FocalLoss(**kwargs),
        'dice': lambda: DiceLoss(**kwargs),
        'tversky': lambda: TverskyLoss(**kwargs),
        'combined': lambda: CombinedLoss(**kwargs),
        # Built-in losses
        'mse': lambda: losses.MeanSquaredError(**kwargs),
        'mae': lambda: losses.MeanAbsoluteError(**kwargs),
        'bce': lambda: losses.BinaryCrossentropy(**kwargs),
        'cce': lambda: losses.CategoricalCrossentropy(**kwargs),
        'scce': lambda: losses.SparseCategoricalCrossentropy(**kwargs),
    }
    
    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss: {loss_type}. Available: {list(loss_registry.keys())}")
    
    return loss_registry[loss_type]()
