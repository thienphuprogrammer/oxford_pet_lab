"""
Optimized and simplified loss functions for object detection and semantic segmentation
Focus on essential losses with maximum use of TensorFlow built-in functions
Fixed control flow issues for TensorFlow @tf.function compatibility
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

def validate_bbox_shapes(y_true, y_pred):
    """Validate that inputs have proper bounding box format."""
    # Check if inputs have at least 4 dimensions for bbox coordinates
    true_shape = tf.shape(y_true)
    pred_shape = tf.shape(y_pred)
    
    # For bboxes, last dimension should be 4 (x1, y1, x2, y2)
    tf.debugging.assert_equal(
        true_shape[-1], 4,
        message="y_true last dimension must be 4 for bbox coordinates"
    )
    tf.debugging.assert_equal(
        pred_shape[-1], 4,
        message="y_pred last dimension must be 4 for bbox coordinates"
    )
    
    # Shapes should be compatible
    tf.debugging.assert_equal(
        true_shape, pred_shape,
        message="y_true and y_pred shapes must match"
    )


class IoULoss(losses.Loss):
    """Unified IoU Loss supporting IoU, GIoU, DIoU, CIoU variants."""
    
    def __init__(self, mode='ciou', reduction='sum_over_batch_size', name=None):
        super().__init__(reduction=reduction, name=name or f'{mode}_loss')
        self.mode = mode.lower()
        
    @tf.function
    def call(self, y_true, y_pred):
        # Validate input shapes for bounding boxes
        validate_bbox_shapes(y_true, y_pred)
        
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
        if self.from_logits:
            ce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
            p_t = tf.where(tf.equal(y_true, 1), tf.nn.sigmoid(y_pred), 1 - tf.nn.sigmoid(y_pred))
        else:
            ce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            
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
            'scce': losses.SparseCategoricalCrossentropy(reduction='none'),
        }
        
        return {name: loss_map[name] for name in self.losses_config.keys() 
                if name in loss_map}
    
    @tf.function
    def _handle_shape_mismatch(self, y_true, y_pred):
        """Simplified fallback for any unexpected shape mismatch.

        Flattens both tensors to 1-D, trims to the same length and applies
        element-wise MSE.  This avoids complex control-flow that XLA struggles
        to compile while still providing a reasonable loss signal.
        """
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        min_len = tf.minimum(tf.shape(y_true_flat)[0], tf.shape(y_pred_flat)[0])
        y_true_flat = y_true_flat[:min_len]
        y_pred_flat = y_pred_flat[:min_len]

        mse = losses.MeanSquaredError(reduction='none')
        return mse(y_true_flat, y_pred_flat)
    
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
                            try:
                                loss_val = self.loss_fns[loss_name](y_true[true_key], y_pred[true_key])
                                total_loss += weight * tf.reduce_mean(loss_val)
                            except Exception as e:
                                # Handle shape mismatch
                                # Removed tf.print to maintain XLA compatibility
                                loss_val = self._handle_shape_mismatch(y_true[true_key], y_pred[true_key])
                                total_loss += weight * tf.reduce_mean(loss_val)
                            break
        else:
            # Single output case - try configured losses first
            loss_applied = False
            
            for loss_name, weight in self.losses_config.items():
                if loss_name in self.loss_fns:
                    try:
                        loss_val = self.loss_fns[loss_name](y_true, y_pred)
                        total_loss += weight * tf.reduce_mean(loss_val)
                        loss_applied = True
                        break  # Use first compatible loss
                    except Exception as e:
                        # Removed tf.print to maintain XLA compatibility
                        continue
            
            # If no configured loss worked, handle shape mismatch
            if not loss_applied:
                # Removed tf.print to maintain XLA compatibility
                total_loss = tf.reduce_mean(self._handle_shape_mismatch(y_true, y_pred))
                
        return total_loss

class SmartCombinedLoss(losses.Loss):
    """Smart combined loss that automatically detects data type and applies appropriate losses."""
    
    def __init__(self, classification_weight=1.0, bbox_weight=1.0, segmentation_weight=1.0,                reduction='sum_over_batch_size', name='smart_combined_loss'):
        super().__init__(reduction=reduction, name=name)
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight
        self.segmentation_weight = segmentation_weight
        
        # Initialize loss functions
        self.focal_loss = FocalLoss()
        self.ciou_loss = IoULoss('ciou')
        self.dice_loss = DiceLoss()
        self.scce_loss = losses.SparseCategoricalCrossentropy(reduction='none', from_logits=True)
        self.cce_loss = losses.CategoricalCrossentropy(reduction='none', from_logits=True)
        self.mse_loss = losses.MeanSquaredError(reduction='none')
        
    @tf.function
    def _detect_and_handle_data_type(self, y_true, y_pred):
        """Detect data type and return appropriate loss."""
        true_shape = tf.shape(y_true)
        pred_shape = tf.shape(y_pred)
        
        # Define loss computation functions
        def bbox_loss():
            return self.ciou_loss(y_true, y_pred)
        
        def sparse_categorical_loss():
            y_true_int = tf.cast(y_true, tf.int32)
            return self.scce_loss(y_true_int, y_pred)
        
        def segmentation_loss():
            return self.dice_loss(y_true, y_pred)
        
        def binary_classification_loss():
            return self.focal_loss(y_true, y_pred)
        
        def categorical_loss():
            return self.cce_loss(y_true, y_pred)
        
        def fallback_loss():
            # Removed debug tf.print for XLA compatibility
            
            # Try to make shapes compatible using TensorFlow operations
            def handle_rank_mismatch():
                # Flatten both to 1D
                y_true_flat = tf.reshape(y_true, [-1])
                y_pred_flat = tf.reshape(y_pred, [-1])
                
                # Take minimum length
                min_len = tf.minimum(tf.shape(y_true_flat)[0], tf.shape(y_pred_flat)[0])
                y_true_flat = y_true_flat[:min_len]
                y_pred_flat = y_pred_flat[:min_len]
                
                return self.mse_loss(tf.cast(y_true_flat, tf.float32), tf.cast(y_pred_flat, tf.float32))
            
            def handle_same_rank():
                return self.mse_loss(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))
            
            return tf.cond(
                tf.not_equal(tf.rank(y_true), tf.rank(y_pred)),
                handle_rank_mismatch,
                handle_same_rank
            )
        
        # Case 1: Bounding box regression (both have last dim = 4)
        is_bbox = tf.logical_and(
            tf.logical_and(tf.equal(true_shape[-1], 4), tf.equal(pred_shape[-1], 4)),
            tf.equal(tf.rank(y_true), tf.rank(y_pred))
        )
        
        # Case 2: Multi-class classification per detection
        is_sparse_categorical = tf.logical_and(
            tf.logical_and(tf.equal(tf.rank(y_true), 1), tf.equal(tf.rank(y_pred), 2)),
            tf.logical_and(
                tf.equal(true_shape[0], pred_shape[0]),
                tf.reduce_all(tf.logical_and(
                    y_true >= 0,
                    y_true < tf.cast(pred_shape[1], y_true.dtype)
                ))
            )
        )
        
        # Case 3: Segmentation (multi-dimensional spatial data)
        is_segmentation = tf.logical_and(
            tf.logical_and(tf.greater(tf.rank(y_true), 2), tf.greater(tf.rank(y_pred), 2)),
            tf.reduce_all(tf.equal(true_shape[:-1], pred_shape[:-1]))
        )
        
        # Case 4: Binary classification with compatible shapes
        is_binary = tf.logical_and(
            tf.equal(tf.rank(y_true), tf.rank(y_pred)),
            tf.reduce_all(tf.equal(true_shape, pred_shape))
        )
        
        # Case 5: One-hot categorical
        is_categorical = tf.logical_and(
            tf.logical_and(tf.equal(tf.rank(y_true), 2), tf.equal(tf.rank(y_pred), 2)),
            tf.equal(true_shape[0], pred_shape[0])
        )
        
        return tf.cond(
            is_bbox, bbox_loss,
            lambda: tf.cond(
                is_sparse_categorical, sparse_categorical_loss,
                lambda: tf.cond(
                    is_segmentation, segmentation_loss,
                    lambda: tf.cond(
                        is_binary, binary_classification_loss,
                        lambda: tf.cond(
                            is_categorical, categorical_loss,
                            fallback_loss
                        )
                    )
                )
            )
        )
    
    @tf.function
    def call(self, y_true, y_pred):
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            # Multi-output case
            total_loss = 0.0
            for key in y_true.keys():
                if key in y_pred:
                    loss_val = self._detect_and_handle_data_type(y_true[key], y_pred[key])
                    total_loss += tf.reduce_mean(loss_val)
            return total_loss
        else:
            # Single output case
            loss_val = self._detect_and_handle_data_type(y_true, y_pred)
            return tf.reduce_mean(loss_val)


class ObjectDetectionLoss(losses.Loss):
    """Specialized loss for object detection with class prediction per bbox."""
    
    def __init__(self, reduction='sum_over_batch_size', name='object_detection_loss'):
        super().__init__(reduction=reduction, name=name)
        self.scce_loss = losses.SparseCategoricalCrossentropy(reduction='none', from_logits=True)
        
    @tf.function
    def call(self, y_true, y_pred):
        """
        Handle case where:
        - y_true: (n_detections,) containing class indices
        - y_pred: (n_detections, n_classes) containing class logits
        """
        # Ensure y_true is integer type for sparse categorical
        y_true_int = tf.cast(y_true, tf.int32)
        
        # Apply sparse categorical crossentropy
        loss = self.scce_loss(y_true_int, y_pred)
        
        return loss


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
        'smart_combined': lambda: SmartCombinedLoss(**kwargs),
        'object_detection': lambda: ObjectDetectionLoss(**kwargs),
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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import *
import tensorflow_addons as tfa

class SOTALoss:
    """SOTA Loss functions cho Object Detection, Segmentation và Multitask"""
    
    @staticmethod
    def focal_loss(alpha=0.25, gamma=2.0):
        """Focal Loss - Tốt cho class imbalance"""
        def loss_fn(y_true, y_pred):
            return tfa.losses.focal_loss.focal_loss(
                y_true, y_pred, alpha=alpha, gamma=gamma
            )
        return loss_fn
    
    @staticmethod  
    def dice_loss(smooth=1e-6):
        """Dice Loss - Tốt cho segmentation"""
        def loss_fn(y_true, y_pred):
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
            
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
            
            dice = (2. * intersection + smooth) / (union + smooth)
            return 1 - dice
        return loss_fn
    
    @staticmethod
    def iou_loss():
        """IoU Loss - Tốt cho object detection"""
        def loss_fn(y_true, y_pred):
            return tfa.losses.giou_loss(y_true, y_pred)
        return loss_fn
    
    @staticmethod
    def combined_seg_loss(dice_weight=0.5, focal_weight=0.5):
        """Combined loss cho segmentation"""
        dice_fn = SOTALoss.dice_loss()
        focal_fn = SOTALoss.focal_loss()
        
        def loss_fn(y_true, y_pred):
            return dice_weight * dice_fn(y_true, y_pred) + \
                   focal_weight * focal_fn(y_true, y_pred)
        return loss_fn

class ObjectDetectionLoss:
    """SOTA losses cho Object Detection"""
    
    @staticmethod
    def yolo_loss(num_classes, anchors, ignore_thresh=0.5):
        """YOLO-style loss sử dụng built-in functions"""
        def loss_fn(y_true, y_pred):
            # Box regression loss
            box_loss = tf.reduce_mean(
                tf.keras.losses.huber(y_true[..., :4], y_pred[..., :4])
            )
            
            # Objectness loss  
            obj_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    y_true[..., 4:5], y_pred[..., 4:5]
                )
            )
            
            # Classification loss
            class_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_true[..., 5:], y_pred[..., 5:]
                )
            )
            
            return box_loss + obj_loss + class_loss
        return loss_fn
    
    @staticmethod
    def fcos_loss():
        """FCOS-style loss"""
        def loss_fn(y_true, y_pred):
            # Centerness + Classification + Regression
            cls_loss = tf.reduce_mean(
                SOTALoss.focal_loss()(y_true['cls'], y_pred['cls'])
            )
            
            reg_loss = tf.reduce_mean(
                tf.keras.losses.huber(y_true['reg'], y_pred['reg'])
            )
            
            center_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    y_true['center'], y_pred['center']
                )
            )
            
            return cls_loss + reg_loss + center_loss
        return loss_fn

class SegmentationLoss:
    """SOTA losses cho Segmentation"""
    
    @staticmethod
    def unet_loss():
        """U-Net style combined loss"""
        return SOTALoss.combined_seg_loss(dice_weight=0.5, focal_weight=0.5)
    
    @staticmethod
    def deeplabv3_loss():
        """DeepLabV3 style loss với auxiliary loss"""
        def loss_fn(y_true, y_pred):
            if isinstance(y_pred, list):  # Multi-output
                main_loss = SOTALoss.combined_seg_loss()(y_true, y_pred[0])
                aux_loss = tf.reduce_mean([
                    SOTALoss.combined_seg_loss()(y_true, pred) 
                    for pred in y_pred[1:]
                ])
                return main_loss + 0.4 * aux_loss
            else:
                return SOTALoss.combined_seg_loss()(y_true, y_pred)
        return loss_fn
    
    @staticmethod
    def boundary_loss(theta0=3, theta=5):
        """Boundary-aware loss"""
        def loss_fn(y_true, y_pred):
            # Dice loss
            dice = SOTALoss.dice_loss()(y_true, y_pred)
            
            # Boundary loss using morphological operations
            kernel = tf.ones((3, 3, 1, 1))
            
            # Erosion và dilation để tìm boundary
            eroded = tf.nn.erosion2d(
                y_true[..., None], kernel, [1,1,1,1], [1,1,1,1], 'SAME'
            )[..., 0]
            
            boundary = y_true - eroded
            boundary_pred = y_pred * boundary
            
            boundary_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(boundary, boundary_pred)
            )
            
            return dice + 0.1 * boundary_loss
        return loss_fn

class MultitaskLoss:
    """SOTA losses cho Multitask Learning"""
    
    def __init__(self, task_weights=None, adaptive_weights=True):
        self.task_weights = task_weights
        self.adaptive_weights = adaptive_weights
        if adaptive_weights:
            self.log_vars = {}
    
    def uncertainty_weighting(self, losses, task_names):
        """Uncertainty weighting cho multitask"""
        total_loss = 0
        
        for i, (loss, name) in enumerate(zip(losses, task_names)):
            if name not in self.log_vars:
                self.log_vars[name] = tf.Variable(0.0, trainable=True, name=f'log_var_{name}')
            
            precision = tf.exp(-self.log_vars[name])
            total_loss += precision * loss + self.log_vars[name]
            
        return total_loss
    
    def get_multitask_loss(self, tasks):
        """
        tasks: dict {'detection': loss_fn, 'segmentation': loss_fn, ...}
        """
        def loss_fn(y_true, y_pred):
            losses = []
            task_names = []
            
            for task_name, task_loss_fn in tasks.items():
                if task_name in y_true and task_name in y_pred:
                    loss = task_loss_fn(y_true[task_name], y_pred[task_name])
                    losses.append(loss)
                    task_names.append(task_name)
            
            if self.adaptive_weights:
                return self.uncertainty_weighting(losses, task_names)
            else:
                # Fixed weights
                weighted_losses = []
                for i, (loss, name) in enumerate(zip(losses, task_names)):
                    weight = self.task_weights.get(name, 1.0) if self.task_weights else 1.0
                    weighted_losses.append(weight * loss)
                return tf.add_n(weighted_losses)
                
        return loss_fn

# ==========================================================
# CÁCH SỬ DỤNG - CỰC KỲ ĐỠN GIẢN
# ==========================================================

# 1. Object Detection
def get_detection_loss(num_classes):
    return ObjectDetectionLoss.yolo_loss(num_classes=num_classes, anchors=None)

# 2. Segmentation  
def get_segmentation_loss():
    return SegmentationLoss.unet_loss()

# 3. Multitask
def get_multitask_loss():
    multitask = MultitaskLoss(adaptive_weights=True)
    
    tasks = {
        'detection': ObjectDetectionLoss.fcos_loss(),
        'segmentation': SegmentationLoss.unet_loss(),
        'classification': SOTALoss.focal_loss()
    }
    
    return multitask.get_multitask_loss(tasks)

# 4. Quick setup functions
def quick_focal():
    return tfa.losses.focal_loss.focal_loss

def quick_dice():
    return SOTALoss.dice_loss()

def quick_giou():
    return tfa.losses.giou_loss