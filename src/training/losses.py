"""
Optimized custom loss functions with SOTA techniques for object detection and semantic segmentation
"""
import tensorflow as tf
from typing import Dict, Any
import numpy as np
from tensorflow.keras import backend as K
"""
Optimized custom loss functions with SOTA techniques for object detection and semantic segmentation
Fixed bugs and improved performance for pet detection/segmentation dataset
"""
import tensorflow as tf
from typing import Dict, Any, Optional, Union
import numpy as np
from tensorflow.keras import backend as K


class AdaptiveSmoothL1Loss(tf.keras.losses.Loss):
    """Adaptive Smooth L1 loss with learnable beta parameter"""
    
    def __init__(
        self,
        beta: float = 1.0,
        adaptive: bool = True,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'adaptive_smooth_l1_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        self.adaptive = adaptive
        if adaptive:
            self.beta_var = tf.Variable(
                initial_value=tf.constant(beta, dtype=tf.float32), 
                trainable=True, 
                name='beta'
            )
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        diff = tf.abs(y_true - y_pred)
        beta = self.beta_var if self.adaptive else tf.constant(self.beta, dtype=tf.float32)
        
        # Adaptive beta clipping with proper bounds
        beta = tf.clip_by_value(beta, 0.01, 5.0)
        
        loss = tf.where(
            diff < beta,
            0.5 * tf.square(diff) / beta,
            diff - 0.5 * beta
        )
        return tf.reduce_mean(loss)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'beta': self.beta, 'adaptive': self.adaptive})
        return config


class EIoULoss(tf.keras.losses.Loss):
    """Efficient IoU Loss - SOTA improvement over GIoU"""
    
    def __init__(
        self,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'eiou_loss'
    ):
        super().__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """EIoU Loss implementation with improved numerical stability"""
        eps = 1e-7
        
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle different input formats (normalize coordinates if needed)
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        # Calculate intersection
        x1_max = tf.maximum(y_true[..., 0], y_pred[..., 0])
        y1_max = tf.maximum(y_true[..., 1], y_pred[..., 1])
        x2_min = tf.minimum(y_true[..., 2], y_pred[..., 2])
        y2_min = tf.minimum(y_true[..., 3], y_pred[..., 3])
        
        intersection_area = tf.maximum(0.0, x2_min - x1_max) * tf.maximum(0.0, y2_min - y1_max)
        
        # Calculate areas with safety checks
        true_w = tf.maximum(eps, y_true[..., 2] - y_true[..., 0])
        true_h = tf.maximum(eps, y_true[..., 3] - y_true[..., 1])
        pred_w = tf.maximum(eps, y_pred[..., 2] - y_pred[..., 0])
        pred_h = tf.maximum(eps, y_pred[..., 3] - y_pred[..., 1])
        
        true_area = true_w * true_h
        pred_area = pred_w * pred_h
        union_area = true_area + pred_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / (union_area + eps)
        
        # Calculate enclosing box
        x1_min = tf.minimum(y_true[..., 0], y_pred[..., 0])
        y1_min = tf.minimum(y_true[..., 1], y_pred[..., 1])
        x2_max = tf.maximum(y_true[..., 2], y_pred[..., 2])
        y2_max = tf.maximum(y_true[..., 3], y_pred[..., 3])
        
        enclosing_w = tf.maximum(eps, x2_max - x1_min)
        enclosing_h = tf.maximum(eps, y2_max - y1_min)
        enclosing_area = enclosing_w * enclosing_h
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + eps)
        
        # Calculate aspect ratio penalty (EIoU improvement)
        aspect_penalty = (4 / (np.pi ** 2)) * tf.square(
            tf.atan(true_w / (true_h + eps)) - tf.atan(pred_w / (pred_h + eps))
        )
        
        # Calculate focal penalty with improved stability
        focal_penalty = aspect_penalty * tf.square(1 - iou) / (1 - iou + aspect_penalty + eps)
        
        # EIoU = GIoU - focal_penalty
        eiou = giou - focal_penalty
        
        # Return loss with proper handling of edge cases
        loss = 1.0 - eiou
        return tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))


class CIoULoss(tf.keras.losses.Loss):
    """Complete IoU Loss with improved numerical stability"""
    
    def __init__(self, 
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, 
                 name='ciou_loss'):
        super().__init__(reduction, name)

    @tf.function
    def call(self, y_true, y_pred):
        eps = 1e-7
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip coordinates to valid range
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        # Convert (x1,y1,x2,y2) → (cx,cy,w,h)
        true_center = (y_true[..., :2] + y_true[..., 2:]) * 0.5
        pred_center = (y_pred[..., :2] + y_pred[..., 2:]) * 0.5
        
        true_wh = tf.maximum(eps, y_true[..., 2:] - y_true[..., :2])
        pred_wh = tf.maximum(eps, y_pred[..., 2:] - y_pred[..., :2])
        
        # Calculate intersection
        inter_wh = tf.maximum(0.0,
            tf.minimum(y_true[..., 2:], y_pred[..., 2:]) -
            tf.maximum(y_true[..., :2], y_pred[..., :2])
        )
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        
        # Calculate areas
        area_true = true_wh[..., 0] * true_wh[..., 1]
        area_pred = pred_wh[..., 0] * pred_wh[..., 1]
        union = area_true + area_pred - inter_area
        
        # IoU
        iou = inter_area / (union + eps)

        # Center distance penalty
        center_dist = tf.reduce_sum(tf.square(true_center - pred_center), -1)
        enclose_wh = tf.maximum(y_true[..., 2:], y_pred[..., 2:]) - \
                     tf.minimum(y_true[..., :2], y_pred[..., :2])
        c2 = tf.reduce_sum(tf.square(enclose_wh), -1) + eps

        # Aspect-ratio penalty with improved stability
        v = 4 / (np.pi**2) * tf.square(
            tf.atan(true_wh[..., 0]/(true_wh[..., 1] + eps)) -
            tf.atan(pred_wh[..., 0]/(pred_wh[..., 1] + eps))
        )
        
        # Improved alpha calculation
        alpha = v / (1 - iou + v + eps)
        alpha = tf.stop_gradient(alpha)  # Stop gradient for stability

        ciou = iou - (center_dist / c2 + alpha * v)
        loss = 1 - ciou
        
        return tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))


class PolyLoss(tf.keras.losses.Loss):
    """Polynomial Loss - SOTA for classification tasks with improved numerical stability"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 1.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'poly_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon
        self.alpha = alpha
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross entropy
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        
        # Calculate pt (probability of true class)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        pt = tf.clip_by_value(pt, 1e-7, 1.0)
        
        # Calculate Poly loss
        poly_loss = ce + self.alpha * tf.pow(1 - pt, self.epsilon + 1)
        
        return poly_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing
        })
        return config


class VarifocalLoss(tf.keras.losses.Loss):
    """Varifocal Loss - SOTA for dense object detection with improvements"""
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'varifocal_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate focal weight with improved numerical stability
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_weight = tf.pow(alpha_t, self.gamma)
        
        # Calculate varifocal loss with proper weighting
        ce_loss = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        vf_loss = focal_weight * y_true * ce_loss
        
        return tf.reduce_mean(vf_loss)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class DiceLoss(tf.keras.losses.Loss):
    """Improved Dice loss for semantic segmentation"""
    
    def __init__(self,
        smooth: float = 1.0,
        squared_denom: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'dice_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        self.squared_denom = squared_denom
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle different input shapes
        if len(y_true.shape) > 2:
            # Flatten spatial dimensions but keep batch and channel dims
            axes = list(range(1, len(y_true.shape) - 1))  # Spatial dimensions
            intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
            true_sum = tf.reduce_sum(y_true, axis=axes)
            pred_sum = tf.reduce_sum(y_pred, axis=axes)
        else:
            intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
            true_sum = tf.reduce_sum(y_true, axis=-1)
            pred_sum = tf.reduce_sum(y_pred, axis=-1)
        
        # Calculate Dice coefficient
        if self.squared_denom:
            denom = true_sum + pred_sum
        else:
            denom = tf.square(true_sum) + tf.square(pred_sum)
            
        dice_coef = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        
        # Handle per-class dice for multi-class segmentation
        if len(dice_coef.shape) > 1:
            dice_coef = tf.reduce_mean(dice_coef, axis=-1)
        
        return 1.0 - dice_coef
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'squared_denom': self.squared_denom
        })
        return config


class AdaptiveDiceLoss(tf.keras.losses.Loss):
    """Adaptive Dice Loss with learnable smooth parameter and class weights"""
    
    def __init__(
        self,
        smooth: float = 1.0,
        adaptive: bool = True,
        power: float = 1.0,
        class_weights: Optional[tf.Tensor] = None,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'adaptive_dice_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        self.adaptive = adaptive
        self.power = power
        self.class_weights = class_weights
        
        if adaptive:
            self.smooth_var = tf.Variable(
                initial_value=tf.constant(smooth, dtype=tf.float32), 
                trainable=True, 
                name='smooth'
            )
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Handle different input shapes (3D for segmentation)
        if len(y_true.shape) > 2:
            axes = list(range(1, len(y_true.shape) - 1))
            intersection = tf.reduce_sum(tf.pow(y_true * y_pred, self.power), axis=axes)
            true_sum = tf.reduce_sum(tf.pow(y_true, self.power), axis=axes)
            pred_sum = tf.reduce_sum(tf.pow(y_pred, self.power), axis=axes)
        else:
            intersection = tf.reduce_sum(tf.pow(y_true * y_pred, self.power), axis=-1)
            true_sum = tf.reduce_sum(tf.pow(y_true, self.power), axis=-1)
            pred_sum = tf.reduce_sum(tf.pow(y_pred, self.power), axis=-1)
        
        # Use adaptive or fixed smooth
        smooth = self.smooth_var if self.adaptive else tf.constant(self.smooth, dtype=tf.float32)
        smooth = tf.clip_by_value(smooth, 0.01, 10.0)
        
        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + smooth) / (true_sum + pred_sum + smooth)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            dice_coef = dice_coef * tf.cast(self.class_weights, tf.float32)
        
        # Handle per-class dice
        if len(dice_coef.shape) > 1:
            dice_coef = tf.reduce_mean(dice_coef, axis=-1)
        
        return 1.0 - dice_coef
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'adaptive': self.adaptive,
            'power': self.power
        })
        return config


class BoundaryLoss(tf.keras.losses.Loss):
    """Improved Boundary Loss for better edge preservation in segmentation"""
    
    def __init__(
        self,
        theta0: float = 3.0,
        theta: float = 5.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'boundary_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.theta0 = theta0
        self.theta = theta
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        def get_boundaries(mask):
            # Improved gradient calculation for boundary detection
            if len(mask.shape) == 3:
                mask = tf.expand_dims(mask, -1)
            elif len(mask.shape) == 4 and mask.shape[-1] > 1:
                # For multi-class, take argmax
                mask = tf.expand_dims(tf.cast(tf.argmax(mask, axis=-1), tf.float32), -1)
            
            # Sobel operators for edge detection
            sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
            sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
            
            sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
            
            grad_x = tf.nn.conv2d(mask, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            grad_y = tf.nn.conv2d(mask, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            
            boundaries = tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-12)
            return tf.squeeze(boundaries, -1)
        
        # Get boundaries
        boundaries_true = get_boundaries(y_true)
        boundaries_pred = get_boundaries(y_pred)
        
        # Calculate boundary loss with L1 distance
        boundary_diff = tf.abs(boundaries_true - boundaries_pred)
        boundary_loss = tf.reduce_mean(boundary_diff)
        
        return boundary_loss


class UnifiedDetectionLoss(tf.keras.losses.Loss):
    """Unified Detection Loss combining multiple SOTA techniques with proper error handling"""
    
    def __init__(
        self,
        bbox_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        quality_loss_weight: float = 1.0,
        use_eiou: bool = True,
        use_varifocal: bool = True,
        use_quality_focal: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'unified_detection_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.quality_loss_weight = quality_loss_weight
        self.use_eiou = use_eiou
        self.use_varifocal = use_varifocal
        self.use_quality_focal = use_quality_focal
        
        # Initialize losses
        self.bbox_loss = EIoULoss() if use_eiou else CIoULoss()
        self.cls_loss = VarifocalLoss() if use_varifocal else PolyLoss()
        if use_quality_focal:
            from .quality_focal_loss import QualityFocalLoss  # Import if available
            self.quality_loss = QualityFocalLoss()
        
    def call(self, y_true: Union[Dict[str, tf.Tensor], tf.Tensor], 
             y_pred: Union[Dict[str, tf.Tensor], tf.Tensor]) -> tf.Tensor:
        
        # Handle both dictionary and tensor inputs
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            return self._call_dict_inputs(y_true, y_pred)
        else:
            return self._call_tensor_inputs(y_true, y_pred)
    
    def _call_dict_inputs(self, y_true: Dict[str, tf.Tensor], 
                         y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Cast all inputs to float32
        for key in y_true:
            y_true[key] = tf.cast(y_true[key], tf.float32)
        for key in y_pred:
            y_pred[key] = tf.cast(y_pred[key], tf.float32)
        
        total_loss = 0.0
        
        # Handle bounding box loss - support multiple key naming conventions
        bbox_keys = ['head_bbox', 'bbox', 'bbox_output', 'bounding_box']
        bbox_true = None
        bbox_pred = None
        
        for key in bbox_keys:
            if key in y_true and key in y_pred:
                bbox_true = y_true[key]
                bbox_pred = y_pred[key]
                break
        
        if bbox_true is not None and bbox_pred is not None:
            bbox_loss = self.bbox_loss(bbox_true, bbox_pred)
            total_loss += self.bbox_loss_weight * bbox_loss
        
        # Handle classification loss - support multiple key naming conventions
        cls_keys = ['label', 'class', 'cls', 'class_output', 'species']
        cls_true = None
        cls_pred = None
        
        for key in cls_keys:
            if key in y_true and key in y_pred:
                cls_true = y_true[key]
                cls_pred = y_pred[key]
                break
        
        if cls_true is not None and cls_pred is not None:
            # Convert integer labels to one-hot if needed
            if len(cls_true.shape) == 1 or (len(cls_true.shape) == 2 and cls_true.shape[-1] == 1):
                num_classes = tf.reduce_max(cls_true) + 1
                cls_true = tf.one_hot(tf.cast(cls_true, tf.int32), depth=num_classes)
                cls_true = tf.squeeze(cls_true)
            
            cls_loss = self.cls_loss(cls_true, cls_pred)
            total_loss += self.cls_loss_weight * cls_loss
        
        # Add quality loss if available and enabled
        if (self.use_quality_focal and 
            'quality' in y_true and 'quality' in y_pred and 
            hasattr(self, 'quality_loss')):
            quality_loss = self.quality_loss(y_true['quality'], y_pred['quality'])
            total_loss += self.quality_loss_weight * quality_loss
        
        return total_loss
    
    def _call_tensor_inputs(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Handle tensor inputs by assuming they are bbox coordinates"""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self.bbox_loss(y_true, y_pred)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'bbox_loss_weight': self.bbox_loss_weight,
            'cls_loss_weight': self.cls_loss_weight,
            'quality_loss_weight': self.quality_loss_weight,
            'use_eiou': self.use_eiou,
            'use_varifocal': self.use_varifocal,
            'use_quality_focal': self.use_quality_focal
        })
        return config


class UnifiedSegmentationLoss(tf.keras.losses.Loss):
    """Unified Segmentation Loss combining multiple SOTA techniques"""
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        hausdorff_weight: float = 0.3,
        use_adaptive_dice: bool = True,
        use_boundary_loss: bool = True,
        use_hausdorff_loss: bool = True,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = 'unified_segmentation_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.hausdorff_weight = hausdorff_weight
        self.use_adaptive_dice = use_adaptive_dice
        self.use_boundary_loss = use_boundary_loss
        self.use_hausdorff_loss = use_hausdorff_loss
        
        # Initialize losses
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy()
        self.dice_loss = AdaptiveDiceLoss() if use_adaptive_dice else DiceLoss()
        if use_boundary_loss:
            self.boundary_loss = BoundaryLoss()
        if use_hausdorff_loss:
            self.hausdorff_loss = HausdorffLoss()
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate base losses
        ce_loss = self.ce_loss(y_true, y_pred)
        dice_loss = self.dice_loss(y_true, y_pred)
        
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        # Add boundary loss
        if self.use_boundary_loss:
            boundary_loss = self.boundary_loss(y_true, y_pred)
            total_loss += self.boundary_weight * boundary_loss
        
        # Add Hausdorff loss
        if self.use_hausdorff_loss:
            hausdorff_loss = self.hausdorff_loss(y_true, y_pred)
            total_loss += self.hausdorff_weight * hausdorff_loss
        
        return total_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'ce_weight': self.ce_weight,
            'dice_weight': self.dice_weight,
            'boundary_weight': self.boundary_weight,
            'hausdorff_weight': self.hausdorff_weight,
            'use_adaptive_dice': self.use_adaptive_dice,
            'use_boundary_loss': self.use_boundary_loss,
            'use_hausdorff_loss': self.use_hausdorff_loss
        })
        return config



# Updated factory function with SOTA losses
def get_sota_loss_function(
    loss_config: Dict[str, Any]
) -> tf.keras.losses.Loss:
    """Factory function to get SOTA loss function based on configuration"""
    loss_type = loss_config.get('loss_type', 'mse').lower()
    
    if loss_type == 'adaptive_smooth_l1':
        return AdaptiveSmoothL1Loss(
            beta=loss_config.get('beta', 1.0),
            adaptive=loss_config.get('adaptive', True)
        )
    elif loss_type == 'eiou':
        return EIoULoss()
    elif loss_type == 'poly':
        return PolyLoss(
            epsilon=loss_config.get('epsilon', 1.0),
            alpha=loss_config.get('alpha', 1.0)
        )
    elif loss_type == 'varifocal':
        return VarifocalLoss(
            alpha=loss_config.get('alpha', 0.75),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'adaptive_dice':
        return AdaptiveDiceLoss(
            smooth=loss_config.get('smooth', 1.0),
            adaptive=loss_config.get('adaptive', True)
        )
    elif loss_type == 'combo':
        return ComboLoss(
            alpha=loss_config.get('alpha', 0.5),
            beta=loss_config.get('beta', 0.5)
        )
    elif loss_type == 'quality_focal':
        return QualityFocalLoss(
            beta=loss_config.get('beta', 2.0)
        )
    elif loss_type == 'unified_detection':
        return UnifiedDetectionLoss(
            bbox_loss_weight=loss_config.get('bbox_loss_weight', 2.0),
            cls_loss_weight=loss_config.get('cls_loss_weight', 1.0),
            quality_loss_weight=loss_config.get('quality_loss_weight', 1.0)
        )
    elif loss_type == 'unified_segmentation':
        return UnifiedSegmentationLoss(
            ce_weight=loss_config.get('ce_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            boundary_weight=loss_config.get('boundary_weight', 0.5)
        )
    elif loss_type == 'boundary':
        return BoundaryLoss()
    elif loss_type == 'hausdorff':
        return HausdorffLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage configurations
SOTA_LOSS_CONFIGS = {
    'detection_sota': {
        'type': 'unified_detection',
        'bbox_loss_weight': 2.0,
        'cls_loss_weight': 1.0,
        'quality_loss_weight': 1.0,
        'use_eiou': True,
        'use_varifocal': True,
        'use_quality_focal': True
    },
    'segmentation_sota': {
        'type': 'unified_segmentation',
        'ce_weight': 1.0,
        'dice_weight': 1.0,
        'boundary_weight': 0.5,
        'hausdorff_weight': 0.3,
        'use_adaptive_dice': True,
        'use_boundary_loss': True,
        'use_hausdorff_loss': True
    },
    'classification_sota': {
        'type': 'poly',
        'epsilon': 1.0,
        'alpha': 1.0,
        'from_logits': False
    }
}