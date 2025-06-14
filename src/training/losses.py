"""
Optimized custom loss functions with SOTA techniques for object detection and semantic segmentation
"""
import tensorflow as tf
from typing import Dict, Any
import numpy as np
from tensorflow.keras import backend as K


class AdaptiveSmoothL1Loss(tf.keras.losses.Loss):
    """Adaptive Smooth L1 loss with learnable beta parameter"""
    
    def __init__(
        self,
        beta: float = 1.0,
        adaptive: bool = True,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'adaptive_smooth_l1_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        self.adaptive = adaptive
        if adaptive:
            self.beta_var = tf.Variable(initial_value=beta, trainable=True, name='beta')
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        diff = tf.abs(y_true - y_pred)
        beta = self.beta_var if self.adaptive else self.beta
        
        # Adaptive beta clipping
        beta = tf.clip_by_value(beta, 0.1, 2.0)
        
        loss = tf.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
        return tf.reduce_mean(loss, axis=-1)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'beta': self.beta, 'adaptive': self.adaptive})
        return config


class EIoULoss(tf.keras.losses.Loss):
    """Efficient IoU Loss - SOTA improvement over GIoU"""
    
    def __init__(
        self,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'eiou_loss'
    ):
        super().__init__(reduction=reduction, name=name)

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """EIoU Loss implementation"""
        eps = 1e-7
        
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
        iou = intersection_area / (union_area + eps)
        
        # Calculate enclosing box
        x1_min = tf.minimum(y_true[..., 0], y_pred[..., 0])
        y1_min = tf.minimum(y_true[..., 1], y_pred[..., 1])
        x2_max = tf.maximum(y_true[..., 2], y_pred[..., 2])
        y2_max = tf.maximum(y_true[..., 3], y_pred[..., 3])
        
        enclosing_area = (x2_max - x1_min) * (y2_max - y1_min)
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + eps)
        
        # Calculate aspect ratio penalty (EIoU improvement)
        true_w = y_true[..., 2] - y_true[..., 0]
        true_h = y_true[..., 3] - y_true[..., 1]
        pred_w = y_pred[..., 2] - y_pred[..., 0]
        pred_h = y_pred[..., 3] - y_pred[..., 1]
        
        enclosing_w = x2_max - x1_min
        enclosing_h = y2_max - y1_min
        
        aspect_penalty = (4 / (np.pi ** 2)) * tf.square(
            tf.atan(true_w / (true_h + eps)) - tf.atan(pred_w / (pred_h + eps))
        )
        
        # Calculate focal penalty
        focal_penalty = aspect_penalty * tf.square(1 - iou) / (1 - iou + aspect_penalty + eps)
        
        # EIoU = GIoU - focal_penalty
        eiou = giou - focal_penalty
        
        return 1.0 - eiou
    

class CIoULoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name='ciou_loss'):
        super().__init__(reduction, name)

    @tf.function
    def call(self, y_true, y_pred):
        # (x1,y1,x2,y2) → (cx,cy,w,h)
        true = tf.concat([(y_true[..., :2] + y_true[..., 2:]) * 0.5,
                          y_true[..., 2:] - y_true[..., :2]], axis=-1)
        pred = tf.concat([(y_pred[..., :2] + y_pred[..., 2:]) * 0.5,
                          y_pred[..., 2:] - y_pred[..., :2]], axis=-1)
        # IoU
        inter_wh = tf.maximum(
            tf.minimum(y_true[..., 2:], y_pred[..., 2:]) -
            tf.maximum(y_true[..., :2], y_pred[..., :2]), 0.)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        area_true = (y_true[..., 2]-y_true[..., 0]) * (y_true[..., 3]-y_true[..., 1])
        area_pred = (y_pred[..., 2]-y_pred[..., 0]) * (y_pred[..., 3]-y_pred[..., 1])
        union = area_true + area_pred - inter_area
        iou = inter_area / (union + K.epsilon())

        # center distance penalty
        center_dist = tf.reduce_sum(tf.square(true[..., :2] - pred[..., :2]), -1)
        enclose_wh = tf.maximum(y_true[..., 2:], y_pred[..., 2:]) - \
                     tf.minimum(y_true[..., :2], y_pred[..., :2])
        c2 = tf.reduce_sum(tf.square(enclose_wh), -1) + K.epsilon()

        # aspect-ratio penalty
        v = 4 / (3.14159265**2) * tf.square(
            tf.atan(true[..., 2]/(true[..., 3]+K.epsilon())) -
            tf.atan(pred[..., 2]/(pred[..., 3]+K.epsilon())))
        with tf.GradientTape() as tape:
            S = tf.stop_gradient(1 - iou)
        alpha = v / (S + v + K.epsilon())

        ciou = iou - (center_dist / c2 + alpha * v)
        return 1 - ciou


class PolyLoss(tf.keras.losses.Loss):
    """Polynomial Loss - SOTA for classification tasks"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 1.0,
        from_logits: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'poly_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.epsilon = epsilon
        self.alpha = alpha
        self.from_logits = from_logits
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Calculate cross entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # Calculate Poly loss
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        poly_loss = ce + self.alpha * tf.pow(1 - pt, self.epsilon + 1)
        
        return poly_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config


class VarifocalLoss(tf.keras.losses.Loss):
    """Varifocal Loss - SOTA for dense object detection"""
    
    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
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
        
        # Calculate focal weight
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        focal_weight = tf.pow(self.alpha * y_true + (1 - self.alpha) * (1 - y_true), self.gamma)
        
        # Calculate varifocal loss
        loss = -focal_weight * y_true * tf.math.log(p_t + 1e-8)
        
        return tf.reduce_mean(loss)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class HausdorffLoss(tf.keras.losses.Loss):
    """Hausdorff Distance Loss for segmentation - better boundary preservation"""
    
    def __init__(
        self,
        alpha: float = 2.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'hausdorff_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate distance transforms (approximated)
        def distance_transform(mask):
            # Simple approximation using morphological operations
            kernel = tf.ones((3, 3, 1, 1))
            eroded = tf.nn.erosion2d(mask[..., None], kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')
            distance = mask[..., None] - eroded
            return tf.squeeze(distance, axis=-1)
        
        # Calculate Hausdorff distance components
        dt_true = distance_transform(y_true)
        dt_pred = distance_transform(y_pred)
        
        # Hausdorff loss
        hausdorff_loss = tf.reduce_mean(
            tf.pow(dt_true, self.alpha) + tf.pow(dt_pred, self.alpha)
        )
        
        return hausdorff_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config


class BoundaryLoss(tf.keras.losses.Loss):
    """Boundary Loss for better edge preservation in segmentation"""
    
    def __init__(
        self,
        theta0: float = 3.0,
        theta: float = 5.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'boundary_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.theta0 = theta0
        self.theta = theta
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate gradients for boundary detection
        def get_boundaries(mask):
            # Sobel operators
            sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
            sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
            
            sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
            sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
            
            if len(mask.shape) == 3:
                mask = tf.expand_dims(mask, -1)
            
            grad_x = tf.nn.conv2d(mask, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
            grad_y = tf.nn.conv2d(mask, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
            
            boundaries = tf.sqrt(grad_x**2 + grad_y**2)
            return tf.squeeze(boundaries, -1)
        
        # Get boundaries
        boundaries_true = get_boundaries(y_true)
        boundaries_pred = get_boundaries(y_pred)
        
        # Boundary loss
        boundary_loss = tf.reduce_mean(
            tf.abs(boundaries_true - boundaries_pred)
        )
        
        return boundary_loss


class DiceLoss(tf.keras.losses.Loss):
    """Dice loss for semantic segmentation"""
    
    def __init__(self,
        smooth: float = 1.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'dice_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        if hasattr(tf.keras.losses, 'Dice'):
            self.dice_loss = tf.keras.losses.Dice(
                smooth=smooth,
                reduction=reduction,
                name=name
            )
        else:
            self.dice_loss = None
            self.smooth = smooth
        
    def call(self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.dice_loss is not None:
            return self.dice_loss(y_true, y_pred)
        
        # Flatten the tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        y_true_flat = tf.cast(y_true_flat, tf.float32)
        y_pred_flat = tf.cast(y_pred_flat, tf.float32)
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
        
        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice_coef
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config


class AdaptiveDiceLoss(tf.keras.losses.Loss):
    """Adaptive Dice Loss with learnable smooth parameter"""
    
    def __init__(
        self,
        smooth: float = 1.0,
        adaptive: bool = True,
        power: float = 1.0,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'adaptive_dice_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.smooth = smooth
        self.adaptive = adaptive
        self.power = power
        
        if adaptive:
            self.smooth_var = tf.Variable(initial_value=smooth, trainable=True, name='smooth')
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Flatten tensors
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        # Calculate intersection and union with power
        intersection = tf.reduce_sum(tf.pow(y_true_flat * y_pred_flat, self.power))
        union = tf.reduce_sum(tf.pow(y_true_flat, self.power)) + tf.reduce_sum(tf.pow(y_pred_flat, self.power))
        
        # Use adaptive or fixed smooth
        smooth = self.smooth_var if self.adaptive else self.smooth
        smooth = tf.clip_by_value(smooth, 0.1, 10.0)
        
        # Calculate Dice coefficient
        dice_coef = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice_coef
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'smooth': self.smooth,
            'adaptive': self.adaptive,
            'power': self.power
        })
        return config


class ComboLoss(tf.keras.losses.Loss):
    """Combo Loss: Combines Dice and Cross-Entropy with optimal weighting"""
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        eps: float = 1e-9,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'combo_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Dice component
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
        dice_coef = (2.0 * intersection + self.eps) / (
            tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + self.eps
        )
        dice_loss = 1.0 - dice_coef
        
        # Cross-entropy component
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Combine losses
        combo_loss = self.alpha * ce_loss + self.beta * dice_loss
        
        return combo_loss
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'eps': self.eps
        })
        return config


class QualityFocalLoss(tf.keras.losses.Loss):
    """Quality Focal Loss - SOTA for object detection quality estimation"""
    
    def __init__(
        self,
        beta: float = 2.0,
        from_logits: bool = False,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
        name: str = 'quality_focal_loss'
    ):
        super().__init__(reduction=reduction, name=name)
        self.beta = beta
        self.from_logits = from_logits
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Calculate scaling factor
        scale_factor = tf.abs(y_true - y_pred)
        
        # Calculate focal loss with quality weighting
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        focal_loss = tf.pow(scale_factor, self.beta) * ce_loss
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'beta': self.beta,
            'from_logits': self.from_logits
        })
        return config


class UnifiedDetectionLoss(tf.keras.losses.Loss):
    """Unified Detection Loss combining multiple SOTA techniques"""
    
    def __init__(
        self,
        bbox_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        quality_loss_weight: float = 1.0,
        use_eiou: bool = True,
        use_varifocal: bool = True,
        use_quality_focal: bool = True,
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
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
            self.quality_loss = QualityFocalLoss()
        
    def call(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Cast inputs
        for key in y_true:
            y_true[key] = tf.cast(y_true[key], tf.float32)
            y_pred[key] = tf.cast(y_pred[key], tf.float32)
        
        # Calculate losses
        bbox_loss = self.bbox_loss(y_true['bbox'], y_pred['bbox'])
        cls_loss = self.cls_loss(y_true['class'], y_pred['class'])
        
        total_loss = (
            self.bbox_loss_weight * bbox_loss +
            self.cls_loss_weight * cls_loss
        )
        
        # Add quality loss if available
        if self.use_quality_focal and 'quality' in y_true:
            quality_loss = self.quality_loss(y_true['quality'], y_pred['quality'])
            total_loss += self.quality_loss_weight * quality_loss
        
        return total_loss
    
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
        reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
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
def get_sota_loss_function(loss_config: Dict[str, Any]) -> tf.keras.losses.Loss:
    """Factory function to get SOTA loss function based on configuration"""
    loss_type = loss_config.get('type', 'mse').lower()
    
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