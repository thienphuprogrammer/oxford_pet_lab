import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import *

class SOTALoss:
    """SOTA Loss functions cho Object Detection, Segmentation và Multitask"""
    
    @staticmethod
    def focal_loss(alpha=0.25, gamma=2.0, from_logits=False):
        """
        Sparse Categorical Focal Loss - cho multiclass classification
        """
        def loss_fn(y_true, y_pred):
            # Convert to probabilities if logits
            if from_logits:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            
            # Clip predictions to prevent log(0)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            
            # Convert sparse labels to one-hot if needed
            if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
                num_classes = tf.shape(y_pred)[-1]
                y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
            else:
                y_true_one_hot = y_true
            
            # Calculate cross entropy
            ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
            
            # Get the probability of the true class
            p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
            
            # Calculate focal weight
            focal_weight = alpha * tf.pow(1 - p_t, gamma)
            
            # Apply focal weight
            focal_loss = focal_weight * ce
            
            return tf.reduce_mean(focal_loss)
        
        return loss_fn
    
    @staticmethod
    def sparse_categorical_focal_loss(alpha=0.25, gamma=2.0, from_logits=False):
        """
        Alternative implementation using SparseCategoricalCrossentropy as base
        """
        def loss_fn(y_true, y_pred):
            # Standard sparse categorical crossentropy
            scce = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=from_logits, reduction='none'
            )
            ce_loss = scce(y_true, y_pred)
            
            # Convert to probabilities if needed
            if from_logits:
                y_pred = tf.nn.softmax(y_pred, axis=-1)
            
            # Get probability of true class
            y_true_int = tf.cast(y_true, tf.int32)
            if len(y_true.shape) > 1:
                y_true_int = tf.squeeze(y_true_int, axis=-1)
            
            # Gather probabilities for true classes
            p_t = tf.gather(y_pred, y_true_int, axis=-1, batch_dims=1)
            
            # Calculate focal weight
            focal_weight = alpha * tf.pow(1 - p_t, gamma)
            
            # Apply focal weight
            focal_loss = focal_weight * ce_loss
            
            return tf.reduce_mean(focal_loss)
        
        return loss_fn
        """
        Binary focal loss - cho binary classification
        """
        return tf.keras.losses.BinaryFocalCrossentropy(
            alpha=alpha, 
            gamma=gamma, 
            from_logits=from_logits
        )
    
    @staticmethod  
    def dice_loss(smooth=1e-6):
        def loss_fn(y_true, y_pred):
            # Fix: use correct parameter name
            dice = tf.keras.metrics.Dice(smooth=smooth)
            return 1.0 - dice(y_true, y_pred)
        return loss_fn
    
    @staticmethod
    def iou_loss():
        """IoU Loss - Tốt cho object detection"""
        def loss_fn(y_true, y_pred):
            intersection = tf.reduce_sum(y_true * y_pred, axis=-1)
            union = tf.reduce_sum(y_true + y_pred, axis=-1) - intersection
            iou = intersection / (union + 1e-7)
            return 1.0 - tf.reduce_mean(iou)
        return loss_fn
    
    @staticmethod
    def combined_seg_loss(dice_weight=0.5, focal_weight=0.5):
        d_fn = SOTALoss.dice_loss()
        f_fn = SOTALoss.binary_focal_loss()  # Use binary for segmentation
        def loss_fn(y_true, y_pred):
            return dice_weight * d_fn(y_true, y_pred) + focal_weight * f_fn(y_true, y_pred)
        return loss_fn

    @staticmethod
    def binary_focal_loss(alpha=0.25, gamma=2.0, from_logits=False):
        """
        Binary focal loss - cho binary classification
        """
        def loss_fn(y_true, y_pred):
            # Standard binary focal loss
            bce = tf.keras.losses.BinaryFocalCrossentropy(
                alpha=alpha, 
                gamma=gamma, 
                from_logits=from_logits
            )
            return bce(y_true, y_pred)
        return loss_fn

class ObjectDetectionLoss:
    """SOTA losses cho Object Detection"""
    
    @staticmethod
    def yolo_loss(num_classes):
        huber = tf.keras.losses.Huber()
        bce = tf.keras.losses.BinaryCrossentropy()
        sce = tf.keras.losses.SparseCategoricalCrossentropy()
        def loss_fn(y_true, y_pred):
            box_loss  = tf.reduce_mean(huber(y_true[..., :4], y_pred[..., :4]))
            obj_loss  = tf.reduce_mean(bce   (y_true[..., 4:5], y_pred[..., 4:5]))
            cls_loss  = tf.reduce_mean(sce   (y_true[..., 5:],  y_pred[..., 5:]))
            return box_loss + obj_loss + cls_loss
        return loss_fn
    
    @staticmethod
    def fcos_loss(cls_weight=1.0, reg_weight=1.0, center_weight=1.0):
        """
        FCOS-style loss - expects structured input
        y_true should be dict/tuple: {'cls': cls_true, 'reg': reg_true, 'center': center_true}
        y_pred should be dict/tuple: {'cls': cls_pred, 'reg': reg_pred, 'center': center_pred}
        """
        def loss_fn(y_true, y_pred):
            total_loss = 0.0
            
            # Handle different input formats
            if isinstance(y_true, dict) and isinstance(y_pred, dict):
                # Dictionary format
                if 'cls' in y_true and 'cls' in y_pred:
                    cls_loss = SOTALoss.focal_loss()(y_true['cls'], y_pred['cls'])
                    total_loss += cls_weight * cls_loss
                
                if 'reg' in y_true and 'reg' in y_pred:
                    reg_loss = tf.keras.losses.huber(y_true['reg'], y_pred['reg'])
                    total_loss += reg_weight * reg_loss
                
                if 'center' in y_true and 'center' in y_pred:
                    center_loss = tf.keras.losses.binary_crossentropy(
                        y_true['center'], y_pred['center']
                    )
                    total_loss += center_weight * center_loss
            
            elif isinstance(y_true, (list, tuple)) and isinstance(y_pred, (list, tuple)):
                # List/tuple format: [cls, reg, center]
                if len(y_true) >= 1 and len(y_pred) >= 1:
                    cls_loss = SOTALoss.focal_loss()(y_true[0], y_pred[0])
                    total_loss += cls_weight * cls_loss
                
                if len(y_true) >= 2 and len(y_pred) >= 2:
                    reg_loss = tf.keras.losses.huber(y_true[1], y_pred[1])
                    total_loss += reg_weight * reg_loss
                
                if len(y_true) >= 3 and len(y_pred) >= 3:
                    center_loss = tf.keras.losses.binary_crossentropy(y_true[2], y_pred[2])
                    total_loss += center_weight * center_loss
            
            else:
                # Single tensor - assume classification only
                total_loss = SOTALoss.focal_loss()(y_true, y_pred)
            
            return total_loss
        
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
def get_detection_loss():
    return ObjectDetectionLoss.fcos_loss()

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
