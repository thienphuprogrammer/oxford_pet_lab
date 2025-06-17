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
def get_detection_loss():
    return ObjectDetectionLoss.yolo_loss(num_classes=80, anchors=None)

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
