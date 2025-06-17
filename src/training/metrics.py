import tensorflow as tf
from tensorflow.keras.metrics import *
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

class SOTAMetrics:
    """SOTA Metrics sử dụng built-in functions"""
    
    # ========== SEGMENTATION METRICS ==========
    @staticmethod
    def get_segmentation_metrics():
        """Bộ metrics hoàn chỉnh cho segmentation"""
        return [
            # IoU metrics
            tfa.metrics.MeanIoU(num_classes=2, name='iou'),
            tfa.metrics.MeanIoU(num_classes=2, name='miou'),
            
            # F1 Score variants
            tfa.metrics.F1Score(num_classes=2, average='macro', name='f1_macro'),
            tfa.metrics.F1Score(num_classes=2, average='micro', name='f1_micro'),
            
            # Precision & Recall
            Precision(name='precision'),
            Recall(name='recall'),
            
            # Basic metrics
            BinaryAccuracy(name='accuracy'),
            AUC(name='auc'),
            
            # Custom Dice
            SOTAMetrics._dice_coefficient()
        ]
    
    @staticmethod
    def _dice_coefficient():
        """Dice coefficient metric"""
        class DiceCoefficient(tf.keras.metrics.Metric):
            def __init__(self, name='dice', **kwargs):
                super().__init__(name=name, **kwargs)
                self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
                self.count = self.add_weight(name='count', initializer='zeros')
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
                y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
                
                intersection = tf.reduce_sum(y_true * y_pred)
                union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
                dice = (2. * intersection + 1e-7) / (union + 1e-7)
                
                self.dice_sum.assign_add(dice)
                self.count.assign_add(1.)
            
            def result(self):
                return self.dice_sum / self.count
            
            def reset_state(self):
                self.dice_sum.assign(0.)
                self.count.assign(0.)
        
        return DiceCoefficient()
    
    # ========== OBJECT DETECTION METRICS ==========
    @staticmethod
    def get_detection_metrics():
        """Bộ metrics cho object detection"""
        return [
            # mAP approximation using IoU
            tfa.metrics.MeanIoU(num_classes=2, name='detection_iou'),
            
            # Box regression metrics  
            MeanAbsoluteError(name='box_mae'),
            MeanSquaredError(name='box_mse'),
            RootMeanSquaredError(name='box_rmse'),
            
            # Classification metrics for objectness
            BinaryAccuracy(name='obj_accuracy'),
            Precision(name='obj_precision'),
            Recall(name='obj_recall'),
            AUC(name='obj_auc'),
            
            # Multi-class classification for classes
            CategoricalAccuracy(name='cls_accuracy'),
            TopKCategoricalAccuracy(k=5, name='cls_top5'),
            
            # Custom mAP approximation
            SOTAMetrics._map_approximation()
        ]
    
    @staticmethod
    def _map_approximation():
        """mAP approximation metric"""
        class mAPApproximation(tf.keras.metrics.Metric):
            def __init__(self, name='map_approx', **kwargs):
                super().__init__(name=name, **kwargs)
                self.map_sum = self.add_weight(name='map_sum', initializer='zeros')
                self.count = self.add_weight(name='count', initializer='zeros')
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                # Simplified mAP calculation
                # In practice, use official COCO evaluation
                iou = tfa.losses.giou_loss(y_true[..., :4], y_pred[..., :4])
                conf = tf.reduce_mean(y_pred[..., 4])  # objectness confidence
                
                map_score = (1 - iou) * conf  # Simplified mAP
                self.map_sum.assign_add(tf.reduce_mean(map_score))
                self.count.assign_add(1.)
            
            def result(self):
                return self.map_sum / self.count
            
            def reset_state(self):
                self.map_sum.assign(0.)
                self.count.assign(0.)
        
        return mAPApproximation()
    
    # ========== CLASSIFICATION METRICS ==========
    @staticmethod
    def get_classification_metrics(num_classes):
        """Bộ metrics cho classification"""
        if num_classes == 2:
            return [
                BinaryAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc'),
                tfa.metrics.F1Score(num_classes=2, name='f1'),
                BinaryCrossentropy(name='bce'),
            ]
        else:
            return [
                CategoricalAccuracy(name='accuracy'),
                TopKCategoricalAccuracy(k=5, name='top5'),
                tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='f1_macro'),
                tfa.metrics.F1Score(num_classes=num_classes, average='micro', name='f1_micro'),
                CategoricalCrossentropy(name='cce'),
            ]
    
    # ========== MULTITASK METRICS ==========
    @staticmethod
    def get_multitask_metrics():
        """Metrics cho multitask learning"""
        return {
            # Detection task
            'detection': [
                tfa.metrics.MeanIoU(num_classes=2, name='det_iou'),
                MeanAbsoluteError(name='det_mae'),
                BinaryAccuracy(name='det_acc'),
            ],
            
            # Segmentation task  
            'segmentation': [
                tfa.metrics.MeanIoU(num_classes=2, name='seg_iou'),
                tfa.metrics.F1Score(num_classes=2, name='seg_f1'),
                BinaryAccuracy(name='seg_acc'),
                SOTAMetrics._dice_coefficient(),
            ],
            
            # Classification task
            'classification': [
                CategoricalAccuracy(name='cls_acc'),
                TopKCategoricalAccuracy(k=5, name='cls_top5'),
                tfa.metrics.F1Score(num_classes=10, average='macro', name='cls_f1'),
            ]
        }

# ==========================================================
# QUICK SETUP FUNCTIONS - CỰC KỲ ĐƠN GIẢN
# ==========================================================

def get_segmentation_setup():
    """Setup hoàn chỉnh cho segmentation model"""
    return {
        'metrics': SOTAMetrics.get_segmentation_metrics(),
        'monitor': 'val_iou',  # metric để monitor
        'mode': 'max'
    }

def get_detection_setup():
    """Setup hoàn chỉnh cho detection model"""  
    return {
        'metrics': SOTAMetrics.get_detection_metrics(),
        'monitor': 'val_map_approx',
        'mode': 'max'
    }

def get_classification_setup(num_classes=10):
    """Setup hoàn chỉnh cho classification model"""
    return {
        'metrics': SOTAMetrics.get_classification_metrics(num_classes),
        'monitor': 'val_f1' if num_classes == 2 else 'val_f1_macro',
        'mode': 'max'
    }

def get_multitask_setup():
    """Setup hoàn chỉnh cho multitask model"""
    return {
        'metrics': SOTAMetrics.get_multitask_metrics(),
        'monitor': 'val_seg_iou',  # hoặc metric khác
        'mode': 'max'
    }
