from tensorflow.keras.metrics import (
    MeanIoU, Precision, Recall, BinaryAccuracy, AUC, 
    CategoricalAccuracy, TopKCategoricalAccuracy, 
    MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, 
    BinaryCrossentropy, CategoricalCrossentropy
)
import tensorflow as tf

# === Float32-safe metric wrappers ===
class Float32CategoricalAccuracy(CategoricalAccuracy):
    """CategoricalAccuracy that safely casts predictions to float32 for mixed-precision models."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        return super().update_state(y_true, y_pred, sample_weight)

class Float32TopKCategoricalAccuracy(TopKCategoricalAccuracy):
    """TopKCategoricalAccuracy that safely casts predictions to float32 for mixed-precision models."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        return super().update_state(y_true, y_pred, sample_weight)


class SOTAMetrics:
    # ========= SEGMENTATION METRICS =========
    @staticmethod
    def get_segmentation_metrics(num_classes: int):
        return [
            MeanIoU(num_classes=num_classes, name='iou'),
            # Nếu muốn nhiều tên IoU có thể tạo thêm instance khác (nhưng nên dùng 1 thôi)
            # F1 Macro/Micro (custom)
            SOTAMetrics._f1_score(average='macro', num_classes=num_classes, name='f1_macro'),
            SOTAMetrics._f1_score(average='micro', num_classes=num_classes, name='f1_micro'),
            Precision(name='precision'),
            Recall(name='recall'),
            BinaryAccuracy(name='accuracy'),
            AUC(name='auc'),
            SOTAMetrics._dice_coefficient(),
        ]
    
    @staticmethod
    def _dice_coefficient():
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

    @staticmethod
    def _f1_score(average='macro', num_classes=2, name='f1'):
        class F1Score(tf.keras.metrics.Metric):
            def __init__(self, name=name, average=average, num_classes=num_classes, **kwargs):
                super().__init__(name=name, **kwargs)
                self.average = average
                self.num_classes = num_classes
                self.precision = Precision()
                self.recall = Recall()
            def update_state(self, y_true, y_pred, sample_weight=None):
                self.precision.update_state(y_true, y_pred, sample_weight)
                self.recall.update_state(y_true, y_pred, sample_weight)
            def result(self):
                p = self.precision.result()
                r = self.recall.result()
                return 2 * p * r / (p + r + 1e-7)
            def reset_state(self):
                self.precision.reset_state()
                self.recall.reset_state()
        return F1Score(name=name, average=average, num_classes=num_classes)

    # ========= OBJECT DETECTION METRICS =========
    @staticmethod
    def get_detection_metrics_structured(model, num_classes: int):
        """
        Returns detection metrics properly structured for multi-output model
        """
        # Option 1: If you know your output names
        if hasattr(model, 'output_names') and len(model.output_names) == 2:
            output1_name = model.output_names[0]  # e.g., 'classification'
            output2_name = model.output_names[1]  # e.g., 'detection'
            
            return {
                output1_name: [
                    Float32CategoricalAccuracy(name='cls_accuracy'),
                    Float32TopKCategoricalAccuracy(k=3, name='cls_top3'),
                ],
                output2_name: [
                    MeanIoU(num_classes=num_classes, name='detection_iou'),
                    MeanAbsoluteError(name='box_mae'),
                    MeanSquaredError(name='box_mse'),
                    RootMeanSquaredError(name='box_rmse'),
                    BinaryAccuracy(name='obj_accuracy'),
                    Precision(name='obj_precision'),
                    Recall(name='obj_recall'),
                    AUC(name='obj_auc'),
                    # Add your mAP approximation here
                    SOTAMetrics._map_approximation()
                ]
            }
        
        # Option 2: Use indexed outputs (most common case)
        else:
            return [
                # Metrics for output 0 (classification)
                [
                    Float32CategoricalAccuracy(name='cls_accuracy'),
                    Float32TopKCategoricalAccuracy(k=3, name='cls_top3'),
                ],
                
                # Metrics for output 1 (detection/boxes)
                [
                    MeanIoU(num_classes=num_classes, name='detection_iou'),
                    MeanAbsoluteError(name='box_mae'),
                    MeanSquaredError(name='box_mse'),
                    RootMeanSquaredError(name='box_rmse'),
                    BinaryAccuracy(name='obj_accuracy'),
                    Precision(name='obj_precision'),
                    Recall(name='obj_recall'),
                    AUC(name='obj_auc'),
                    # Add your mAP approximation here
                    SOTAMetrics._map_approximation(),
                ]
            ]

    @staticmethod
    def _map_approximation():
        class mAPApproximation(tf.keras.metrics.Metric):
            def __init__(self, name='map_approx', **kwargs):
                super().__init__(name=name, **kwargs)
                self.map_sum = self.add_weight(name='map_sum', initializer='zeros')
                self.count = self.add_weight(name='count', initializer='zeros')
            
            def update_state(self, y_true, y_pred, sample_weight=None):
                # Calculate IoU for bounding boxes
                iou = tf.reduce_mean(tf.image.iou(y_true[..., :4], y_pred[..., :4], [2]))
                
                # Calculate confidence score
                conf = tf.reduce_mean(y_pred[..., 4])  # objectness
                
                # Approximate mAP using IoU and confidence
                map_score = iou * conf
                
                self.map_sum.assign_add(map_score)
                self.count.assign_add(1.)
            
            def result(self):
                return self.map_sum / self.count
            
            def reset_state(self):
                self.map_sum.assign(0.)
                self.count.assign(0.)
        
        return mAPApproximation()

    # ========= CLASSIFICATION METRICS =========
    @staticmethod
    def get_classification_metrics(num_classes: int):
        if num_classes == 2:
            return [
                BinaryAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc'),
                SOTAMetrics._f1_score(num_classes=2, name='f1'),
                BinaryCrossentropy(name='bce'),
            ]
        else:
            return [
                Float32CategoricalAccuracy(name='accuracy'),
                Float32TopKCategoricalAccuracy(k=5, name='top5'),
                SOTAMetrics._f1_score(num_classes=num_classes, average='macro', name='f1_macro'),
                SOTAMetrics._f1_score(num_classes=num_classes, average='micro', name='f1_micro'),
                CategoricalCrossentropy(name='cce'),
            ]

    # ========= MULTITASK METRICS =========
    @staticmethod
    def get_multitask_metrics(num_detection_classes: int, num_segmentation_classes: int):
        return {
            'detection': [
                MeanIoU(num_classes=num_detection_classes, name='det_iou'),
                MeanAbsoluteError(name='det_mae'),
                BinaryAccuracy(name='det_acc'),
            ],
            'segmentation': [
                MeanIoU(num_classes=num_segmentation_classes, name='seg_iou'),
                SOTAMetrics._f1_score(num_classes=num_segmentation_classes, name='seg_f1'),
                BinaryAccuracy(name='seg_acc'),
                SOTAMetrics._dice_coefficient(),
            ],
            'classification': [
                Float32CategoricalAccuracy(name='cls_acc'),
                Float32TopKCategoricalAccuracy(k=5, name='cls_top5'),
                SOTAMetrics._f1_score(num_classes=num_detection_classes, average='macro', name='cls_f1'),
            ]
        }
