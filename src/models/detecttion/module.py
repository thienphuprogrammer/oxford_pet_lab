import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class AttentionBlock(layers.Layer):
    """Self-attention mechanism for feature enhancement"""
    
    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        
    def build(self, input_shape):
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(self.channels // self.reduction, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, self.channels))
        super().build(input_shape)
    
    def call(self, inputs):
        # Channel attention
        attention = self.global_pool(inputs)
        attention = self.fc1(attention)
        attention = self.fc2(attention)
        attention = self.reshape(attention)
        
        # Apply attention
        return inputs * attention

class FPN(layers.Layer):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, feature_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        # Lateral connections
        self.lateral_convs = []
        self.output_convs = []
        
        num_levels = len(input_shape)
        for i in range(num_levels):
            lateral_conv = layers.Conv2D(
                self.feature_dim, 1, padding='same',
                name=f'lateral_conv_{i}'
            )
            output_conv = layers.Conv2D(
                self.feature_dim, 3, padding='same',
                name=f'output_conv_{i}'
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
        
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        super().build(input_shape)
    
    def call(self, inputs):
        # inputs should be a list of feature maps from different scales
        # From high resolution to low resolution
        
        # Start from the highest level feature
        laterals = []
        for i, feature in enumerate(inputs):
            lateral = self.lateral_convs[i](feature)
            laterals.append(lateral)
        
        # Top-down pathway
        fpn_features = []
        prev_feature = laterals[-1]  # Start from lowest resolution
        fpn_features.append(self.output_convs[-1](prev_feature))
        
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample previous feature
            upsampled = self.upsample(prev_feature)
            
            # Add with lateral connection
            # Ensure spatial dimensions match
            target_shape = tf.shape(laterals[i])
            upsampled = tf.image.resize(upsampled, [target_shape[1], target_shape[2]])
            
            merged = laterals[i] + upsampled
            fpn_feature = self.output_convs[i](merged)
            fpn_features.insert(0, fpn_feature)
            prev_feature = merged
        
        return fpn_features


class BiFPN(layers.Layer):
            def __init__(self, feature_dim=64, **kwargs):
                super().__init__(**kwargs)
                self.feature_dim = feature_dim
                
            def build(self, input_shape):
                num_levels = len(input_shape)
                
                # Resizing convs to match feature dimensions
                self.resample_convs = []
                for i in range(num_levels):
                    conv = layers.Conv2D(self.feature_dim, 1, padding='same', name=f'resample_{i}')
                    self.resample_convs.append(conv)
                
                # BiFPN convs
                self.bifpn_convs = []
                for i in range(num_levels):
                    conv = layers.Conv2D(self.feature_dim, 3, padding='same', name=f'bifpn_{i}')
                    self.bifpn_convs.append(conv)
                
                super().build(input_shape)
            
            def call(self, inputs):
                # Resample all inputs to same channel dimension
                resampled = []
                for i, inp in enumerate(inputs):
                    resampled.append(self.resample_convs[i](inp))
                
                # Top-down path
                top_down = []
                prev = resampled[-1]  # Start from lowest resolution
                top_down.append(prev)
                
                for i in range(len(resampled) - 2, -1, -1):
                    # Upsample and add
                    target_shape = tf.shape(resampled[i])
                    upsampled = tf.image.resize(prev, [target_shape[1], target_shape[2]])
                    merged = resampled[i] + upsampled
                    top_down.insert(0, merged)
                    prev = merged
                
                # Bottom-up path
                bottom_up = []
                prev = top_down[0]  # Start from highest resolution
                bottom_up.append(self.bifpn_convs[0](prev))
                
                for i in range(1, len(top_down)):
                    # Downsample and add
                    target_shape = tf.shape(top_down[i])
                    downsampled = layers.MaxPooling2D(pool_size=2)(prev)
                    downsampled = tf.image.resize(downsampled, [target_shape[1], target_shape[2]])
                    merged = top_down[i] + downsampled
                    bottom_up.append(self.bifpn_convs[i](merged))
                    prev = merged
                
                return bottom_up
        

# Training utilities
class FocalLoss(keras.losses.Loss):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Compute cross entropy
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # Compute p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        
        # Compute alpha_t
        alpha_t = tf.ones_like(y_true) * self.alpha
        
        # Compute focal weight
        focal_weight = alpha_t * tf.pow(1 - p_t, self.gamma)
        
        # Compute focal loss
        focal_loss = focal_weight * ce
        
        return tf.reduce_mean(focal_loss)

class IoULoss(keras.losses.Loss):
    """IoU Loss for bounding box regression"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        # y_true and y_pred shape: [batch_size, 4] (x_min, y_min, x_max, y_max)
        
        # Calculate intersection
        intersection_x_min = tf.maximum(y_true[:, 0], y_pred[:, 0])
        intersection_y_min = tf.maximum(y_true[:, 1], y_pred[:, 1])
        intersection_x_max = tf.minimum(y_true[:, 2], y_pred[:, 2])
        intersection_y_max = tf.minimum(y_true[:, 3], y_pred[:, 3])
        
        intersection_area = tf.maximum(0.0, intersection_x_max - intersection_x_min) * \
                           tf.maximum(0.0, intersection_y_max - intersection_y_min)
        
        # Calculate union
        area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
        area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
        union_area = area_true + area_pred - intersection_area
        
        # Calculate IoU
        iou = intersection_area / (union_area + 1e-7)
        
        # Return 1 - IoU as loss
        return 1 - tf.reduce_mean(iou)

# Example usage and training configuration
def get_training_config():
    """Get optimized training configuration"""
    return {
        'optimizer': keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-4
        ),
        'losses': {
            'bbox_output': IoULoss(),
            'class_output': FocalLoss(alpha=0.25, gamma=2.0)
        },
        'loss_weights': {
            'bbox_output': 1.0,
            'class_output': 1.0
        },
        'metrics': {
            'bbox_output': ['mae'],
            'class_output': ['accuracy', 'top_5_accuracy']
        },
        'callbacks': [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            )
        ]
    }

# Data augmentation utilities
class MixUp(layers.Layer):
    """MixUp augmentation for improved generalization"""
    
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.gamma([batch_size, 1, 1, 1], self.alpha, self.alpha)
        lambda_val = tf.minimum(lambda_val, 1.0)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_inputs = lambda_val * inputs + (1 - lambda_val) * tf.gather(inputs, indices)
        
        return mixed_inputs

class CutMix(layers.Layer):
    """CutMix augmentation for better localization"""
    
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def call(self, inputs, training=None):
        if not training:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.gamma([batch_size], self.alpha, self.alpha)
        lambda_val = tf.clip_by_value(lambda_val, 0.0, 1.0)
        
        # Calculate cut size
        cut_ratio = tf.sqrt(1.0 - lambda_val)
        cut_h = tf.cast(cut_ratio * tf.cast(height, tf.float32), tf.int32)
        cut_w = tf.cast(cut_ratio * tf.cast(width, tf.float32), tf.int32)
        
        # Random center point
        cy = tf.random.uniform([batch_size], 0, height, dtype=tf.int32)
        cx = tf.random.uniform([batch_size], 0, width, dtype=tf.int32)
        
        # Calculate bounding box
        y1 = tf.maximum(0, cy - cut_h // 2)
        y2 = tf.minimum(height, cy + cut_h // 2)
        x1 = tf.maximum(0, cx - cut_w // 2)
        x2 = tf.minimum(width, cx + cut_w // 2)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_inputs = tf.gather(inputs, indices)
        
        # Create mask
        mask = tf.ones_like(inputs)
        for i in range(batch_size):
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.stack([
                    tf.fill([y2[i] - y1[i], x2[i] - x1[i]], i),
                    tf.range(y1[i], y2[i])[:, None] * tf.ones([1, x2[i] - x1[i]], dtype=tf.int32),
                    tf.ones([y2[i] - y1[i], 1], dtype=tf.int32) * tf.range(x1[i], x2[i])[None, :]
                ], axis=-1),
                tf.zeros([y2[i] - y1[i], x2[i] - x1[i], tf.shape(inputs)[-1]])
            )
        
        # Apply cut and mix
        mixed_inputs = mask * inputs + (1.0 - mask) * shuffled_inputs
        
        return mixed_inputs

# Advanced training strategies
class CosineLRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule with warm restarts"""
    
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_restart_factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warm_restart_factor = warm_restart_factor
    
    def __call__(self, step):
        # Determine current cycle
        cycle = tf.floor(1 + tf.cast(step, tf.float32) / tf.cast(self.decay_steps, tf.float32))
        x = tf.cast(step, tf.float32) / tf.cast(self.decay_steps, tf.float32) - cycle + 1
        
        # Cosine annealing
        cosine_decay = 0.5 * (1 + tf.cos(3.14159 * x))
        
        # Apply warm restart scaling
        restart_factor = tf.pow(self.warm_restart_factor, cycle - 1)
        
        return self.alpha + (self.initial_learning_rate - self.alpha) * cosine_decay / restart_factor

# Model ensemble utilities
class ModelEnsemble:
    """Ensemble multiple models for better performance"""
    
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, inputs):
        """Ensemble prediction with weighted averaging"""
        predictions = []
        
        for model in self.models:
            pred = model(inputs, training=False)
            predictions.append(pred)
        
        # Weighted average
        ensemble_bbox = tf.zeros_like(predictions[0]['bbox_output'])
        ensemble_class = tf.zeros_like(predictions[0]['class_output'])
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            ensemble_bbox += weight * pred['bbox_output']
            ensemble_class += weight * pred['class_output']
        
        return {
            'bbox_output': ensemble_bbox,
            'class_output': ensemble_class
        }
    
    def test_time_augmentation(self, inputs, augmentations=None):
        """Test time augmentation for robust predictions"""
        if augmentations is None:
            augmentations = [
                lambda x: x,  # Original
                lambda x: tf.image.flip_left_right(x),  # Horizontal flip
                lambda x: tf.image.rot90(x, k=1),  # 90° rotation
                lambda x: tf.image.rot90(x, k=3),  # 270° rotation
            ]
        
        tta_predictions = []
        
        for aug in augmentations:
            aug_inputs = aug(inputs)
            pred = self.predict(aug_inputs)
            
            # Reverse augmentation for bbox if needed
            if aug == tf.image.flip_left_right:
                # Adjust bbox coordinates for horizontal flip
                bbox = pred['bbox_output']
                bbox = tf.stack([
                    1.0 - bbox[:, 2],  # x_min = 1 - x_max
                    bbox[:, 1],        # y_min unchanged
                    1.0 - bbox[:, 0],  # x_max = 1 - x_min
                    bbox[:, 3]         # y_max unchanged
                ], axis=1)
                pred['bbox_output'] = bbox
            
            tta_predictions.append(pred)
        
        # Average TTA predictions
        avg_bbox = tf.reduce_mean(tf.stack([p['bbox_output'] for p in tta_predictions]), axis=0)
        avg_class = tf.reduce_mean(tf.stack([p['class_output'] for p in tta_predictions]), axis=0)
        
        return {
            'bbox_output': avg_bbox,
            'class_output': avg_class
        }

# Knowledge distillation utilities
class KnowledgeDistillation:
    """Knowledge distillation for model compression"""
    
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
    
    def distillation_loss(self, y_true, y_pred_student, y_pred_teacher):
        """Combined distillation loss"""
        
        # Hard target loss
        hard_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred_student)
        
        # Soft target loss (knowledge distillation)
        teacher_soft = tf.nn.softmax(y_pred_teacher / self.temperature)
        student_soft = tf.nn.softmax(y_pred_student / self.temperature)
        
        soft_loss = keras.losses.categorical_crossentropy(teacher_soft, student_soft)
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_step(self, x, y_true):
        """Custom training step with knowledge distillation"""
        
        # Teacher predictions (no gradients)
        with tf.GradientTape() as tape:
            teacher_pred = self.teacher_model(x, training=False)
            student_pred = self.student_model(x, training=True)
            
            # Calculate distillation loss
            class_loss = self.distillation_loss(
                y_true['class_output'],
                student_pred['class_output'],
                teacher_pred['class_output']
            )
            
            # Bbox loss (standard)
            bbox_loss = keras.losses.mean_squared_error(
                y_true['bbox_output'],
                student_pred['bbox_output']
            )
            
            total_loss = class_loss + bbox_loss
        
        # Update student model
        gradients = tape.gradient(total_loss, self.student_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.student_model.trainable_variables))
        
        return {'loss': total_loss, 'class_loss': class_loss, 'bbox_loss': bbox_loss}

# Progressive training utilities
class ProgressiveTraining:
    """Progressive training with increasing input resolution"""
    
    def __init__(self, model, initial_size=128, final_size=224, stages=4):
        self.model = model
        self.initial_size = initial_size
        self.final_size = final_size
        self.stages = stages
        self.current_stage = 0
    
    def get_current_size(self):
        """Get current training size"""
        size_increment = (self.final_size - self.initial_size) / (self.stages - 1)
        current_size = self.initial_size + size_increment * self.current_stage
        return int(current_size)
    
    def should_increase_size(self, epoch, epochs_per_stage=10):
        """Check if should increase training size"""
        if epoch > 0 and epoch % epochs_per_stage == 0 and self.current_stage < self.stages - 1:
            self.current_stage += 1
            return True
        return False
    
    def resize_dataset(self, dataset, new_size):
        """Resize dataset to new size"""
        def resize_fn(image, label):
            image = tf.image.resize(image, [new_size, new_size])
            return image, label
        
        return dataset.map(resize_fn)

# Evaluation metrics
class DetectionMetrics:
    """Comprehensive detection metrics"""
    
    def __init__(self, num_classes, iou_thresholds=[0.5, 0.75]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
    
    def calculate_iou(self, boxes1, boxes2):
        """Calculate IoU between two sets of boxes"""
        # boxes format: [x_min, y_min, x_max, y_max]
        
        # Calculate intersection
        x1 = tf.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = tf.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = tf.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = tf.minimum(boxes1[:, 3], boxes2[:, 3])
        
        intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
        
        # Calculate areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate union
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        return iou
    
    def calculate_map(self, y_true_boxes, y_pred_boxes, y_true_classes, y_pred_classes, y_pred_scores):
        """Calculate mean Average Precision (mAP)"""
        aps = []
        
        for class_id in range(self.num_classes):
            # Filter predictions and ground truth for this class
            class_mask_true = y_true_classes == class_id
            class_mask_pred = y_pred_classes == class_id
            
            if not tf.reduce_any(class_mask_true):
                continue
            
            gt_boxes = y_true_boxes[class_mask_true]
            pred_boxes = y_pred_boxes[class_mask_pred]
            pred_scores = y_pred_scores[class_mask_pred]
            
            # Sort by confidence
            sorted_indices = tf.argsort(pred_scores, direction='DESCENDING')
            pred_boxes = tf.gather(pred_boxes, sorted_indices)
            pred_scores = tf.gather(pred_scores, sorted_indices)
            
            # Calculate AP for different IoU thresholds
            threshold_aps = []
            for iou_threshold in self.iou_thresholds:
                ap = self._calculate_ap_single_class(gt_boxes, pred_boxes, iou_threshold)
                threshold_aps.append(ap)
            
            aps.append(tf.reduce_mean(threshold_aps))
        
        return tf.reduce_mean(aps) if aps else 0.0
    
    def _calculate_ap_single_class(self, gt_boxes, pred_boxes, iou_threshold):
        """Calculate AP for single class and IoU threshold"""
        if len(pred_boxes) == 0:
            return 0.0
        
        # Match predictions to ground truth
        ious = self.calculate_iou(pred_boxes, gt_boxes)
        max_ious = tf.reduce_max(ious, axis=1)
        
        # Determine true positives
        tp = max_ious >= iou_threshold
        fp = ~tp
        
        # Calculate precision and recall
        tp_cumsum = tf.cumsum(tf.cast(tp, tf.float32))
        fp_cumsum = tf.cumsum(tf.cast(fp, tf.float32))
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        recall = tp_cumsum / len(gt_boxes)
        
        # Calculate AP using 11-point interpolation
        recall_levels = tf.linspace(0.0, 1.0, 11)
        ap = 0.0
        
        for r in recall_levels:
            precisions_above_r = precision[recall >= r]
            if len(precisions_above_r) > 0:
                ap += tf.reduce_max(precisions_above_r)
        
        return ap / 11.0
