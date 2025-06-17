from src.models.base_model import ModelBuilder
import tensorflow as tf
from tensorflow import keras
from typing import Tuple

from src.training.callbacks import *
from src.training.losses import *
from src.training.metrics import *
from src.training.optimizers import *

class UniversalTrainer:
    """
    Universal Training class cho Object Detection, Segmentation và Multitask
    Code ngắn gọn, tận dụng thư viện có sẵn
    """
    
    def __init__(self, model, task_type: str = 'detection', model_name: str = 'model'):
        """
        Args:
            task_type: 'detection', 'segmentation', 'multitask'
            model_name: Tên model để save
        """
        self.task_type = task_type.lower()
        self.model = model
        self.model_name = model_name
        self.history = None

    # === OPTIMIZERS ===
    def get_optimizer(self, optimizer_type: str = 'auto', learning_rate: float = 1e-3):
        """Get optimizer tối ưu cho từng task"""
        if optimizer_type == 'auto':
            optimizer_type = {
                'detection': 'adamw',
                'segmentation': 'lamb', 
                'multitask': 'lookahead'
            }[self.task_type]
        
        optimizers_map = {
            'adamw': tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
            'lamb': tf.keras.optimizers.Lamb(learning_rate=learning_rate),
            'lookahead': tf.keras.optimizers.Lookahead(
                tf.keras.optimizers.AdamW(learning_rate=learning_rate)
            ),
            'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate)
        }
        
        return optimizers_map.get(optimizer_type, keras.optimizers.Adam(learning_rate))
    
    # === LOSSES ===
    def get_loss(self, loss_type: str = 'auto'):
        """Get loss function tối ưu cho từng task"""
        if loss_type == 'auto':
            loss_configs = {
                'detection': get_detection_loss(),
                'segmentation': get_segmentation_loss(),
                'multitask': get_multitask_loss()
            }
            return loss_configs[self.task_type]
        else:
            return loss_configs[loss_type]
            
    # === METRICS ===
    def get_metrics(self, metrics_type: str = 'auto'):
        """Get metrics tối ưu cho từng task"""
        metrics_configs = {
            'detection': SOTAMetrics.get_detection_metrics(),
            'segmentation': SOTAMetrics.get_segmentation_metrics(),
            'multitask': SOTAMetrics.get_multitask_metrics(),
        }
        return metrics_configs[self.task_type]
        
    
    # === CALLBACKS ===
    def get_callbacks(
        self, callback_type: str = 'sota', monitor: str = 'val_loss', 
                     patience: int = 15, **kwargs):
        """Get callbacks tối ưu"""
        if callback_type == 'sota':
            base_callbacks = get_sota_callbacks(monitor=monitor, patience=patience)
        elif callback_type == 'advanced':
            base_callbacks = get_advanced_callbacks(patience=patience, monitor=monitor)
        
        return base_callbacks
    
    # === MAIN TRAINING METHOD ===
    def train(self, train_data, val_data=None, epochs: int = 100, 
              optimizer: str = 'auto', loss: str = 'auto', 
              metrics: str = 'auto', callbacks_type: str = 'sota',
              learning_rate: float = 1e-3, **kwargs):
        """
        Main training method - Siêu đơn giản!
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            optimizer: Optimizer type ('auto', 'adamw', 'lamb', etc.)
            loss: Loss type ('auto', 'focal', 'dice', etc.)
            metrics: Metrics type ('auto' or custom list)
            callbacks_type: Callback type ('sota', 'advanced', 'basic')
            learning_rate: Learning rate
        """
        if self.model is None:
            raise ValueError("Model is not built! Call build_model() first.")
        
        # Compile model với auto-config
        self.model.compile(
            optimizer=self.get_optimizer(optimizer, learning_rate),
            loss=self.get_loss(loss),
            metrics=self.get_metrics(metrics)
        )
        
        # Get callbacks
        callback_list = self.get_callbacks(callbacks_type, **kwargs)
        
        # Fit model
        self.history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callback_list,
            verbose=1,
            **kwargs
        )
        
        return self.history
    
    # === UTILITY METHODS ===
    def save_model(self, filepath: str = None):
        """Save model"""
        if filepath is None:
            filepath = f'{self.model_name}_final.h5'
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, data):
        """Predict"""
        return self.model.predict(data)
    
    def evaluate(self, test_data):
        """Evaluate model"""
        return self.model.evaluate(test_data)

# === QUICK SETUP FUNCTIONS ===
def quick_detection_trainer(input_shape, num_classes, model_name='detection_model'):
    """Quick setup cho Object Detection"""
    trainer = UniversalTrainer('detection', model_name)
    trainer.build_model(input_shape, num_classes, 'efficientdet')
    return trainer

def quick_segmentation_trainer(input_shape, num_classes, model_name='segmentation_model'):
    """Quick setup cho Segmentation"""
    trainer = UniversalTrainer('segmentation', model_name)
    trainer.build_model(input_shape, num_classes, 'unet')
    return trainer

def quick_multitask_trainer(input_shape, num_classes, model_name='multitask_model'):
    """Quick setup cho Multitask"""
    trainer = UniversalTrainer('multitask', model_name)
    trainer.build_model(input_shape, num_classes)
    return trainer