from src.models.base_model import ModelBuilder
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import json
from datetime import datetime

from src.training.callbacks import *
from src.training.losses import *
from src.training.metrics import *
from src.training.optimizers import *

class UniversalTrainer:
    """
    Universal Training class for Object Detection, Segmentation and Multitask Learning.
    Provides a unified interface for training different types of models with optimized configurations.
    """
    
    def __init__(
        self, 
        model: tf.keras.Model,
        num_classes: int = 37,
        task_type: str = 'detection',
        model_name: str = 'model',
        mixed_precision: bool = True
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            num_classes: Number of classes for classification
            task_type: Type of task ('detection', 'segmentation', 'multitask')
            model_name: Name of the model for saving
            mixed_precision: Whether to use mixed precision training
        """
        self.task_type = task_type.lower()
        if self.task_type not in ['detection', 'segmentation', 'multitask']:
            raise ValueError(f"Invalid task_type: {task_type}. Must be one of ['detection', 'segmentation', 'multitask']")
        
        self.model = model
        self.num_classes = num_classes
        self.model_name = model_name
        self.history = None
        self.mixed_precision = mixed_precision
        
        # Enable mixed precision if requested
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Initialize training state
        self._training_state = {
            'is_training': False,
            'current_epoch': 0,
            'best_metric': float('inf'),
            'last_save_path': None
        }

    def get_optimizer(
        self, 
        optimizer_type: str = 'auto',
        learning_rate: float = 1e-3
    ) -> tf.keras.optimizers.Optimizer:
        """
        Get the appropriate optimizer for the task.

        Args:
            optimizer_type: Type of optimizer ('auto' or specific optimizer name)
            learning_rate: Learning rate for the optimizer

        Returns:
            Configured optimizer
        """
        try:
            if optimizer_type == 'auto':
                optimizer_type = {
                    'detection': SOTAOptimizers.get_detection_optimizer(),
                    'segmentation': SOTAOptimizers.get_segmentation_optimizer(),
                    'multitask': SOTAOptimizers.get_multitask_optimizer()
                }[self.task_type]
            
            return optimizer_type
        except Exception as e:
            raise ValueError(f"Failed to get optimizer: {str(e)}")

    def get_loss(
        self,
        loss_type: str = 'auto'
    ) -> Union[Dict[str, str], tf.keras.losses.Loss]:
        """
        Get the appropriate loss function for the task.

        Args:
            loss_type: Type of loss ('auto' or specific loss name)

        Returns:
            Configured loss function(s)
        """
        try:
            loss_configs = {
                'detection': get_detection_loss(),
                'segmentation': get_segmentation_loss(),
                'multitask': get_multitask_loss()
            }
            
            if loss_type == 'auto':
                return loss_configs[self.task_type]
            return loss_configs[loss_type]
        except Exception as e:
            raise ValueError(f"Failed to get loss function: {str(e)}")

    def get_metrics(
        self,
        metrics_type: str = 'auto'
    ) -> Union[Dict[str, List[str]], List[tf.keras.metrics.Metric]]:
        """
        Get the appropriate metrics for the task.

        Args:
            metrics_type: Type of metrics ('auto' or specific metrics name)

        Returns:
            Configured metrics
        """
        try:
            metrics_configs = {
                'detection': SOTAMetrics.get_detection_metrics_structured(
                    self.model, num_classes=self.num_classes
                ),
                'segmentation': SOTAMetrics.get_segmentation_metrics(
                    num_classes=self.num_classes
                ),
                'multitask': SOTAMetrics.get_multitask_metrics(
                    num_detection_classes=self.num_classes,
                    num_segmentation_classes=self.num_classes
                )
            }
            return metrics_configs[self.task_type]
        except Exception as e:
            raise ValueError(f"Failed to get metrics: {str(e)}")

    def get_callbacks(
        self,
        callback_type: str = 'sota',
        monitor: str = 'val_loss',
        patience: int = 15,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Get the appropriate callbacks for training.

        Args:
            callback_type: Type of callbacks ('sota', 'advanced', 'basic')
            monitor: Metric to monitor for early stopping
            patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
            **kwargs: Additional callback parameters

        Returns:
            List of configured callbacks
        """
        try:
            if callback_type == 'sota':
                base_callbacks = get_sota_callbacks(
                    monitor=monitor,
                    patience=patience,
                    save_dir=save_dir
                )
            elif callback_type == 'advanced':
                base_callbacks = get_advanced_callbacks(
                    patience=patience,
                    monitor=monitor,
                    save_dir=save_dir
                )
            else:
                raise ValueError(f"Unknown callback type: {callback_type}")
            
            return base_callbacks
        except Exception as e:
            raise ValueError(f"Failed to get callbacks: {str(e)}")

    def train(
        self,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        optimizer: str = 'auto',
        loss: str = 'auto',
        metrics: str = 'auto',
        callbacks_type: str = 'sota',
        learning_rate: float = 1e-3,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """
        Train the model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of epochs
            optimizer: Optimizer type
            loss: Loss type
            metrics: Metrics type
            callbacks_type: Callback type
            learning_rate: Learning rate
            save_dir: Directory to save model checkpoints
            **kwargs: Additional training parameters

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model is not built! Call build_model() first.")
        
        try:
            self._training_state['is_training'] = True
            
            # Get optimizer, loss, and metrics
            optimizer = self.get_optimizer(optimizer, learning_rate)
            loss_fn = self.get_loss(loss)
            metrics_fn = self.get_metrics(metrics)
            
            # Configure loss and metrics based on task type
            if self.task_type == 'detection':
                loss_fn = {
                    'bbox': 'mse',
                    'label': 'sparse_categorical_crossentropy'
                }
                metrics_fn = {
                    'bbox': ['mae'],
                    'label': ['accuracy']
                }
            elif self.task_type == 'multitask':
                loss_fn = {
                    'bbox': 'mse',
                    'label': 'sparse_categorical_crossentropy',
                    'segmentation': 'sparse_categorical_crossentropy'
                }
                metrics_fn = {
                    'bbox': ['mae'],
                    'label': ['accuracy'],
                    'segmentation': ['accuracy']
                }
            
            # Compile model
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics_fn
            )
            
            # Get callbacks
            callback_list = self.get_callbacks(
                callback_type,
                save_dir=save_dir,
                **kwargs
            )
            
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
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
        finally:
            self._training_state['is_training'] = False

    def save_model(
        self,
        filepath: Optional[str] = None,
        save_format: str = 'h5',
        include_optimizer: bool = True
    ) -> str:
        """
        Save the model.

        Args:
            filepath: Path to save the model
            save_format: Format to save the model ('h5' or 'tf')
            include_optimizer: Whether to save optimizer state

        Returns:
            Path where the model was saved
        """
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f'{self.model_name}_{timestamp}.{save_format}'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save(
                filepath,
                save_format=save_format,
                include_optimizer=include_optimizer
            )
            
            # Save training state
            state_file = f"{filepath}.state.json"
            with open(state_file, 'w') as f:
                json.dump(self._training_state, f)
            
            self._training_state['last_save_path'] = filepath
            print(f"Model saved to {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")

    def load_model(
        self,
        filepath: str,
        custom_objects: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Load a saved model.

        Args:
            filepath: Path to the saved model
            custom_objects: Custom objects to load with the model
        """
        try:
            self.model = keras.models.load_model(
                filepath,
                custom_objects=custom_objects
            )
            
            # Load training state if available
            state_file = f"{filepath}.state.json"
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    self._training_state = json.load(f)
            
            print(f"Model loaded from {filepath}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(
        self,
        data: tf.data.Dataset,
        batch_size: Optional[int] = None
    ) -> tf.Tensor:
        """
        Make predictions using the model.

        Args:
            data: Input data
            batch_size: Batch size for prediction

        Returns:
            Model predictions
        """
        try:
            return self.model.predict(
                data,
                batch_size=batch_size
            )
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def evaluate(
        self,
        test_data: tf.data.Dataset,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            test_data: Test dataset
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            results = self.model.evaluate(
                test_data,
                batch_size=batch_size,
                return_dict=True
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")

# === QUICK SETUP FUNCTIONS ===
def quick_detection_trainer(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    model_name: str = 'detection_model'
) -> UniversalTrainer:
    """
    Quick setup for Object Detection.

    Args:
        input_shape: Input shape of the model
        num_classes: Number of classes
        model_name: Name of the model

    Returns:
        Configured UniversalTrainer instance
    """
    trainer = UniversalTrainer(
        task_type='detection',
        model_name=model_name
    )
    trainer.build_model(input_shape, num_classes, 'efficientdet')
    return trainer

def quick_segmentation_trainer(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    model_name: str = 'segmentation_model'
) -> UniversalTrainer:
    """
    Quick setup for Segmentation.

    Args:
        input_shape: Input shape of the model
        num_classes: Number of classes
        model_name: Name of the model

    Returns:
        Configured UniversalTrainer instance
    """
    trainer = UniversalTrainer(
        task_type='segmentation',
        model_name=model_name
    )
    trainer.build_model(input_shape, num_classes, 'unet')
    return trainer

def quick_multitask_trainer(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    model_name: str = 'multitask_model'
) -> UniversalTrainer:
    """
    Quick setup for Multitask Learning.

    Args:
        input_shape: Input shape of the model
        num_classes: Number of classes
        model_name: Name of the model

    Returns:
        Configured UniversalTrainer instance
    """
    trainer = UniversalTrainer(
        task_type='multitask',
        model_name=model_name
    )
    trainer.build_model(input_shape, num_classes)
    return trainer