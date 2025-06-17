from typing import Dict, Optional
import tensorflow as tf
from pathlib import Path

from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.models.base_model import BaseModel
from src.training.losses import get_loss
from src.training.optimizers import create_detection_optimizer, create_segmentation_optimizer, create_classification_optimizer
from src.training.metrics import get_metrics
from src.training.callbacks import get_callbacks

class Trainer:
    """Simplified and optimized trainer using TensorFlow's built-in features."""
    
    def __init__(
        self,
        model: BaseModel,
        task_type: str,
        backbone_name: str,
        config: Optional[Config] = None,
        models_config: Optional[ModelConfigs] = None,
    ):
        self.model = model
        self.task_type = task_type
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        
        # Setup all components
        self._setup_components(backbone_name)
        
    def _setup_components(self, backbone_name: str):
        """Setup all training components efficiently."""
        # Optimizer
        if self.task_type == 'detection':
            self.optimizer = create_detection_optimizer(
                model=backbone_name,
                optimizer_type=getattr(self.config, 'OPTIMIZER_TYPE', 'sgd'),
                custom_lr=getattr(self.config, 'LEARNING_RATE', None),
                batch_size=getattr(self.config, 'BATCH_SIZE', 16),
                total_epochs=getattr(self.config, 'EPOCHS', 300),
                backbone=backbone_name,
            )
        elif self.task_type == 'segmentation':
            self.optimizer = create_segmentation_optimizer(
                model=backbone_name,
                optimizer_type=getattr(self.config, 'OPTIMIZER_TYPE', 'sgd'),
                custom_lr=getattr(self.config, 'LEARNING_RATE', None),
                batch_size=getattr(self.config, 'BATCH_SIZE', 16),
                total_epochs=getattr(self.config, 'EPOCHS', 300),
                backbone=backbone_name,
            )
        elif self.task_type == 'classification':
            self.optimizer = create_classification_optimizer(
                model=backbone_name,
                optimizer_type=getattr(self.config, 'OPTIMIZER_TYPE', 'sgd'),
                custom_lr=getattr(self.config, 'LEARNING_RATE', None),
                batch_size=getattr(self.config, 'BATCH_SIZE', 16),
                total_epochs=getattr(self.config, 'EPOCHS', 300),
                backbone=backbone_name,
            )
        
        # Loss and metrics
        loss_cfg = self.models_config.LOSS_CONFIGS[self.task_type]
        if isinstance(loss_cfg, dict):
            loss_type = loss_cfg.get('loss_type', list(loss_cfg.keys())[0] if loss_cfg else 'mse')
            extra_args = {k: v for k, v in loss_cfg.items() if k != 'loss_type'}
            self.loss_fn = get_loss(loss_type, **extra_args)
        else:
            self.loss_fn = get_loss(loss_cfg)

        self.metrics = get_metrics(self.task_type, self.config.NUM_CLASSES_DETECTION)
        
        # Mixed precision
        if getattr(self.config, 'USE_MIXED_PRECISION', False):
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
        
        # Memory optimization
        self._setup_memory_optimization()
        
        # Callbacks (let TensorFlow handle most of them)
        self.callbacks = self._get_callbacks(backbone_name)
        
    def _setup_memory_optimization(self):
        """Setup memory optimization."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    # Only set memory growth if it hasn't been set previously
                    if not tf.config.experimental.get_memory_growth(gpu):
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    # GPU has already been initialized; memory growth setting is immutable
                    pass
        
        if getattr(self.config, 'USE_XLA', False):
            tf.config.optimizer.set_jit(True)
    
    def _get_callbacks(self, backbone_name: str):
        """Get optimized callbacks."""
        log_dir = Path(self.config.LOGS_DIR) / self.task_type
        ckpt_dir = Path(self.config.RESULTS_DIR) / 'checkpoints' / self.task_type
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        return get_callbacks(
            task=self.task_type,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            enable_lr_schedule=True,  # This will work now
            optimizer_has_schedule=False  # Indicates optimizer has no schedule
        )

    def _resolve_initial_lr(self) -> float:
        """Safely derive a scalar learning-rate for callbacks without invoking schedule evaluation inside Keras property."""
        import numbers, tensorflow as tf
        # Try to access the raw hyper-parameter without triggering the @property logic.
        lr_obj = getattr(self.optimizer, '_learning_rate', None)
        if lr_obj is None:
            # Fallback to internal hyper storage (works for most built-in optimizers)
            try:
                lr_obj = self.optimizer._get_hyper('learning_rate')
            except Exception:
                lr_obj = None
        if lr_obj is None:
            return float(getattr(self.config, 'LEARNING_RATE', 1e-3))
        # ------------------------------------------------------------------
        # Convert to scalar float
        # ------------------------------------------------------------------
        if isinstance(lr_obj, numbers.Number):
            return float(lr_obj)
        if isinstance(lr_obj, tf.Variable):
            return float(lr_obj.numpy())
        # If it's a schedule or callable, evaluate at step 0
        try:
            val = lr_obj(0) if callable(lr_obj) else lr_obj
            if hasattr(val, 'numpy'):
                val = val.numpy()
            return float(val)
        except Exception:
            return float(getattr(self.config, 'LEARNING_RATE', 1e-3))
    

    
    @tf.function
    def _train_step(self, batch):
        """Unified training step for all task types."""
        inputs, targets = batch
        
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
            
            # Handle mixed precision
            if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                loss = self.optimizer.get_scaled_loss(loss)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                metric.update_state(targets, predictions)
        else:
            self.metrics.update_state(targets, predictions)
        
        return loss
    
    @tf.function
    def _val_step(self, batch):
        """Unified validation step for all task types."""
        inputs, targets = batch
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(targets, predictions)
        
        # Update metrics
        if isinstance(self.metrics, list):
            for metric in self.metrics:
                metric.update_state(targets, predictions)
        else:
            self.metrics.update_state(targets, predictions)
        
        return loss
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100
    ):
        """Simplified training loop using TensorFlow's Model.fit approach."""
        
        # Optimize datasets
        train_dataset = self._optimize_dataset(train_dataset, True)
        if val_dataset:
            val_dataset = self._optimize_dataset(val_dataset, False)
        
        # Compile model (let TensorFlow handle the training loop)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics if isinstance(self.metrics, list) else [self.metrics]
        )
        
        # Use TensorFlow's built-in fit method
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=1
        )
        
        return history
    
    def _optimize_dataset(self, dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
        """Optimize dataset using TensorFlow best practices."""
        batch_size = getattr(self.config, 'BATCH_SIZE', 16)
        
        if is_training:
            dataset = dataset.shuffle(1000)
        
        # Assume dataset is already batched by the data loader to avoid double-batching issues.
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset