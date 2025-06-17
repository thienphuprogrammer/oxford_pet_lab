from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K

from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.models.base_model import BaseModel
from src.training.losses import get_sota_loss_function
from src.training.optimizers import DetectionOptimizer
from src.training.metrics import get_metrics
from src.training.callbacks import get_optimized_callbacks
from src.utils.file_utils import save_json, load_json


@dataclass
class TrainingState:
    """Encapsulates training state for better management."""
    current_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_metric: float = 0.0
    training_time: float = 0.0
    is_training: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        return cls(**data)


class BaseTrainer(ABC):
    """Optimized abstract base trainer leveraging TensorFlow's built-in features."""
    
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
        self.backbone_name = backbone_name
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        
        # Initialize training state
        self.state = TrainingState()
        
        # Use TensorFlow's built-in logging
        self._setup_tf_logging()

        # Setup components
        self._setup_optimization()
        self._setup_loss_and_metrics()
        self._setup_callbacks()
        self._setup_mixed_precision()
        self._setup_memory_optimization()
        
        
    def _setup_tf_logging(self):
        """Setup TensorFlow's built-in logging and monitoring."""
        # Create log directory
        self.log_dir = Path(self.config.LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard logging
        self.tensorboard_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        # Use tf.print for performance-friendly logging
        self.step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
        
    def _setup_optimization(self):
        """Setup optimizer with enhanced configuration."""
        det_opt_factory = DetectionOptimizer(
            model_type='retinanet',
            backbone=self.backbone_name,
        )

        optimizer_type = getattr(self.config, 'OPTIMIZER_TYPE', 'sgd')
        batch_size = getattr(self.config, 'BATCH_SIZE', 16)
        total_epochs = getattr(self.config, 'EPOCHS', 300)
        custom_lr = getattr(self.config, 'LEARNING_RATE', None)

        self.optimizer = det_opt_factory.create_optimizer(
            optimizer_type=optimizer_type,
            custom_lr=custom_lr,
            batch_size=batch_size,
            total_epochs=total_epochs,
        )

        # Use TensorFlow's gradient accumulation strategy
        self.gradient_accumulation_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        if self.gradient_accumulation_steps > 1:
            self._setup_gradient_accumulation()
            
    def _setup_gradient_accumulation(self):
        """Setup gradient accumulation using TensorFlow's strategy."""
        self.accumulate_gradient_ops = []
        self.accumulated_gradients = []
        
        # Initialize accumulated gradients
        for var in self.model.trainable_variables:
            self.accumulated_gradients.append(
                tf.Variable(tf.zeros_like(var), trainable=False)
            )
            
    def _setup_loss_and_metrics(self):
        """Setup loss functions and metrics."""
        self.loss_fn = get_sota_loss_function(
            self.models_config.LOSS_CONFIGS[self.task_type]
        )
        
        # Use TensorFlow's built-in metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        
        # Custom metrics
        self.metrics = get_metrics(
            self.task_type,
            self.config.NUM_CLASSES_DETECTION
        )
        
        # Wrap metrics in TensorFlow's metric container
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
            
    def _setup_callbacks(self):
        """Setup training callbacks using TensorFlow's callback system."""
        pretrained = self.config.PRETRAINED
        
        # Get custom callbacks
        custom_callbacks = get_optimized_callbacks(
            task_type=self.task_type,
            backbone_name=self.backbone_name,
            pretrained=pretrained,
            config=self.config,
            model_config=self.models_config
        )
        
        # Add TensorFlow's built-in callbacks
        tf_callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                profile_batch=0  # Disable profiling for performance
            ),
            tf.keras.callbacks.CSVLogger(
                str(self.log_dir / 'training.csv'),
                append=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.log_dir / '.best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=1,
                min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        self.callbacks = custom_callbacks + tf_callbacks
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training using TensorFlow's built-in support."""
        if getattr(self.config, 'USE_MIXED_PRECISION', False):
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            
            # Scale loss for mixed precision
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
            tf.print("Mixed precision training enabled")
            
    def _setup_memory_optimization(self):
        """Setup memory optimization using TensorFlow's strategies."""
        # Enable memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    # Optional: Set memory limit
                    # tf.config.experimental.set_memory_limit(gpu, 1024 * 6)  # 6GB limit
            except RuntimeError as e:
                tf.print(f"Memory growth setting failed: {e}")
                
        # Enable XLA compilation for better performance
        if getattr(self.config, 'USE_XLA', False):
            tf.config.optimizer.set_jit(True)
            
    @tf.function
    def _distributed_train_step(self, batch):
        """Optimized training step using tf.function decorator."""
        with tf.GradientTape() as tape:
            predictions = self.model(batch['inputs'], training=True)
            loss = self.loss_fn(batch['targets'], predictions)
            
            # Handle mixed precision
            if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss
                
        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Handle mixed precision gradients
        if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
            
        # Apply gradients with optional accumulation
        if self.gradient_accumulation_steps > 1:
            self._accumulate_gradients(gradients)
        else:
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
        # Update metrics
        self.train_loss.update_state(loss)
        for metric in self.metrics:
            if hasattr(metric, 'update_state'):
                metric.update_state(batch['targets'], predictions)
                
        return {'loss': loss, 'predictions': predictions}
    
    @tf.function
    def _distributed_val_step(self, batch):
        """Optimized validation step using tf.function decorator."""
        predictions = self.model(batch['inputs'], training=False)
        loss = self.loss_fn(batch['targets'], predictions)
        
        # Update validation metrics
        self.val_loss.update_state(loss)
        
        return {'loss': loss, 'predictions': predictions}
        
    def _accumulate_gradients(self, gradients):
        """Accumulate gradients using TensorFlow operations."""
        # Add gradients to accumulated gradients
        for i, grad in enumerate(gradients):
            if grad is not None:
                self.accumulated_gradients[i].assign_add(grad)
                
        self.step_counter.assign_add(1)
        
        # Apply gradients when accumulation is complete
        if tf.equal(self.step_counter % self.gradient_accumulation_steps, 0):
            # Average accumulated gradients
            averaged_gradients = [
                acc_grad / self.gradient_accumulation_steps 
                for acc_grad in self.accumulated_gradients
            ]
            
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(averaged_gradients, self.model.trainable_variables)
            )
            
            # Reset accumulated gradients
            for acc_grad in self.accumulated_gradients:
                acc_grad.assign(tf.zeros_like(acc_grad))
    
    @abstractmethod
    def train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Abstract method for custom training step logic."""
        pass
        
    @abstractmethod
    def validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Abstract method for custom validation step logic."""
        pass
        
    def train_epoch(self, train_dataset, val_dataset=None) -> Dict[str, float]:
        """Optimized training epoch using TensorFlow's dataset API."""
        epoch_start_time = time.time()
        
        # Reset metrics
        self.train_loss.reset_state()
        self.val_loss.reset_state()
        for metric in self.metrics:
            if hasattr(metric, 'reset_state'):
                metric.reset_state()
        
        # Training phase with progress bar
        print(f"\nTraining Epoch {self.state.current_epoch + 1}")
        train_progbar = Progbar(
            target=None,  # Will be set automatically
            stateful_metrics=['loss', 'lr']
        )
        
        # Training loop
        step = 0
        for batch in train_dataset:
            step_results = self._distributed_train_step(batch)
            step += 1
            
            # Update progress bar
            values = [
                ('loss', float(step_results['loss'])),
                ('lr', float(self.optimizer.learning_rate))
            ]
            train_progbar.update(step, values=values)
            
        # Validation phase
        val_metrics = {}
        if val_dataset is not None:
            print(f"\nValidation Epoch {self.state.current_epoch + 1}")
            val_progbar = Progbar(target=None)
            
            val_step = 0
            for batch in val_dataset:
                self._distributed_val_step(batch)
                val_step += 1
                val_progbar.update(val_step, values=[('val_loss', float(self.val_loss.result()))])
                
            val_metrics = {'val_loss': float(self.val_loss.result())}
            
        # Compile epoch results
        train_metrics = {
            'loss': float(self.train_loss.result()),
            'learning_rate': float(self.optimizer.learning_rate),
            'epoch_time': time.time() - epoch_start_time
        }
        
        # Add custom metrics
        for metric in self.metrics:
            if hasattr(metric, 'result'):
                metric_name = getattr(metric, 'name', metric.__class__.__name__)
                train_metrics[metric_name] = float(metric.result())
        
        # Combine all metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        
        # Log to TensorBoard
        self._log_to_tensorboard(epoch_metrics, self.state.current_epoch)
        
        return epoch_metrics
    
    def _log_to_tensorboard(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to TensorBoard using TensorFlow's summary API."""
        with self.tensorboard_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=epoch)
            self.tensorboard_writer.flush()
            
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        resume_from_checkpoint: bool = False,
        validation_freq: int = 1
    ):
        """Enhanced training loop using TensorFlow's training utilities."""
        
        # Optimize datasets
        train_dataset = self._optimize_dataset(train_dataset, is_training=True)
        if val_dataset is not None:
            val_dataset = self._optimize_dataset(val_dataset, is_training=False)
        
        # Setup for resuming
        if resume_from_checkpoint:
            self._load_checkpoint()
            
        # Create callback list
        callback_list = tf.keras.callbacks.CallbackList(
            self.callbacks,
            add_history=True,
            add_progbar=False,  # We handle progress bars manually
            model=self.model,
            verbose=1
        )
        
        # Training setup
        callback_list.on_train_begin()
        
        try:
            for epoch in range(self.state.current_epoch, epochs):
                self.state.current_epoch = epoch
                
                # Epoch callbacks
                callback_list.on_epoch_begin(epoch)
                
                # Training epoch
                epoch_metrics = self.train_epoch(
                    train_dataset, 
                    val_dataset if epoch % validation_freq == 0 else None
                )
                
                # Update state
                self._update_training_state(epoch_metrics)
                
                # Epoch end callbacks
                callback_list.on_epoch_end(epoch, epoch_metrics)
                
                # Check for early stopping
                if callback_list.model.stop_training:
                    print("Early stopping triggered")
                    break
                    
        except KeyboardInterrupt:
            print("Training interrupted by user")
            
        finally:
            callback_list.on_train_end()
            self._finalize_training()
            
    def _optimize_dataset(self, dataset: tf.data.Dataset, is_training: bool) -> tf.data.Dataset:
        """Optimize dataset using TensorFlow's performance best practices."""
        # Apply standard optimizations
        if is_training:
            # Shuffle and repeat for training
            dataset = dataset.shuffle(
                buffer_size=getattr(self.config, 'SHUFFLE_BUFFER_SIZE', 1000),
                reshuffle_each_iteration=True
            )
            dataset = dataset.repeat()
            
        # Batch the dataset
        batch_size = getattr(self.config, 'BATCH_SIZE', 16)
        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        
        # Performance optimizations
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Use experimental optimizations
        if getattr(self.config, 'USE_DATASET_OPTIMIZATIONS', True):
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.experimental_optimization.map_parallelization = True
            options.experimental_optimization.parallel_batch = True
            dataset = dataset.with_options(options)
            
        return dataset
    
    def _update_training_state(self, metrics: Dict[str, float]):
        """Update training state based on epoch metrics."""
        current_val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        
        if current_val_loss < self.state.best_val_loss:
            self.state.best_val_loss = current_val_loss
            
        # Update best metric if available
        for key, value in metrics.items():
            if 'accuracy' in key or 'f1' in key or 'precision' in key or 'recall' in key:
                if value > self.state.best_val_metric:
                    self.state.best_val_metric = value
                    
    def _load_checkpoint(self):
        """Load checkpoint using TensorFlow's checkpoint system."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        
        # Use TensorFlow's checkpoint manager
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model,
            step=self.step_counter
        )
        
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(checkpoint_dir),
            max_to_keep=3
        )
        
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("No checkpoint found")
            
    def _finalize_training(self):
        """Finalize training with cleanup and final saves."""
        self.state.is_training = False
        
        # Save final state
        final_results = {
            'training_summary': {
                'task_type': self.task_type,
                'backbone_name': self.backbone_name,
                'total_epochs': self.state.current_epoch + 1,
                'best_val_loss': self.state.best_val_loss,
                'best_val_metric': self.state.best_val_metric
            },
            'final_state': self.state.to_dict()
        }
        
        results_path = Path(self.config.LOGS_DIR) / 'final_results.json'
        save_json(final_results, results_path)
        
        # Close TensorBoard writer
        self.tensorboard_writer.close()
        
        print(f"Training completed. Results saved to {results_path}")
