from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time

import tensorflow as tf
from tensorflow.keras import mixed_precision

from pathlib import Path
from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.models.base_model import BaseModel
from src.training.losses import get_sota_loss_function
from src.training.optimizers import DetectionOptimizer
from src.training.metrics import get_metrics
from src.training.callbacks import get_optimized_callbacks
from src.utils.file_utils import save_json, load_json
from src.utils.plot_utils import plot_training_history




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


class TrainingLogger:
    """Enhanced logger with better performance and features."""
    
    def __init__(self, log_dir: str, log_level: str = 'INFO'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with rotation
        self._setup_logging(log_level)
        
        # Initialize metrics storage with better structure
        self.history = {
            'train': {'loss': [], 'metrics': {}},
            'val': {'loss': [], 'metrics': {}},
            'meta': {'learning_rate': [], 'epoch_time': []}
        }
        
        # Performance monitoring
        self.step_times = []
        self.memory_usage = []
        
    def _setup_logging(self, log_level: str):
        """Setup logging with proper configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # File handler with rotation
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(f'trainer_{id(self)}')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
    def log_epoch(self, epoch: int, logs: Dict[str, float], train_type: str = 'train'):
        """Enhanced epoch logging with categorization."""
        # Categorize metrics
        loss_metrics = {k: v for k, v in logs.items() if 'loss' in k}
        other_metrics = {k: v for k, v in logs.items() if 'loss' not in k and k not in ['learning_rate', 'epoch_time']}
        
        # Update history
        if train_type in ['train', 'val']:
            if logs.get('loss') is not None:
                self.history[train_type]['loss'].append(logs['loss'])
            
            for key, value in other_metrics.items():
                if key not in self.history[train_type]['metrics']:
                    self.history[train_type]['metrics'][key] = []
                self.history[train_type]['metrics'][key].append(value)
        
        # Update meta information
        if 'learning_rate' in logs:
            self.history['meta']['learning_rate'].append(logs['learning_rate'])
        if 'epoch_time' in logs:
            self.history['meta']['epoch_time'].append(logs['epoch_time'])
        
        # Enhanced console logging
        self._log_formatted_epoch(epoch, logs, train_type)
        
    def _log_formatted_epoch(self, epoch: int, logs: Dict[str, float], train_type: str):
        """Format and log epoch information."""
        log_msg = f"Epoch {epoch + 1:3d} [{train_type.upper()}]: "
        
        # Primary metrics first
        primary_metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
        for metric in primary_metrics:
            if metric in logs:
                log_msg += f"{metric}: {logs[metric]:.4f} | "
        
        # Other metrics
        other_metrics = {k: v for k, v in logs.items() if k not in primary_metrics + ['learning_rate', 'epoch_time']}
        if other_metrics:
            log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in other_metrics.items()])
        
        # Performance info
        if 'epoch_time' in logs:
            log_msg += f" | Time: {logs['epoch_time']:.2f}s"
        if 'learning_rate' in logs:
            log_msg += f" | LR: {logs['learning_rate']:.6f}"
        
        self.logger.info(log_msg)
        
    def log_performance(self, step_time: float, memory_usage: Optional[float] = None):
        """Log performance metrics."""
        self.step_times.append(step_time)
        if memory_usage:
            self.memory_usage.append(memory_usage)
            
    def save_history(self, file_path: Optional[str] = None):
        """Save training history with metadata."""
        save_path = file_path or self.log_dir / 'training_history.json'
        
        # Add summary statistics
        history_with_stats = self.history.copy()
        history_with_stats['summary'] = {
            'total_epochs': len(self.history['train']['loss']),
            'best_train_loss': min(self.history['train']['loss']) if self.history['train']['loss'] else None,
            'best_val_loss': min(self.history['val']['loss']) if self.history['val']['loss'] else None,
            'avg_epoch_time': sum(self.history['meta']['epoch_time']) / len(self.history['meta']['epoch_time']) if self.history['meta']['epoch_time'] else None,
            'total_training_time': sum(self.history['meta']['epoch_time']) if self.history['meta']['epoch_time'] else None
        }
        
        save_json(history_with_stats, save_path)
        
    def load_history(self, file_path: Optional[str] = None):
        """Load training history."""
        load_path = file_path or self.log_dir / 'training_history.json'
        if load_path.exists():
            loaded_history = load_json(load_path)
            # Remove summary if present
            if 'summary' in loaded_history:
                del loaded_history['summary']
            self.history = loaded_history



class BaseTrainer(ABC):
    """Abstract base trainer with common functionality."""
    
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
        
        # Setup components
        self._setup_logging()
        self._setup_optimization()
        self._setup_loss_and_metrics()
        self._setup_callbacks()
        
        # Performance optimization
        self._setup_mixed_precision()
        self._setup_memory_optimization()
        
    def _setup_logging(self):
        """Setup logging components."""
        self.logger = TrainingLogger(
            self.config.LOGS_DIR,
            log_level=getattr(self.config, 'LOG_LEVEL', 'INFO')
        )
        
    def _setup_optimization(self):
        """Setup optimizer with enhanced configuration."""

        self.optimizer = DetectionOptimizer(
            model_type='retinanet',
            backbone=self.backbone_name,
        )
        
        # Gradient accumulation setup
        self.gradient_accumulation_steps = getattr(self.config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.accumulated_gradients = []
        
    def _setup_loss_and_metrics(self):
        """Setup loss functions and metrics."""
        self.loss_fn = get_sota_loss_function(
            self.models_config.LOSS_CONFIGS[self.task_type]
        )
        self.metrics = get_metrics(
            self.task_type,
            self.config.NUM_CLASSES_DETECTION
        )
        
    def _setup_callbacks(self):
        """Setup training callbacks."""
        pretrained = self.config.PRETRAINED
        self.callbacks = get_optimized_callbacks(
            task_type=self.task_type,
            backbone_name=self.backbone_name,
            pretrained=pretrained,
            config=self.config,
            model_config=self.models_config
        )
        
    def _setup_mixed_precision(self):
        """Setup mixed precision training if enabled."""
        if getattr(self.config, 'USE_MIXED_PRECISION', False):
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            self.logger.logger.info("Mixed precision training enabled")
            
    def _setup_memory_optimization(self):
        """Setup memory optimization strategies."""
        # Enable memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                self.logger.logger.warning(f"Memory growth setting failed: {e}")
                
    @abstractmethod
    def train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Abstract method for training step."""
        pass
        
    @abstractmethod
    def validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Abstract method for validation step."""
        pass
        
    def train_epoch(self, train_dataset, val_dataset=None) -> Dict[str, float]:
        """Enhanced training epoch with better error handling and monitoring."""
        epoch_start_time = time.time()
        
        # Reset metrics
        for metric in self.metrics:
            metric.reset_state()
        
        # Training phase
        train_metrics = self._run_training_phase(train_dataset)
        
        # Validation phase
        val_metrics = {}
        if val_dataset is not None:
            val_metrics = self._run_validation_phase(val_dataset)
            
        # Combine metrics
        epoch_metrics = {
            **train_metrics,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'learning_rate': float(self.optimizer.learning_rate),
            'epoch_time': time.time() - epoch_start_time
        }
        
        # Log epoch results
        self.logger.log_epoch(self.state.current_epoch, epoch_metrics)
        
        return epoch_metrics
        
    def _run_training_phase(self, dataset) -> Dict[str, float]:
        """Run training phase with gradient accumulation."""
        self.state.is_training = True
        
        total_loss = 0.0
        num_batches = 0
        
        # Reset accumulated gradients
        self.accumulated_gradients = []
        
        for batch_idx, batch in enumerate(dataset):
            step_start_time = time.time()
            
            # Training step with gradient accumulation
            step_metrics = self._training_step_with_accumulation(batch, batch_idx)
            
            total_loss += step_metrics['loss']
            num_batches += 1
            
            # Log step performance
            step_time = time.time() - step_start_time
            self.logger.log_performance(step_time)
            
            # Memory cleanup periodically
            if batch_idx % 100 == 0:
                self._cleanup_memory()
                
        # Final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get additional metrics
        metric_results = self._get_metric_results()
        
        return {
            'loss': avg_loss,
            **metric_results
        }
        
    def _training_step_with_accumulation(self, batch, batch_idx: int) -> Dict[str, float]:
        """Training step with gradient accumulation support."""
        with tf.GradientTape() as tape:
            step_results = self.train_step(batch)
            loss = step_results['loss']
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            
        # Calculate gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Accumulate gradients
        if not self.accumulated_gradients:
            self.accumulated_gradients = gradients
        else:
            self.accumulated_gradients = [
                acc_grad + grad for acc_grad, grad in zip(self.accumulated_gradients, gradients)
            ]
            
        # Apply gradients when accumulation is complete
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.apply_gradients(zip(self.accumulated_gradients, self.model.trainable_variables))
            self.accumulated_gradients = []
            
        return {'loss': float(loss)}
        
    def _run_validation_phase(self, dataset) -> Dict[str, float]:
        """Run validation phase."""
        self.state.is_training = False
        
        total_loss = 0.0
        num_batches = 0
        
        # Reset validation metrics
        val_metrics = get_metrics(self.task_type, self.models_config.NUM_CLASSES[self.task_type])
        
        for batch in dataset:
            step_results = self.validation_step(batch)
            total_loss += step_results['loss']
            num_batches += 1
            
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Get validation metrics
        val_metric_results = val_metrics.result() if hasattr(val_metrics, 'result') else {}
        
        return {
            'loss': avg_loss,
            **val_metric_results
        }
        
    def _get_metric_results(self) -> Dict[str, float]:
        """Get current metric results."""
        if hasattr(self.metrics, 'result'):
            results = self.metrics.result()
            return {k: float(v) for k, v in results.items()}
        return {}
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM issues."""
        try:
            import gc
            gc.collect()
            if tf.config.experimental.list_physical_devices('GPU'):
                tf.keras.backend.clear_session()
        except Exception as e:
            self.logger.logger.debug(f"Memory cleanup warning: {e}")
            
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        resume_from_checkpoint: bool = False,
        validation_freq: int = 1
    ):
        """Enhanced training loop with better error handling."""
        # Setup
        if resume_from_checkpoint:
            self._load_checkpoint()
            
        self._save_config()
        
        start_time = time.time()
        
        try:
            self._run_training_loop(train_dataset, val_dataset, epochs, validation_freq)
            
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
            
        except Exception as e:
            self.logger.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise
            
        finally:
            self._finalize_training(start_time)
            
    def _run_training_loop(self, train_dataset, val_dataset, epochs: int, validation_freq: int):
        """Main training loop."""
        for epoch in range(self.state.current_epoch, epochs):
            self.state.current_epoch = epoch
            
            # Training epoch
            epoch_metrics = self.train_epoch(train_dataset, val_dataset if epoch % validation_freq == 0 else None)
            
            # Execute callbacks
            self._execute_callbacks(epoch, epoch_metrics)
            
            # Update best metrics and save checkpoint
            self._update_best_metrics(epoch_metrics)
            
            # Early stopping check
            if self._should_stop_early():
                self.logger.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
                
    def _execute_callbacks(self, epoch: int, logs: Dict[str, float]):
        """Execute all callbacks safely."""
        for callback in self.callbacks:
            try:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, logs)
            except Exception as e:
                self.logger.logger.warning(f"Callback {callback.__class__.__name__} failed: {e}")
                
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics and save checkpoint if improved."""
        current_val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        
        if current_val_loss < self.state.best_val_loss:
            self.state.best_val_loss = current_val_loss
            self._save_checkpoint(is_best=True)
            
        # Regular checkpoint save
        if self.state.current_epoch % getattr(self.config, 'CHECKPOINT_FREQ', 10) == 0:
            self._save_checkpoint(is_best=False)
            
    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered."""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop') and callback.should_stop:
                return True
        return False
        
    def _finalize_training(self, start_time: float):
        """Finalize training process."""
        self.state.training_time = time.time() - start_time
        self.state.is_training = False
        
        # Save final results
        self._save_final_results()
        
        # Plot training history
        self._plot_training_history()
        
        self.logger.logger.info(f"Training completed in {self.state.training_time:.2f} seconds")
        
    def _save_config(self):
        """Save training configuration."""
        config_data = {
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config),
            'model_config': self.models_config.__dict__ if hasattr(self.models_config, '__dict__') else str(self.models_config),
            'task_type': self.task_type,
            'backbone_name': self.backbone_name
        }
        
        config_path = Path(self.config.LOGS_DIR) / 'config.json'
        save_json(config_data, config_path)
        
    def _save_checkpoint(self, is_best: bool = False):
        """Enhanced checkpoint saving."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training state
        checkpoint_data = {
            'state': self.state.to_dict(),
            'optimizer_config': self.optimizer.get_config(),
            'history': self.logger.history,
            'epoch': self.state.current_epoch
        }
        
        # Save checkpoint metadata
        checkpoint_path = checkpoint_dir / ('best_checkpoint.json' if is_best else 'latest_checkpoint.json')
        save_json(checkpoint_data, checkpoint_path)
        
        # Save model weights
        weights_path = checkpoint_dir / ('best_model_weights.h5' if is_best else f'model_weights_epoch_{self.state.current_epoch}.h5')
        self.model.save_weights(str(weights_path))
        
        self.logger.logger.info(f"Checkpoint saved: {'best' if is_best else 'latest'}")
        
    def _load_checkpoint(self):
        """Enhanced checkpoint loading."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        
        # Try to load best checkpoint first, then latest
        for checkpoint_name in ['best_checkpoint.json', 'latest_checkpoint.json']:
            checkpoint_path = checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                checkpoint_data = load_json(checkpoint_path)
                
                # Restore state
                self.state = TrainingState.from_dict(checkpoint_data['state'])
                self.logger.history = checkpoint_data.get('history', self.logger.history)
                
                # Load model weights
                weights_name = checkpoint_name.replace('checkpoint.json', 'model_weights.h5')
                weights_path = checkpoint_dir / weights_name
                if weights_path.exists():
                    self.model.load_weights(str(weights_path))
                    
                self.logger.logger.info(f"Resumed from {checkpoint_name} at epoch {self.state.current_epoch}")
                return
                
        self.logger.logger.warning("No checkpoint found to resume from")
        
    def _save_final_results(self):
        """Save comprehensive final results."""
        results = {
            'training_summary': {
                'task_type': self.task_type,
                'backbone_name': self.backbone_name,
                'total_epochs': self.state.current_epoch + 1,
                'total_training_time': self.state.training_time,
                'best_val_loss': self.state.best_val_loss,
                'best_val_metric': self.state.best_val_metric
            },
            'final_state': self.state.to_dict(),
            'config_summary': {
                'optimizer': self.optimizer.__class__.__name__,
                'loss_function': self.loss_fn.__class__.__name__ if hasattr(self.loss_fn, '__class__') else str(self.loss_fn),
                'mixed_precision': getattr(self.config, 'USE_MIXED_PRECISION', False),
                'gradient_accumulation_steps': self.gradient_accumulation_steps
            }
        }
        
        results_path = Path(self.config.LOGS_DIR) / 'final_results.json'
        save_json(results, results_path)
        
        # Save training history
        self.logger.save_history()
        
    def _plot_training_history(self):
        """Enhanced plotting of training history."""
        if len(self.logger.history['train']['loss']) > 0:
            plot_path = Path(self.config.LOGS_DIR) / 'training_history.png'
            try:
                plot_training_history(self.logger.history, str(plot_path))
            except Exception as e:
                self.logger.logger.warning(f"Failed to plot training history: {e}")

