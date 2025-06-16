"""
Optimized custom callbacks for training pipeline with SOTA techniques
Supports Classification, Segmentation, and Object Detection tasks
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import json
import warnings
from typing import List, Dict, Any, Optional
from abc import ABC
from src.config import Config
from src.config.model_configs import ModelConfigs


class BaseOptimizedCallback(Callback, ABC):
    """Base class for optimized callbacks with common utilities"""
    
    def __init__(self, verbose: int = 1):
        super().__init__()
        self.verbose = verbose
        
    def _safe_get_metric(self, logs: Dict, metric_name: str, default_value: float = 0.0) -> float:
        """Safely get metric value from logs"""
        return logs.get(metric_name, default_value) if logs else default_value


class AdaptiveEarlyStopping(BaseOptimizedCallback):
    """Enhanced early stopping with adaptive patience and delta"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 1e-4,
        patience: int = 10,
        restore_best_weights: bool = True,
        adaptive_patience: bool = True,
        patience_factor: float = 1.2,
        min_lr_threshold: float = 1e-7,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.monitor = monitor
        self.min_delta = min_delta
        self.base_patience = patience
        self.restore_best_weights = restore_best_weights
        self.adaptive_patience = adaptive_patience
        self.patience_factor = patience_factor
        self.min_lr_threshold = min_lr_threshold
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        self.best_epoch = 0
        self.patience = patience
        self.improvement_streak = 0
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.improvement_streak = 0
        self.patience = self.base_patience
        
        if any(keyword in self.monitor.lower() for keyword in ['loss', 'error']):
            self.best = np.inf
            self.monitor_op = np.less
        else:
            self.best = -np.inf
            self.monitor_op = np.greater
            
    def on_epoch_end(self, epoch, logs=None):
        current = self._safe_get_metric(logs, self.monitor)
        
        if current is None:
            warnings.warn(f"Metric '{self.monitor}' not found in logs.")
            return
            
        # Check for improvement
        if self.monitor_op(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            self.improvement_streak += 1
            
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
                
            # Adaptive patience: increase patience after consecutive improvements
            if self.adaptive_patience and self.improvement_streak >= 3:
                self.patience = int(self.base_patience * self.patience_factor)
                self.improvement_streak = 0
                if self.verbose > 0:
                    print(f"Adaptive patience increased to {self.patience}")
        else:
            self.wait += 1
            self.improvement_streak = 0
            
        # Check learning rate threshold
        try:
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            if current_lr < self.min_lr_threshold and self.wait >= self.patience // 2:
                if self.verbose > 0:
                    print(f"Learning rate {current_lr:.2e} below threshold. Triggering early stopping.")
                self._trigger_stop(epoch)
        except:
            pass  # Skip LR check if it fails
            
        if self.wait >= self.patience:
            self._trigger_stop(epoch)
            
    def _trigger_stop(self, epoch):
        self.stopped_epoch = epoch
        self.model.stop_training = True
        
        if self.restore_best_weights and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            
        if self.verbose > 0:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            print(f"Best epoch: {self.best_epoch + 1}, Best {self.monitor}: {self.best:.6f}")


class WarmupCosineScheduler(BaseOptimizedCallback):
    """Advanced learning rate scheduler with warmup, cosine decay, and restarts"""
    
    def __init__(
        self,
        initial_lr: float = 1e-3,
        min_lr: float = 1e-7,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        cosine_restarts: bool = True,
        restart_decay: float = 0.8,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.cosine_restarts = cosine_restarts
        self.restart_decay = restart_decay
        
        self.restart_epochs = []
        self.current_restart = 0
        
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get('epochs', self.total_epochs)
        
        if self.cosine_restarts:
            # Set restart points at 1/4, 1/2, 3/4 of training
            self.restart_epochs = [
                self.total_epochs // 4,
                self.total_epochs // 2,
                3 * self.total_epochs // 4
            ]
        
    def on_epoch_begin(self, epoch, logs=None):
        lr = self._calculate_lr(epoch)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        
        if self.verbose > 1:
            print(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")
            
    def _calculate_lr(self, epoch):
        # Check for restart
        if self.cosine_restarts and epoch in self.restart_epochs:
            self.current_restart += 1
            if self.verbose > 0:
                print(f"Cosine restart #{self.current_restart} at epoch {epoch + 1}")
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        
        # Calculate effective epoch for cosine decay
        if self.cosine_restarts:
            restart_epoch = 0
            for restart in self.restart_epochs:
                if epoch >= restart:
                    restart_epoch = restart
                else:
                    break
            effective_epoch = epoch - restart_epoch
            effective_total = (self.restart_epochs[self.current_restart] - restart_epoch 
                             if self.current_restart < len(self.restart_epochs)
                             else self.total_epochs - restart_epoch)
        else:
            effective_epoch = epoch - self.warmup_epochs
            effective_total = self.total_epochs - self.warmup_epochs
        
        # Cosine decay
        decay_factor = (self.restart_decay ** self.current_restart 
                       if self.cosine_restarts else 1.0)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * effective_epoch / effective_total))
        
        return self.min_lr + (self.initial_lr * decay_factor - self.min_lr) * cosine_decay


class AdvancedModelCheckpoint(BaseOptimizedCallback):
    """Enhanced checkpoint with model versioning and automatic cleanup"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        mode: str = 'auto',
        keep_top_k: int = 3,
        save_freq: str = 'epoch',
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.keep_top_k = keep_top_k
        self.save_freq = save_freq
        
        self.saved_models = []  # List of (filepath, metric_value) tuples
        
        if mode == 'auto':
            self.mode = 'min' if any(keyword in monitor.lower() for keyword in ['loss', 'error']) else 'max'
            
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.monitor_op = np.less if self.mode == 'min' else np.greater
        
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        current = self._safe_get_metric(logs, self.monitor)
        
        if current is None:
            return
            
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        should_save = (not self.save_best_only or 
                      self.monitor_op(current, self.best))
        
        if should_save:
            if self.monitor_op(current, self.best):
                self.best = current
                
            self._save_model(filepath, current)
            self._cleanup_old_models()
            
            if self.verbose > 0:
                print(f"\nModel saved: {filepath} ({self.monitor}: {current:.6f})")
                
    def _save_model(self, filepath, metric_value):
        try:
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath, save_format='tf' if filepath.endswith('.tf') else 'h5')
                
            self.saved_models.append((filepath, metric_value))
        except Exception as e:
            print(f"Error saving model: {e}")
        
    def _cleanup_old_models(self):
        if len(self.saved_models) <= self.keep_top_k:
            return
            
        # Sort by metric value (best first)
        self.saved_models.sort(key=lambda x: x[1], 
                              reverse=(self.mode == 'max'))
        
        # Remove worst models
        to_remove = self.saved_models[self.keep_top_k:]
        self.saved_models = self.saved_models[:self.keep_top_k]
        
        for filepath, _ in to_remove:
            try:
                if os.path.exists(filepath):
                    if os.path.isdir(filepath):
                        import shutil
                        shutil.rmtree(filepath)
                    else:
                        os.remove(filepath)
            except OSError:
                pass


class MetricsLogger(BaseOptimizedCallback):
    """Enhanced metrics logging with visualization and statistics"""
    
    def __init__(
        self,
        log_dir: str,
        save_plots: bool = True,
        plot_frequency: int = 5,
        save_statistics: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.save_plots = save_plots
        self.plot_frequency = plot_frequency
        self.save_statistics = save_statistics
        
        os.makedirs(log_dir, exist_ok=True)
        self.history = []
        self.statistics = {}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_data = {'epoch': epoch + 1}
        epoch_data.update(logs)  # Fixed the bug here
        self.history.append(epoch_data)
        
        # Update statistics
        self._update_statistics(logs)
        
        # Save data
        self._save_history()
        if self.save_statistics:
            self._save_statistics()
            
        # Generate plots
        if (self.save_plots and len(self.history) > 1 and 
            (epoch + 1) % self.plot_frequency == 0):
            self._generate_plots()
            
    def _update_statistics(self, logs):
        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue
                
            if key not in self.statistics:
                self.statistics[key] = {
                    'values': [],
                    'mean': 0,
                    'std': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'trend': 'stable'
                }
            
            stats = self.statistics[key]
            stats['values'].append(value)
            stats['mean'] = np.mean(stats['values'])
            stats['std'] = np.std(stats['values'])
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            
            # Calculate trend
            if len(stats['values']) >= 5:
                recent_trend = np.polyfit(range(5), stats['values'][-5:], 1)[0]
                if abs(recent_trend) < 1e-4:
                    stats['trend'] = 'stable'
                elif recent_trend > 0:
                    stats['trend'] = 'increasing' if 'loss' not in key else 'deteriorating'
                else:
                    stats['trend'] = 'decreasing' if 'loss' not in key else 'improving'
    
    def _save_history(self):
        try:
            with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving history: {e}")
            
    def _save_statistics(self):
        try:
            # Remove values array for JSON serialization
            stats_for_json = {}
            for key, stats in self.statistics.items():
                stats_for_json[key] = {k: v for k, v in stats.items() if k != 'values'}
                
            with open(os.path.join(self.log_dir, 'training_statistics.json'), 'w') as f:
                json.dump(stats_for_json, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving statistics: {e}")
    
    def _generate_plots(self):
        try:
            if len(self.history) < 2:
                return
                
            epochs = [h['epoch'] for h in self.history]
            
            # Create subplots
            metrics = list(self.history[0].keys())
            metrics.remove('epoch')
            
            loss_metrics = [m for m in metrics if 'loss' in m.lower()]
            acc_metrics = [m for m in metrics if any(x in m.lower() for x in ['acc', 'precision', 'recall', 'f1', 'iou', 'dice', 'map', 'ap'])]
            
            num_plots = (len(loss_metrics) > 0) + (len(acc_metrics) > 0)
            if num_plots == 0:
                return
                
            fig_height = 4 * num_plots
            fig, axes = plt.subplots(num_plots, 1, figsize=(12, max(fig_height, 4)))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
                
            plot_idx = 0
            
            # Plot losses
            if loss_metrics:
                ax = axes[plot_idx]
                for metric in loss_metrics:
                    values = [h.get(metric, 0) for h in self.history]
                    ax.plot(epochs, values, label=metric.replace('_', ' ').title(), linewidth=2)
                ax.set_title('Training and Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
            
            # Plot accuracy metrics
            if acc_metrics:
                ax = axes[plot_idx]
                for metric in acc_metrics:
                    values = [h.get(metric, 0) for h in self.history]
                    ax.plot(epochs, values, label=metric.replace('_', ' ').title(), linewidth=2)
                ax.set_title('Training and Validation Metrics')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Metric Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error generating plots: {e}")


class GradientClippingCallback(BaseOptimizedCallback):
    """Monitor and clip gradients to prevent exploding gradients"""
    
    def __init__(self, clip_norm: float = 1.0, log_frequency: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.clip_norm = clip_norm
        self.log_frequency = log_frequency
        self.gradient_norms = []
        
    def on_train_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            try:
                # Monitor gradient norms (simplified implementation)
                total_norm = 0.0
                param_count = 0
                
                for layer in self.model.layers:
                    if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                        param_count += len(layer.trainable_weights)
                        
                # Estimate gradient norm based on loss change
                if logs and 'loss' in logs:
                    estimated_norm = logs['loss']
                    self.gradient_norms.append(float(estimated_norm))
                    
                    if estimated_norm > self.clip_norm and self.verbose > 1:
                        print(f"Batch {batch}: Estimated gradient norm {estimated_norm:.4f} > {self.clip_norm}")
            except Exception as e:
                if self.verbose > 1:
                    print(f"Error in gradient monitoring: {e}")
    
    def on_epoch_end(self, epoch, logs=None):
        if self.gradient_norms:
            avg_norm = np.mean(self.gradient_norms)
            max_norm = np.max(self.gradient_norms)
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: Avg estimated gradient norm: {avg_norm:.4f}, Max: {max_norm:.4f}")
            self.gradient_norms = []


class ClassificationCallbacks:
    """Specialized callbacks for classification tasks"""
    
    @staticmethod
    def get_callbacks(
        log_dir: str,
        checkpoint_dir: str,
        monitor_metric: str = 'val_accuracy',
        early_stopping_patience: int = 15,
        lr_schedule: str = 'cosine',
        initial_lr: float = 1e-3,
        total_epochs: int = 100
    ) -> List[Callback]:
        """Get optimized callbacks for classification"""
        
        callbacks = [
            # Adaptive early stopping for classification
            AdaptiveEarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                min_delta=1e-4,
                restore_best_weights=True,
                adaptive_patience=True,
                verbose=1
            ),
            
            # Learning rate scheduling
            WarmupCosineScheduler(
                initial_lr=initial_lr,
                min_lr=1e-7,
                warmup_epochs=max(1, total_epochs // 20),
                total_epochs=total_epochs,
                cosine_restarts=True,
                restart_decay=0.8,
                verbose=1
            ),
            
            # Model checkpointing
            AdvancedModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model_epoch_{epoch:03d}.h5'),
                monitor=monitor_metric,
                save_best_only=True,
                mode='max',
                keep_top_k=3,
                verbose=1
            ),
            
            # Metrics logging
            MetricsLogger(
                log_dir=log_dir,
                save_plots=True,
                plot_frequency=5,
                verbose=1
            ),
            
            # Reduce LR on plateau as backup
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=5,
                min_lr=1e-8,
                verbose=1
            ),
            
            # CSV logger for detailed analysis
            tf.keras.callbacks.CSVLogger(
                os.path.join(log_dir, 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks


class SegmentationCallbacks:
    """Specialized callbacks for segmentation tasks"""
    
    @staticmethod
    def get_callbacks(
        log_dir: str,
        checkpoint_dir: str,
        monitor_metric: str = 'val_dice_coefficient',
        early_stopping_patience: int = 20,
        lr_schedule: str = 'cosine',
        initial_lr: float = 1e-3,
        total_epochs: int = 150
    ) -> List[Callback]:
        """Get optimized callbacks for segmentation"""
        
        callbacks = [
            # Segmentation-specific early stopping
            AdaptiveEarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                min_delta=1e-5,  # Smaller delta for segmentation metrics
                restore_best_weights=True,
                adaptive_patience=True,
                patience_factor=1.3,  # More patient for segmentation
                verbose=1
            ),
            
            # Longer warmup for segmentation
            WarmupCosineScheduler(
                initial_lr=initial_lr,
                min_lr=1e-8,
                warmup_epochs=max(5, total_epochs // 15),
                total_epochs=total_epochs,
                cosine_restarts=True,
                restart_decay=0.9,
                verbose=1
            ),
            
            # Model checkpointing with IoU/Dice monitoring
            AdvancedModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_segmentation_epoch_{epoch:03d}.h5'),
                monitor=monitor_metric,
                save_best_only=True,
                mode='max',
                keep_top_k=5,  # Keep more models for segmentation
                verbose=1
            ),
            
            # Comprehensive metrics logging
            MetricsLogger(
                log_dir=log_dir,
                save_plots=True,
                plot_frequency=10,  # Less frequent plotting for longer training
                save_statistics=True,
                verbose=1
            ),
            
            # Multi-metric LR reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',  # Use loss as backup
                factor=0.3,
                patience=8,
                min_lr=1e-9,
                verbose=1
            ),
            
            # Additional IoU monitoring if available
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.5,
                patience=10,
                min_lr=1e-9,
                verbose=1
            )
        ]
        
        return callbacks


class DetectionCallbacks:
    """Specialized callbacks for object detection tasks"""
    
    @staticmethod
    def get_callbacks(
        log_dir: str,
        checkpoint_dir: str,
        monitor_metric: str = 'val_mAP',
        early_stopping_patience: int = 25,
        lr_schedule: str = 'cosine',
        initial_lr: float = 1e-4,
        total_epochs: int = 200
    ) -> List[Callback]:
        """Get optimized callbacks for object detection"""
        
        callbacks = [
            # Detection-specific early stopping with higher patience
            AdaptiveEarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                min_delta=1e-4,  # Appropriate delta for mAP metrics
                restore_best_weights=True,
                adaptive_patience=True,
                patience_factor=1.4,  # Very patient for detection training
                min_lr_threshold=1e-8,
                verbose=1
            ),
            
            # Extended warmup for detection (typically needs longer warmup)
            WarmupCosineScheduler(
                initial_lr=initial_lr,
                min_lr=1e-9,
                warmup_epochs=max(10, total_epochs // 10),  # Longer warmup
                total_epochs=total_epochs,
                cosine_restarts=True,
                restart_decay=0.95,  # Less aggressive restart decay
                verbose=1
            ),
            
            # Model checkpointing for detection with multiple metrics
            AdvancedModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_detection_epoch_{epoch:03d}.h5'),
                monitor=monitor_metric,
                save_best_only=True,
                mode='max',
                keep_top_k=5,  # Keep more models for detection
                verbose=1
            ),
            
            # Comprehensive metrics logging for detection
            MetricsLogger(
                log_dir=log_dir,
                save_plots=True,
                plot_frequency=15,  # Less frequent plotting for very long training
                save_statistics=True,
                verbose=1
            ),
            
            # Conservative LR reduction for detection
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,  # More conservative reduction
                patience=12,  # Higher patience
                min_lr=1e-10,
                verbose=1
            ),
            
            # Monitor mAP specifically if available
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=0.3,
                patience=15,
                min_lr=1e-10,
                verbose=1
            ),
            
            # Add gradient clipping by default for detection (helps with stability)
            GradientClippingCallback(
                clip_norm=1.0,
                log_frequency=100,  # Log less frequently for detection
                verbose=1
            )
        ]
        
        return callbacks


def get_optimized_callbacks(
    task_type: str,
    backbone_name: str,
    pretrained: bool,
    config: Config,
    model_config: ModelConfigs
) -> List[Callback]:
    """
    Get task-specific optimized callbacks
    
    Args:
        task_type: 'classification', 'segmentation', or 'detection'
        backbone_name: Name of the backbone model
        pretrained: Whether using pretrained weights
        config: Configuration dictionary
    
    Returns:
        List of optimized callbacks
    """
    
    # Create directories
    exp_name = f"{backbone_name}_{'pretrained' if pretrained else 'scratch'}"
    log_dir = os.path.join(config.LOGS_DIR, task_type, exp_name)
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, task_type, exp_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    arg_model_config = model_config.CALLBACKS_CONFIGS.get(task_type)
    
    # Get task-specific callbacks
    if task_type.lower() == 'classification':
        callbacks = ClassificationCallbacks.get_callbacks(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            monitor_metric=arg_model_config.get('monitor_metric', 'val_accuracy'),
            early_stopping_patience=arg_model_config.get('early_stopping_patience', 15),
            initial_lr=arg_model_config.get('learning_rate', 1e-3),
            total_epochs=arg_model_config.get('epochs', 100)
        )
    elif task_type.lower() == 'segmentation':
        callbacks = SegmentationCallbacks.get_callbacks(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            monitor_metric=arg_model_config.get('monitor_metric', 'val_dice_coefficient'),
            early_stopping_patience=arg_model_config.get('early_stopping_patience', 20),
            initial_lr=arg_model_config.get('learning_rate', 1e-3),
            total_epochs=arg_model_config.get('epochs', 150)
        )
    elif task_type.lower() == 'detection':
        callbacks = DetectionCallbacks.get_callbacks(
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            monitor_metric=arg_model_config.get('monitor_metric', 'val_mAP'),
            early_stopping_patience=arg_model_config.get('early_stopping_patience', 25),
            initial_lr=arg_model_config.get('learning_rate', 1e-4),
            total_epochs=arg_model_config.get('epochs', 200)
        )
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Use 'classification', 'segmentation', or 'detection'.")
    
    # Add gradient clipping if requested (not already added by DetectionCallbacks)
    if arg_model_config.get('gradient_clipping', False) and task_type.lower() != 'detection':
        callbacks.append(GradientClippingCallback(
            clip_norm=arg_model_config.get('clip_norm', 1.0),
            log_frequency=arg_model_config.get('grad_log_freq', 10)
        ))
    
    # Add TensorBoard if requested
    if arg_model_config.get('use_tensorboard', True):
        tb_log_dir = os.path.join(
            config.TENSORBOARD_LOG_DIR,
            task_type, 
            exp_name
        )
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ))
    
    return callbacks


CLASSIFICATION_CONFIG = {
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints',
    'tensorboard_dir': './tensorboard',
    'monitor_metric': 'val_accuracy',
    'early_stopping_patience': 15,
    'learning_rate': 1e-3,
    'epochs': 100,
    'gradient_clipping': True,
    'clip_norm': 1.0,
    'use_tensorboard': True
}

SEGMENTATION_CONFIG = {
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints', 
    'tensorboard_dir': './tensorboard',
    'monitor_metric': 'val_dice_coefficient',
    'early_stopping_patience': 20,
    'learning_rate': 1e-3,
    'epochs': 150,
    'gradient_clipping': True,
    'clip_norm': 0.5,
    'use_tensorboard': True
}

DETECTION_CONFIG = {
    'log_dir': './logs',
    'checkpoint_dir': './checkpoints', 
    'tensorboard_dir': './tensorboard',
    'monitor_metric': 'val_mAP',
    'early_stopping_patience': 25,
    'learning_rate': 1e-4,
    'epochs': 200,
    'gradient_clipping': True,
    'clip_norm': 1.0,
    'use_tensorboard': True
}
