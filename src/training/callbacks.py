"""
Custom callbacks for training pipeline
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import json

class EarlyStopping(Callback):
    """Custom early stopping callback with more control"""
    
    def __init__(
        self,
        monitor='val_loss',
        patience=7,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=1
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if 'loss' in self.monitor:
            self.best = np.Inf
        else:
            self.best = -np.Inf
            
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if 'loss' in self.monitor:
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
        else:
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")


class LearningRateScheduler(Callback):
    """Custom learning rate scheduler"""
    
    def __init__(self, schedule_type='cosine', initial_lr=0.001, 
                 min_lr=1e-6, warmup_epochs=5):
        super().__init__()
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.schedule_type == 'cosine':
            lr = self._cosine_schedule(epoch)
        elif self.schedule_type == 'step':
            lr = self._step_schedule(epoch)
        elif self.schedule_type == 'exponential':
            lr = self._exponential_schedule(epoch)
        else:
            lr = self.initial_lr
            
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
    def _cosine_schedule(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_lr * epoch / self.warmup_epochs
        
        total_epochs = self.params['epochs']
        cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
    def _step_schedule(self, epoch):
        if epoch < 30:
            return self.initial_lr
        elif epoch < 60:
            return self.initial_lr * 0.1
        else:
            return self.initial_lr * 0.01
            
    def _exponential_schedule(self, epoch):
        return self.initial_lr * (0.95 ** epoch)


class ModelCheckpoint(Callback):
    """Enhanced model checkpoint callback"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True,
                 save_weights_only=False, mode='auto', save_freq='epoch'):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        
        if mode == 'auto':
            if 'loss' in self.monitor:
                self.mode = 'min'
            else:
                self.mode = 'max'
                
        if self.mode == 'min':
            self.best = np.Inf
        else:
            self.best = -np.Inf
            
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if not self.save_best_only:
            self._save_model(epoch)
        else:
            if self.mode == 'min' and current < self.best:
                self.best = current
                self._save_model(epoch)
            elif self.mode == 'max' and current > self.best:
                self.best = current
                self._save_model(epoch)
                
    def _save_model(self, epoch):
        filepath = self.filepath.format(epoch=epoch + 1)
        if self.save_weights_only:
            self.model.save_weights(filepath)
        else:
            self.model.save(filepath)


class ProgressLogger(Callback):
    """Log training progress to file"""
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_data = {'epoch': epoch + 1}
        epoch_data.update(logs)
        self.history.append(epoch_data)
        
        # Save to JSON
        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
            
        # Save plots
        self._save_plots()
        
    def _save_plots(self):
        if len(self.history) < 2:
            return
            
        epochs = [h['epoch'] for h in self.history]
        
        # Loss plots
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        if 'loss' in self.history[0]:
            train_loss = [h.get('loss', 0) for h in self.history]
            plt.plot(epochs, train_loss, label='Training Loss')
        if 'val_loss' in self.history[0]:
            val_loss = [h.get('val_loss', 0) for h in self.history]
            plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Metrics plots
        plt.subplot(1, 2, 2)
        for key in self.history[0].keys():
            if 'acc' in key.lower() or 'iou' in key.lower():
                values = [h.get(key, 0) for h in self.history]
                plt.plot(epochs, values, label=key.replace('_', ' ').title())
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'), dpi=150)
        plt.close()


class MultiTaskMetricsLogger(Callback):
    """Log metrics for multitask learning"""
    
    def __init__(self, log_dir, task_names=['detection', 'segmentation']):
        super().__init__()
        self.log_dir = log_dir
        self.task_names = task_names
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = {task: [] for task in task_names}
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Separate metrics by task
        for task in self.task_names:
            task_metrics = {'epoch': epoch + 1}
            for key, value in logs.items():
                if task in key:
                    task_metrics[key] = value
            self.metrics_history[task].append(task_metrics)
            
        # Save task-specific histories
        for task in self.task_names:
            with open(os.path.join(self.log_dir, f'{task}_history.json'), 'w') as f:
                json.dump(self.metrics_history[task], f, indent=2)
                
        self._save_task_plots()
        
    def _save_task_plots(self):
        if len(self.metrics_history[self.task_names[0]]) < 2:
            return
            
        fig, axes = plt.subplots(len(self.task_names), 2, figsize=(15, 5 * len(self.task_names)))
        if len(self.task_names) == 1:
            axes = axes.reshape(1, -1)
            
        for i, task in enumerate(self.task_names):
            history = self.metrics_history[task]
            epochs = [h['epoch'] for h in history]
            
            # Loss plot
            ax1 = axes[i, 0]
            for key in history[0].keys():
                if 'loss' in key:
                    values = [h.get(key, 0) for h in history]
                    ax1.plot(epochs, values, label=key.replace('_', ' ').title())
            ax1.set_title(f'{task.title()} Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Metrics plot
            ax2 = axes[i, 1]
            for key in history[0].keys():
                if 'loss' not in key and 'epoch' not in key:
                    values = [h.get(key, 0) for h in history]
                    ax2.plot(epochs, values, label=key.replace('_', ' ').title())
            ax2.set_title(f'{task.title()} Metrics')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Metric Value')
            ax2.legend()
            ax2.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'multitask_curves.png'), dpi=150)
        plt.close()


class GradientMonitor(Callback):
    """Monitor gradient norms during training"""
    
    def __init__(self, log_frequency=10):
        super().__init__()
        self.log_frequency = log_frequency
        self.gradient_norms = []
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            # Calculate gradient norms
            gradients = []
            for layer in self.model.layers:
                if hasattr(layer, 'trainable_weights') and layer.trainable_weights:
                    with tf.GradientTape() as tape:
                        # This is a simplified version - in practice you'd need
                        # to compute gradients properly
                        pass
                        
    def on_epoch_end(self, epoch, logs=None):
        if self.gradient_norms:
            avg_grad_norm = np.mean(self.gradient_norms)
            print(f"Average gradient norm: {avg_grad_norm:.6f}")
            self.gradient_norms = []


def get_callbacks(config, model_name):
    """Get standard callbacks for training"""
    callbacks = []
    
    # Early stopping
    if config.get('early_stopping', True):
        callbacks.append(EarlyStopping(
            monitor=config.get('early_stopping_monitor', 'val_loss'),
            patience=config.get('early_stopping_patience', 10),
            restore_best_weights=True
        ))
    
    # Learning rate scheduler
    if config.get('lr_scheduler', True):
        callbacks.append(LearningRateScheduler(
            schedule_type=config.get('lr_schedule_type', 'cosine'),
            initial_lr=config.get('learning_rate', 0.001)
        ))
    
    # Model checkpoint
    checkpoint_dir = os.path.join(config['checkpoint_dir'], model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True
    ))
    
    # Progress logger
    log_dir = os.path.join(config['log_dir'], model_name)
    callbacks.append(ProgressLogger(log_dir))
    
    # TensorBoard
    if config.get('use_tensorboard', True):
        tb_log_dir = os.path.join(config['tensorboard_dir'], model_name)
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
    
    return callbacks

