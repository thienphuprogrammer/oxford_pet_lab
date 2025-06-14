"""
Unified training interface for all tasks.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback

from src.config.config import Config
from src.training.losses import DetectionLoss, SegmentationLoss
from src.training.metrics import DetectionMetrics, SegmentationMetrics
from src.training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.training.optimizers import get_optimizer_config
from src.utils.file_utils import save_json, load_json
from src.utils.plot_utils import plot_training_history


class TrainingLogger:
    """Logger for training progress and metrics."""
    
    def __init__(self, log_dir: str):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics storage
        self.history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def log_epoch(self, epoch: int, logs: Dict[str, float]):
        """Log metrics for an epoch."""
        # Update history
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Log to console
        log_msg = f"Epoch {epoch + 1}: "
        log_msg += ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        self.logger.info(log_msg)
        
    def save_history(self):
        """Save training history to file."""
        save_json(self.history, self.log_dir / 'training_history.json')
        
    def load_history(self):
        """Load training history from file."""
        history_path = self.log_dir / 'training_history.json'
        if history_path.exists():
            self.history = load_json(history_path)


class Trainer:
    """Unified trainer for all tasks."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        task_type: str,
        log_dir: str,
        checkpoint_dir: str,
        config: Config = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            task_type: Type of task ('detection', 'segmentation', 'multitask')
            config: Training configuration
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.task_type = task_type
        self.config = config or Config()
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(log_dir)
        
        # Setup optimizer
        self.optimizer = get_optimizer_config(task_type, config)
        
        # Setup loss functions
        self._setup_loss_functions()
        
        # Setup metrics
        self._setup_metrics()
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_time = 0
        
    def _setup_loss_functions(self):
        """Setup loss functions based on task type."""
        if self.task_type == 'detection':
            self.loss_fn = DetectionLoss(
                bbox_loss_weight=self.config.get('bbox_loss_weight', 1.0),
                cls_loss_weight=self.config.get('cls_loss_weight', 1.0),
                iou_loss_weight=self.config.get('iou_loss_weight', 1.0)
            )
        elif self.task_type == 'segmentation':
            self.loss_fn = SegmentationLoss(
                ce_weight=self.config.get('ce_weight', 1.0),
                dice_weight=self.config.get('dice_weight', 1.0),
                focal_weight=self.config.get('focal_weight', 0.0)
            )
        elif self.task_type == 'multitask':
            self.detection_loss = DetectionLoss(
                bbox_loss_weight=self.config.get('bbox_loss_weight', 1.0),
                cls_loss_weight=self.config.get('cls_loss_weight', 1.0),
                iou_loss_weight=self.config.get('iou_loss_weight', 1.0)
            )
            self.segmentation_loss = SegmentationLoss(
                ce_weight=self.config.get('ce_weight', 1.0),
                dice_weight=self.config.get('dice_weight', 1.0),
                focal_weight=self.config.get('focal_weight', 0.0)
            )
            self.task_weights = {
                'detection': self.config.get('detection_weight', 1.0),
                'segmentation': self.config.get('segmentation_weight', 1.0)
            }
    
    def _setup_metrics(self):
        """Setup metrics based on task type."""
        if self.task_type == 'detection':
            self.metrics = DetectionMetrics()
        elif self.task_type == 'segmentation':
            self.metrics = SegmentationMetrics()
        elif self.task_type == 'multitask':
            self.detection_metrics = DetectionMetrics()
            self.segmentation_metrics = SegmentationMetrics()
    
    def _setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping', False):
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('patience', 10),
                restore_best_weights=True
            ))
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / 'best_model.h5'
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
        
        # Learning rate scheduler
        if 'lr_schedule' in self.config:
            callbacks.append(LearningRateScheduler(
                schedule=self.optimizer.learning_rate,
                verbose=1
            ))
        
        # TensorBoard
        if self.config.get('use_tensorboard', False):
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir / 'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            ))
        
        return callbacks
    
    @tf.function
    def train_step(self, batch):
        """Single training step."""
        if self.task_type == 'detection':
            return self._detection_train_step(batch)
        elif self.task_type == 'segmentation':
            return self._segmentation_train_step(batch)
        elif self.task_type == 'multitask':
            return self._multitask_train_step(batch)
    
    @tf.function
    def _detection_train_step(self, batch):
        """Training step for object detection."""
        images, targets = batch
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.metrics.update_state(targets, predictions)
        
        return loss
    
    @tf.function
    def _segmentation_train_step(self, batch):
        """Training step for semantic segmentation."""
        images, masks = batch
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(masks, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.metrics.update_state(masks, predictions)
        
        return loss
    
    @tf.function
    def _multitask_train_step(self, batch):
        """Training step for multitask learning."""
        images, targets = batch
        detection_targets, segmentation_targets = targets
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            detection_preds, segmentation_preds = predictions
            
            # Calculate losses
            det_loss = self.detection_loss(detection_targets, detection_preds)
            seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
            
            # Weighted combination
            total_loss = (self.task_weights['detection'] * det_loss + 
                         self.task_weights['segmentation'] * seg_loss)
        
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.detection_metrics.update_state(detection_targets, detection_preds)
        self.segmentation_metrics.update_state(segmentation_targets, segmentation_preds)
        
        return total_loss, det_loss, seg_loss
    
    @tf.function
    def validation_step(self, batch):
        """Single validation step."""
        if self.task_type == 'detection':
            return self._detection_validation_step(batch)
        elif self.task_type == 'segmentation':
            return self._segmentation_validation_step(batch)
        elif self.task_type == 'multitask':
            return self._multitask_validation_step(batch)
    
    @tf.function
    def _detection_validation_step(self, batch):
        """Validation step for object detection."""
        images, targets = batch
        
        predictions = self.model(images, training=False)
        loss = self.loss_fn(targets, predictions)
        
        return loss
    
    @tf.function
    def _segmentation_validation_step(self, batch):
        """Validation step for semantic segmentation."""
        images, masks = batch
        
        predictions = self.model(images, training=False)
        loss = self.loss_fn(masks, predictions)
        
        return loss
    
    @tf.function
    def _multitask_validation_step(self, batch):
        """Validation step for multitask learning."""
        images, targets = batch
        detection_targets, segmentation_targets = targets
        
        predictions = self.model(images, training=False)
        detection_preds, segmentation_preds = predictions
        
        # Calculate losses
        det_loss = self.detection_loss(detection_targets, detection_preds)
        seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
        
        # Weighted combination
        total_loss = (self.task_weights['detection'] * det_loss + 
                     self.task_weights['segmentation'] * seg_loss)
        
        return total_loss, det_loss, seg_loss
    
    def train_epoch(self, train_dataset, val_dataset=None):
        """Train for one epoch."""
        epoch_start_time = time.time()
        
        # Reset metrics
        if self.task_type == 'multitask':
            self.detection_metrics.reset_state()
            self.segmentation_metrics.reset_state()
        else:
            self.metrics.reset_state()
        
        # Training loop
        train_loss = 0
        num_batches = 0
        
        for batch in train_dataset:
            loss = self.train_step(batch)
            
            if self.task_type == 'multitask':
                total_loss, det_loss, seg_loss = loss
                train_loss += total_loss
            else:
                train_loss += loss
            
            num_batches += 1
        
        # Calculate average training loss
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        
        # Validation loop
        val_loss = 0
        val_num_batches = 0
        
        if val_dataset is not None:
            for batch in val_dataset:
                loss = self.validation_step(batch)
                
                if self.task_type == 'multitask':
                    total_loss, det_loss, seg_loss = loss
                    val_loss += total_loss
                else:
                    val_loss += loss
                
                val_num_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_num_batches if val_num_batches > 0 else 0
        
        # Get metrics
        logs = {
            'loss': float(avg_train_loss),
            'learning_rate': float(self.optimizer.learning_rate)
        }
        
        if val_dataset is not None:
            logs['val_loss'] = float(avg_val_loss)
        
        # Add task-specific metrics
        if self.task_type == 'detection':
            logs.update(self.metrics.result())
        elif self.task_type == 'segmentation':
            logs.update(self.metrics.result())
        elif self.task_type == 'multitask':
            det_metrics = self.detection_metrics.result()
            seg_metrics = self.segmentation_metrics.result()
            logs.update({f'det_{k}': v for k, v in det_metrics.items()})
            logs.update({f'seg_{k}': v for k, v in seg_metrics.items()})
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        logs['epoch_time'] = epoch_time
        
        self.logger.log_epoch(self.current_epoch, logs)
        
        return logs
    
    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 100,
        resume_from_checkpoint: bool = False
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs to train
            resume_from_checkpoint: Whether to resume from checkpoint
        """
        # Load checkpoint if resuming
        if resume_from_checkpoint:
            self._load_checkpoint()
        
        # Save initial config
        self._save_config()
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                
                # Train epoch
                logs = self.train_epoch(train_dataset, val_dataset)
                
                # Execute callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, logs)
                
                # Save checkpoint
                if logs.get('val_loss', logs['loss']) < self.best_val_loss:
                    self.best_val_loss = logs.get('val_loss', logs['loss'])
                    self._save_checkpoint()
                
                # Check early stopping
                if hasattr(self, 'early_stopping') and self.early_stopping.should_stop:
                    self.logger.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        finally:
            # Calculate total training time
            self.training_time = time.time() - start_time
            
            # Save final results
            self._save_final_results()
            
            # Plot training history
            self._plot_training_history()
    
    def _save_config(self):
        """Save training configuration."""
        config_path = self.log_dir / 'config.json'
        save_json(self.config, config_path)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'training_time': self.training_time,
            'optimizer_state': self.optimizer.get_weights(),
            'history': self.logger.history
        }
        
        checkpoint_path = self.checkpoint_dir / 'checkpoint.json'
        save_json(checkpoint_data, checkpoint_path)
        
        # Save model
        model_path = self.checkpoint_dir / f'model_epoch_{self.current_epoch}.h5'
        self.model.save_weights(str(model_path))
    
    def _load_checkpoint(self):
        """Load training checkpoint."""
        checkpoint_path = self.checkpoint_dir / 'checkpoint.json'
        if checkpoint_path.exists():
            checkpoint_data = load_json(checkpoint_path)
            self.current_epoch = checkpoint_data.get('epoch', 0)
            self.best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            self.training_time = checkpoint_data.get('training_time', 0)
            self.logger.history = checkpoint_data.get('history', self.logger.history)
            
            # Load model weights
            model_path = self.checkpoint_dir / f'model_epoch_{self.current_epoch}.h5'
            if model_path.exists():
                self.model.load_weights(str(model_path))
                self.logger.logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def _save_final_results(self):
        """Save final training results."""
        results = {
            'task_type': self.task_type,
            'total_epochs': self.current_epoch + 1,
            'best_val_loss': self.best_val_loss,
            'total_training_time': self.training_time,
            'final_metrics': self._get_final_metrics(),
            'config': self.config
        }
        
        results_path = self.log_dir / 'final_results.json'
        save_json(results, results_path)
        
        # Save training history
        self.logger.save_history()
    
    def _get_final_metrics(self) -> Dict[str, float]:
        """Get final metrics."""
        if self.task_type == 'detection':
            return self.metrics.result()
        elif self.task_type == 'segmentation':
            return self.metrics.result()
        elif self.task_type == 'multitask':
            det_metrics = self.detection_metrics.result()
            seg_metrics = self.segmentation_metrics.result()
            return {
                **{f'det_{k}': v for k, v in det_metrics.items()},
                **{f'seg_{k}': v for k, v in seg_metrics.items()}
            }
        return {}
    
    def _plot_training_history(self):
        """Plot and save training history."""
        if len(self.logger.history['loss']) > 0:
            plot_path = self.log_dir / 'training_history.png'
            plot_training_history(self.logger.history, str(plot_path))
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Reset metrics
        if self.task_type == 'multitask':
            self.detection_metrics.reset_state()
            self.segmentation_metrics.reset_state()
        else:
            self.metrics.reset_state()
        
        # Evaluation loop
        total_loss = 0
        num_batches = 0
        
        for batch in test_dataset:
            loss = self.validation_step(batch)
            
            if self.task_type == 'multitask':
                total_loss += loss[0]  # Total loss
            else:
                total_loss += loss
            
            num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Get metrics
        results = {'test_loss': float(avg_loss)}
        
        if self.task_type == 'detection':
            results.update(self.metrics.result())
        elif self.task_type == 'segmentation':
            results.update(self.metrics.result())
        elif self.task_type == 'multitask':
            det_metrics = self.detection_metrics.result()
            seg_metrics = self.segmentation_metrics.result()
            results.update({f'det_{k}': v for k, v in det_metrics.items()})
            results.update({f'seg_{k}': v for k, v in seg_metrics.items()})
        
        return results
    
    def predict(self, dataset: tf.data.Dataset) -> List[Any]:
        """
        Generate predictions on dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for batch in dataset:
            if self.task_type in ['detection', 'segmentation']:
                images, _ = batch
            else:  # multitask
                images, _ = batch
            
            preds = self.model(images, training=False)
            predictions.append(preds)
        
        return predictions
    
    def save_model(self, save_path: str):
        """
        Save trained model.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save full model
        self.model.save(str(save_path))
        
        # Save weights only
        weights_path = save_path.parent / f"{save_path.stem}_weights.h5"
        self.model.save_weights(str(weights_path))
        
        self.logger.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load trained model.
        
        Args:
            load_path: Path to load model from
        """
        load_path = Path(load_path)
        
        if load_path.exists():
            self.model = tf.keras.models.load_model(str(load_path))
            self.logger.logger.info(f"Model loaded from {load_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {load_path}")


class MultiTaskTrainer(Trainer):
    """Specialized trainer for multitask learning with advanced features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize task-specific weights that can be learned
        self.task_weights_learnable = self.config.get('learnable_weights', False)
        
        if self.task_weights_learnable:
            # Initialize learnable task weights
            self.task_weight_vars = {
                'detection': tf.Variable(1.0, trainable=True, name='detection_weight'),
                'segmentation': tf.Variable(1.0, trainable=True, name='segmentation_weight')
            }
    
    def _compute_task_weights(self, det_loss, seg_loss):
        """Compute adaptive task weights based on loss magnitudes."""
        if self.task_weights_learnable:
            # Use learnable weights
            det_weight = tf.nn.softplus(self.task_weight_vars['detection'])
            seg_weight = tf.nn.softplus(self.task_weight_vars['segmentation'])
        else:
            # Use uncertainty-based weighting
            det_weight = 1.0 / (2 * tf.square(det_loss) + 1e-8)
            seg_weight = 1.0 / (2 * tf.square(seg_loss) + 1e-8)
        
        return det_weight, seg_weight
    
    @tf.function
    def _multitask_train_step(self, batch):
        """Enhanced training step with adaptive weighting."""
        images, targets = batch
        detection_targets, segmentation_targets = targets
        
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            detection_preds, segmentation_preds = predictions
            
            # Calculate individual losses
            det_loss = self.detection_loss(detection_targets, detection_preds)
            seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
            
            # Compute adaptive weights
            det_weight, seg_weight = self._compute_task_weights(det_loss, seg_loss)
            
            # Weighted combination
            total_loss = det_weight * det_loss + seg_weight * seg_loss
            
            # Add regularization if using learnable weights
            if self.task_weights_learnable:
                reg_loss = 0.01 * (tf.square(det_weight) + tf.square(seg_weight))
                total_loss += reg_loss
        
        # Get all trainable variables
        trainable_vars = self.model.trainable_variables
        if self.task_weights_learnable:
            trainable_vars += list(self.task_weight_vars.values())
        
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.detection_metrics.update_state(detection_targets, detection_preds)
        self.segmentation_metrics.update_state(segmentation_targets, segmentation_preds)
        
        return total_loss, det_loss, seg_loss, det_weight, seg_weight
