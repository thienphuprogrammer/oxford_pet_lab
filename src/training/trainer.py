from typing import Dict
import tensorflow as tf

from src.config.config import Config
from src.config.model_configs import ModelConfigs
from src.models.base_model import BaseModel
from src.training.losses import get_loss
from src.training.optimizers import DetectionOptimizer
from src.training.metrics import get_metrics
from src.training.callbacks import get_optimized_callbacks

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
        det_opt = DetectionOptimizer('retinanet', backbone_name)
        self.optimizer = det_opt.create_optimizer(
            optimizer_type=getattr(self.config, 'OPTIMIZER_TYPE', 'sgd'),
            custom_lr=getattr(self.config, 'LEARNING_RATE', None),
            batch_size=getattr(self.config, 'BATCH_SIZE', 16),
            total_epochs=getattr(self.config, 'EPOCHS', 300),
        )
        
        # Loss and metrics
        self.loss_fn = get_sota_loss_function(
            self.models_config.LOSS_CONFIGS[self.task_type]
        )
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
                tf.config.experimental.set_memory_growth(gpu, True)
        
        if getattr(self.config, 'USE_XLA', False):
            tf.config.optimizer.set_jit(True)
    
    def _get_callbacks(self, backbone_name: str):
        """Get optimized callbacks."""
        log_dir = Path(self.config.LOGS_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom callbacks
        custom_callbacks = get_optimized_callbacks(
            self.task_type, backbone_name, 
            self.config.PRETRAINED, self.config, self.models_config
        )
        
        # Built-in TensorFlow callbacks
        tf_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),
            tf.keras.callbacks.CSVLogger(log_dir / 'training.csv'),
            tf.keras.callbacks.ModelCheckpoint(
                log_dir / 'best_model.weights.h5',
                monitor='val_loss', save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )
        ]
        
        return custom_callbacks + tf_callbacks
    
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
            dataset = dataset.shuffle(1000).repeat()
        
        dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset



class MultiTaskTrainer(BaseTrainer):
    """Enhanced multitask trainer with advanced features."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize task-specific components
        self._setup_multitask_components()
        
    def _setup_multitask_components(self):
        """Setup multitask-specific components."""
        # Task weights
        self.task_weights_learnable = getattr(self.config, 'LEARNABLE_TASK_WEIGHTS', False)
        
        if self.task_weights_learnable:
            self.task_weight_vars = {
                'detection': tf.Variable(1.0, trainable=True, name='detection_weight'),
                'segmentation': tf.Variable(1.0, trainable=True, name='segmentation_weight')
            }
        else:
            self.task_weights = getattr(self.config, 'TASK_WEIGHTS', {'detection': 1.0, 'segmentation': 1.0})
            
        # Task-specific loss functions
        self.detection_loss = get_sota_loss_function(self.models_config.LOSS_FUNCTIONS['detection'])
        self.segmentation_loss = get_sota_loss_function(self.models_config.LOSS_FUNCTIONS['segmentation'])
        
        # Task-specific metrics
        self.detection_metrics = get_metrics('detection', self.models_config.NUM_CLASSES['detection'])
        self.segmentation_metrics = get_metrics('segmentation', self.models_config.NUM_CLASSES['segmentation'])


    @tf.function
    def train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Enhanced multitask training step."""
        images, targets = batch
        detection_targets, segmentation_targets = targets
        
        predictions = self.model(images, training=True)
        detection_preds, segmentation_preds = predictions
        
        # Calculate individual losses
        det_loss = self.detection_loss(detection_targets, detection_preds)
        seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
        
        # Compute task weights
        det_weight, seg_weight = self._compute_task_weights(det_loss, seg_loss)
        
        # Combined loss
        total_loss = det_weight * det_loss + seg_weight * seg_loss
        
        # Update metrics
        self.detection_metrics.update_state(detection_targets, detection_preds)
        self.segmentation_metrics.update_state(segmentation_targets, segmentation_preds)
        
        return {
            'loss': total_loss,
            'det_loss': det_loss,
            'seg_loss': seg_loss,
            'det_weight': det_weight,
            'seg_weight': seg_weight
        }        

    
    @tf.function
    def validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Enhanced multitask validation step."""
        images, targets = batch
        detection_targets, segmentation_targets = targets
        
        predictions = self.model(images, training=False)
        detection_preds, segmentation_preds = predictions
        
        # Calculate losses
        det_loss = self.detection_loss(detection_targets, detection_preds)
        seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
        
        # Compute weights (for consistency)
        det_weight, seg_weight = self._compute_task_weights(det_loss, seg_loss)
        
        total_loss = det_weight * det_loss + seg_weight * seg_loss
        
        return {
            'loss': total_loss,
            'det_loss': det_loss,
            'seg_loss': seg_loss
        }
        
    def _compute_task_weights(self, det_loss: tf.Tensor, seg_loss: tf.Tensor) -> tuple:
        """Compute adaptive task weights."""
        if self.task_weights_learnable:
            det_weight = tf.nn.softplus(self.task_weight_vars['detection'])
            seg_weight = tf.nn.softplus(self.task_weight_vars['segmentation'])
        else:
            # Use fixed weights or uncertainty-based weighting
            use_uncertainty = getattr(self.config, 'USE_UNCERTAINTY_WEIGHTING', False)
            
            if use_uncertainty:
                # Uncertainty-based weighting (homoscedastic uncertainty)
                det_weight = 1.0 / (2 * tf.square(det_loss) + 1e-8)
                seg_weight = 1.0 / (2 * tf.square(seg_loss) + 1e-8)
                
                # Normalize weights to prevent vanishing gradients
                total_weight = det_weight + seg_weight
                det_weight = det_weight / total_weight
                seg_weight = seg_weight / total_weight
            else:
                # Fixed weights from config
                det_weight = tf.constant(self.task_weights['detection'], dtype=tf.float32)
                seg_weight = tf.constant(self.task_weights['segmentation'], dtype=tf.float32)
        
        return det_weight, seg_weight
    
    def _run_training_phase(self, dataset) -> Dict[str, float]:
        """Enhanced training phase for multitask learning."""
        self.state.is_training = True
        
        # Initialize accumulators
        total_loss = 0.0
        total_det_loss = 0.0
        total_seg_loss = 0.0
        total_det_weight = 0.0
        total_seg_weight = 0.0
        num_batches = 0
        
        # Reset metrics
        self.detection_metrics.reset_state()
        self.segmentation_metrics.reset_state()
        
        # Reset accumulated gradients
        self.accumulated_gradients = []
        
        for batch_idx, batch in enumerate(dataset):
            step_start_time = time.time()
            
            # Training step with gradient accumulation
            step_metrics = self._training_step_with_accumulation(batch, batch_idx)
            
            # Accumulate losses
            total_loss += step_metrics['loss']
            total_det_loss += step_metrics['det_loss']
            total_seg_loss += step_metrics['seg_loss']
            total_det_weight += step_metrics['det_weight']
            total_seg_weight += step_metrics['seg_weight']
            num_batches += 1
            
            # Log step performance
            step_time = time.time() - step_start_time
            self.logger.log_performance(step_time)
            
            # Memory cleanup periodically
            if batch_idx % 100 == 0:
                self._cleanup_memory()
                
        # Calculate averages
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_det_loss = total_det_loss / num_batches
            avg_seg_loss = total_seg_loss / num_batches
            avg_det_weight = total_det_weight / num_batches
            avg_seg_weight = total_seg_weight / num_batches
        else:
            avg_loss = avg_det_loss = avg_seg_loss = 0.0
            avg_det_weight = avg_seg_weight = 1.0
        
        # Get task-specific metrics
        det_metric_results = self._get_task_metric_results(self.detection_metrics, 'det_')
        seg_metric_results = self._get_task_metric_results(self.segmentation_metrics, 'seg_')
        
        return {
            'loss': avg_loss,
            'det_loss': avg_det_loss,
            'seg_loss': avg_seg_loss,
            'det_weight': avg_det_weight,
            'seg_weight': avg_seg_weight,
            **det_metric_results,
            **seg_metric_results
        }
    
    def _run_validation_phase(self, dataset) -> Dict[str, float]:
        """Enhanced validation phase for multitask learning."""
        self.state.is_training = False
        
        # Initialize accumulators
        total_loss = 0.0
        total_det_loss = 0.0
        total_seg_loss = 0.0
        num_batches = 0
        
        # Reset validation metrics
        val_det_metrics = get_metrics('detection', self.models_config.NUM_CLASSES['detection'])
        val_seg_metrics = get_metrics('segmentation', self.models_config.NUM_CLASSES['segmentation'])
        
        for batch in dataset:
            step_results = self.validation_step(batch)
            
            total_loss += step_results['loss']
            total_det_loss += step_results['det_loss']
            total_seg_loss += step_results['seg_loss']
            num_batches += 1
            
            # Update validation metrics
            images, targets = batch
            detection_targets, segmentation_targets = targets
            predictions = self.model(images, training=False)
            detection_preds, segmentation_preds = predictions
            
            val_det_metrics.update_state(detection_targets, detection_preds)
            val_seg_metrics.update_state(segmentation_targets, segmentation_preds)
        
        # Calculate averages
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_det_loss = total_det_loss / num_batches
            avg_seg_loss = total_seg_loss / num_batches
        else:
            avg_loss = avg_det_loss = avg_seg_loss = 0.0
        
        # Get validation metrics
        val_det_results = self._get_task_metric_results(val_det_metrics, 'det_')
        val_seg_results = self._get_task_metric_results(val_seg_metrics, 'seg_')
        
        return {
            'loss': avg_loss,
            'det_loss': avg_det_loss,
            'seg_loss': avg_seg_loss,
            **val_det_results,
            **val_seg_results
        }
    
    def _get_task_metric_results(self, metrics, prefix: str = '') -> Dict[str, float]:
        """Extract task-specific metric results with prefix."""
        if hasattr(metrics, 'result'):
            if callable(metrics.result):
                results = metrics.result()
            else:
                results = metrics.result
                
            # Handle different metric result types
            if isinstance(results, dict):
                return {f'{prefix}{k}': float(v) for k, v in results.items()}
            elif hasattr(results, 'numpy'):
                # Single metric case
                return {f'{prefix}metric': float(results.numpy())}
            else:
                return {f'{prefix}metric': float(results)}
        return {}
    
    def _training_step_with_accumulation(self, batch, batch_idx: int) -> Dict[str, float]:
        """Enhanced training step with gradient accumulation for multitask."""
        with tf.GradientTape() as tape:
            # Forward pass
            step_results = self.train_step(batch)
            loss = step_results['loss']
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / self.gradient_accumulation_steps
            
            # Add regularization if enabled
            if getattr(self.config, 'USE_WEIGHT_DECAY', False):
                weight_decay = getattr(self.config, 'WEIGHT_DECAY', 1e-4)
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables
                                   if 'bias' not in v.name and 'batch_norm' not in v.name])
                scaled_loss += weight_decay * l2_loss
        
        # Calculate gradients
        trainable_vars = self.model.trainable_variables
        if self.task_weights_learnable:
            trainable_vars += list(self.task_weight_vars.values())
            
        gradients = tape.gradient(scaled_loss, trainable_vars)
        
        # Gradient clipping if enabled
        if getattr(self.config, 'GRADIENT_CLIPPING', False):
            clip_norm = getattr(self.config, 'GRADIENT_CLIP_NORM', 1.0)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        
        # Accumulate gradients
        if not self.accumulated_gradients:
            self.accumulated_gradients = gradients
        else:
            self.accumulated_gradients = [
                acc_grad + grad for acc_grad, grad in zip(self.accumulated_gradients, gradients)
            ]
        
        # Apply gradients when accumulation is complete
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.apply_gradients(zip(self.accumulated_gradients, trainable_vars))
            self.accumulated_gradients = []
        
        return {
            'loss': float(loss),
            'det_loss': float(step_results['det_loss']),
            'seg_loss': float(step_results['seg_loss']),
            'det_weight': float(step_results['det_weight']),
            'seg_weight': float(step_results['seg_weight'])
        }
    
    def _update_best_metrics(self, metrics: Dict[str, float]):
        """Enhanced metric tracking for multitask learning."""
        current_val_loss = metrics.get('val_loss', metrics.get('loss', float('inf')))
        current_det_loss = metrics.get('val_det_loss', metrics.get('det_loss', float('inf')))
        current_seg_loss = metrics.get('val_seg_loss', metrics.get('seg_loss', float('inf')))
        
        # Update best overall loss
        if current_val_loss < self.state.best_val_loss:
            self.state.best_val_loss = current_val_loss
            self._save_checkpoint(is_best=True, checkpoint_type='overall')
        
        # Track best task-specific losses
        if not hasattr(self.state, 'best_det_loss'):
            self.state.best_det_loss = float('inf')
        if not hasattr(self.state, 'best_seg_loss'):
            self.state.best_seg_loss = float('inf')
            
        if current_det_loss < self.state.best_det_loss:
            self.state.best_det_loss = current_det_loss
            self._save_checkpoint(is_best=True, checkpoint_type='detection')
            
        if current_seg_loss < self.state.best_seg_loss:
            self.state.best_seg_loss = current_seg_loss
            self._save_checkpoint(is_best=True, checkpoint_type='segmentation')
        
        # Regular checkpoint save
        if self.state.current_epoch % getattr(self.config, 'CHECKPOINT_FREQ', 10) == 0:
            self._save_checkpoint(is_best=False, checkpoint_type='regular')
    
    def _save_checkpoint(self, is_best: bool = False, checkpoint_type: str = 'regular'):
        """Enhanced checkpoint saving with task-specific information."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'state': self.state.to_dict(),
            'optimizer_config': self.optimizer.get_config(),
            'history': self.logger.history,
            'epoch': self.state.current_epoch,
            'task_weights': {}
        }
        
        # Save task weights
        if self.task_weights_learnable:
            checkpoint_data['task_weights'] = {
                'detection': float(self.task_weight_vars['detection'].numpy()),
                'segmentation': float(self.task_weight_vars['segmentation'].numpy())
            }
        else:
            checkpoint_data['task_weights'] = self.task_weights
        
        # Determine checkpoint filename
        if is_best:
            checkpoint_name = f'best_{checkpoint_type}_checkpoint.json'
            weights_name = f'best_{checkpoint_type}_weights.h5'
        else:
            checkpoint_name = f'checkpoint_epoch_{self.state.current_epoch}.json'
            weights_name = f'weights_epoch_{self.state.current_epoch}.h5'
        
        # Save checkpoint metadata
        checkpoint_path = checkpoint_dir / checkpoint_name
        save_json(checkpoint_data, checkpoint_path)
        
        # Save model weights
        weights_path = checkpoint_dir / weights_name
        self.model.save_weights(str(weights_path))
        
        self.logger.logger.info(f"Checkpoint saved: {checkpoint_type} ({'best' if is_best else 'regular'})")
    
    def _load_checkpoint(self):
        """Enhanced checkpoint loading with task weight restoration."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        
        # Priority order for loading checkpoints
        checkpoint_candidates = [
            'best_overall_checkpoint.json',
            'best_detection_checkpoint.json',
            'best_segmentation_checkpoint.json'
        ]
        
        # Add latest epoch checkpoints
        epoch_checkpoints = sorted(
            checkpoint_dir.glob('checkpoint_epoch_*.json'),
            key=lambda x: int(x.stem.split('_')[-1]),
            reverse=True
        )
        checkpoint_candidates.extend([f.name for f in epoch_checkpoints[:3]])
        
        for checkpoint_name in checkpoint_candidates:
            checkpoint_path = checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                try:
                    checkpoint_data = load_json(checkpoint_path)
                    
                    # Restore state
                    self.state = TrainingState.from_dict(checkpoint_data['state'])
                    self.logger.history = checkpoint_data.get('history', self.logger.history)
                    
                    # Restore task weights
                    if 'task_weights' in checkpoint_data:
                        if self.task_weights_learnable:
                            for task, weight in checkpoint_data['task_weights'].items():
                                if task in self.task_weight_vars:
                                    self.task_weight_vars[task].assign(weight)
                        else:
                            self.task_weights.update(checkpoint_data['task_weights'])
                    
                    # Load model weights
                    weights_name = checkpoint_name.replace('checkpoint.json', 'weights.h5')
                    weights_path = checkpoint_dir / weights_name
                    if weights_path.exists():
                        self.model.load_weights(str(weights_path))
                    
                    self.logger.logger.info(f"Resumed from {checkpoint_name} at epoch {self.state.current_epoch}")
                    return
                    
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load {checkpoint_name}: {e}")
                    continue
        
        self.logger.logger.warning("No valid checkpoint found to resume from")
    
    def _save_final_results(self):
        """Enhanced final results saving for multitask learning."""
        results = {
            'training_summary': {
                'task_type': 'multitask',
                'tasks': ['detection', 'segmentation'],
                'backbone_name': self.backbone_name,
                'total_epochs': self.state.current_epoch + 1,
                'total_training_time': self.state.training_time,
                'best_val_loss': self.state.best_val_loss,
                'best_det_loss': getattr(self.state, 'best_det_loss', None),
                'best_seg_loss': getattr(self.state, 'best_seg_loss', None)
            },
            'final_state': self.state.to_dict(),
            'task_weights': {},
            'config_summary': {
                'optimizer': self.optimizer.__class__.__name__,
                'detection_loss': self.detection_loss.__class__.__name__ if hasattr(self.detection_loss, '__class__') else str(self.detection_loss),
                'segmentation_loss': self.segmentation_loss.__class__.__name__ if hasattr(self.segmentation_loss, '__class__') else str(self.segmentation_loss),
                'learnable_task_weights': self.task_weights_learnable,
                'mixed_precision': getattr(self.config, 'USE_MIXED_PRECISION', False),
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'uncertainty_weighting': getattr(self.config, 'USE_UNCERTAINTY_WEIGHTING', False)
            }
        }
        
        # Add final task weights
        if self.task_weights_learnable:
            results['task_weights'] = {
                'detection': float(self.task_weight_vars['detection'].numpy()),
                'segmentation': float(self.task_weight_vars['segmentation'].numpy())
            }
        else:
            results['task_weights'] = self.task_weights
        
        # Save results
        results_path = Path(self.config.LOGS_DIR) / 'multitask_final_results.json'
        save_json(results, results_path)
        
        # Save training history
        self.logger.save_history()
        
        # Save task weight evolution if learnable
        if self.task_weights_learnable and hasattr(self.logger, 'task_weight_history'):
            weight_history_path = Path(self.config.LOGS_DIR) / 'task_weight_evolution.json'
            save_json(self.logger.task_weight_history, weight_history_path)
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.task_weights_learnable:
            return {
                'detection': float(self.task_weight_vars['detection'].numpy()),
                'segmentation': float(self.task_weight_vars['segmentation'].numpy())
            }
        else:
            return self.task_weights.copy()
    
    def set_task_weights(self, weights: Dict[str, float]):
        """Set task weights manually."""
        if self.task_weights_learnable:
            for task, weight in weights.items():
                if task in self.task_weight_vars:
                    self.task_weight_vars[task].assign(weight)
        else:
            self.task_weights.update(weights)
        
        self.logger.logger.info(f"Task weights updated: {weights}")
    
    def evaluate_task_performance(self, dataset: tf.data.Dataset) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on each task separately."""
        self.state.is_training = False
        
        # Reset metrics
        det_metrics = get_metrics('detection', self.models_config.NUM_CLASSES['detection'])
        seg_metrics = get_metrics('segmentation', self.models_config.NUM_CLASSES['segmentation'])
        
        total_det_loss = 0.0
        total_seg_loss = 0.0
        num_batches = 0
        
        for batch in dataset:
            images, targets = batch
            detection_targets, segmentation_targets = targets
            
            predictions = self.model(images, training=False)
            detection_preds, segmentation_preds = predictions
            
            # Calculate losses
            det_loss = self.detection_loss(detection_targets, detection_preds)
            seg_loss = self.segmentation_loss(segmentation_targets, segmentation_preds)
            
            total_det_loss += float(det_loss)
            total_seg_loss += float(seg_loss)
            num_batches += 1
            
            # Update metrics
            det_metrics.update_state(detection_targets, detection_preds)
            seg_metrics.update_state(segmentation_targets, segmentation_preds)
        
        # Calculate averages
        avg_det_loss = total_det_loss / num_batches if num_batches > 0 else 0.0
        avg_seg_loss = total_seg_loss / num_batches if num_batches > 0 else 0.0
        
        # Get metric results
        det_results = self._get_task_metric_results(det_metrics)
        seg_results = self._get_task_metric_results(seg_metrics)
        
        return {
            'detection': {
                'loss': avg_det_loss,
                **det_results
            },
            'segmentation': {
                'loss': avg_seg_loss,
                **seg_results
            }
        }