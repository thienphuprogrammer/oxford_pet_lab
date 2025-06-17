from __future__ import annotations
import os
from typing import List, Optional
from pathlib import Path

import tensorflow as tf
import math


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and cosine decay.
    
    This can be used directly in optimizer constructors as an alternative
    to using the LearningRateScheduler callback.
    """
    
    def __init__(self, initial_lr: float, total_steps: int, warmup_steps: int = 0, alpha: float = 0.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        
        # Pre-compute constants
        self.min_lr = initial_lr * alpha
        self.lr_range = initial_lr - self.min_lr
        self.decay_steps = max(1, total_steps - warmup_steps)
    
    def __call__(self, step):
        """Compute learning rate for given step."""
        step = tf.cast(step, tf.float32)
        
        # Warmup phase
        warmup_lr = self.initial_lr * step / max(1, self.warmup_steps)
        
        # Decay phase
        decay_step = (step - self.warmup_steps) / self.decay_steps
        decay_step = tf.clip_by_value(decay_step, 0.0, 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(tf.constant(math.pi) * decay_step))
        decay_lr = self.min_lr + self.lr_range * cosine_decay
        
        # Choose appropriate LR based on current step
        return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
    
    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


def cosine_decay_with_warmup(
    initial_lr: float,
    total_epochs: int,
    warmup_epochs: int = 0,
    alpha: float = 0.0
) -> callable:
    # Pre-compute constants to avoid repeated calculations
    warmup_factor = initial_lr / max(1, warmup_epochs) if warmup_epochs > 0 else 0
    decay_epochs = max(1, total_epochs - warmup_epochs)
    min_lr = initial_lr * alpha
    lr_range = initial_lr - min_lr
    
    def schedule(epoch: int) -> float:
        """Compute learning rate for given epoch (zero-based)."""
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return warmup_factor * (epoch + 1)
        
        progress = (epoch - warmup_epochs) / decay_epochs
        # Clamp progress to [0, 1] to handle edge cases
        progress = max(0.0, min(1.0, progress))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + lr_range * cosine_decay
    
    return schedule


def get_callbacks(
    task: str,
    log_dir: str | Path,
    ckpt_dir: str | Path,
    monitor: str = "val_loss",
    patience: int = 10,
    total_epochs: int = 100,
    initial_lr: float = 1e-3,
    warmup_epochs: int = 0,
    use_tensorboard: bool = True,
    save_weights_only: bool = False,
    enable_lr_schedule: bool = True,
    enable_reduce_lr: bool = True,
    min_delta: float = 1e-4,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-8,
    csv_append: bool = True,
    tensorboard_histogram_freq: int = 0,  # Changed default to 0 for performance
    terminate_on_nan: bool = True,
    optimizer_has_schedule: bool = False  # NEW: Flag to indicate if optimizer already has LR schedule
) -> List[tf.keras.callbacks.Callback]:
    # Convert to Path objects for better path handling
    log_dir = Path(log_dir)
    ckpt_dir = Path(ckpt_dir)
    
    # Create directories efficiently
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine monitoring mode based on metric name
    mode = "min" if any(keyword in monitor.lower() 
                       for keyword in ["loss", "error", "mae", "mse"]) else "max"
    
    # Core callbacks list
    callbacks_list: List[tf.keras.callbacks.Callback] = []
    
    # Early stopping with optimized settings
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        mode=mode,
        verbose=1,
        start_from_epoch=warmup_epochs  # Don't start monitoring during warmup
    )
    callbacks_list.append(early_stop)
    
    # Model checkpoint with proper file extension
    file_ext = ".weights.h5" if save_weights_only else ".h5"
    ckpt_path = ckpt_dir / f"best_{task}_epoch_{{epoch:03d}}{file_ext}"
    
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=save_weights_only,
        verbose=1,
        save_freq='epoch'  # Explicit save frequency
    )
    callbacks_list.append(ckpt)
    
    # Cosine learning rate schedule (only if optimizer doesn't have one)
    if enable_lr_schedule and not optimizer_has_schedule:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            cosine_decay_with_warmup(
                initial_lr=initial_lr,
                total_epochs=total_epochs,
                warmup_epochs=warmup_epochs,
            ),
            verbose=1,  # Show LR changes
        )
        callbacks_list.append(lr_scheduler)
    elif enable_lr_schedule and optimizer_has_schedule:
        print("WARNING: Skipping LearningRateScheduler callback because optimizer already has a learning rate schedule.")
    
    # Reduce LR on plateau as safety net (only if no optimizer schedule)
    if enable_reduce_lr and not optimizer_has_schedule:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=max(2, patience // 3),  # Minimum patience of 2
            min_lr=min_lr,
            mode=mode,
            verbose=1,
            cooldown=2  # Prevent too frequent reductions
        )
        callbacks_list.append(reduce_lr)
    elif enable_reduce_lr and optimizer_has_schedule:
        print("WARNING: Skipping ReduceLROnPlateau callback because optimizer already has a learning rate schedule.")
    
    # CSV logging
    csv_path = log_dir / "training_log.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=str(csv_path), 
        append=csv_append,
        separator=','
    )
    callbacks_list.append(csv_logger)
    
    # TensorBoard logging
    if use_tensorboard:
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=tensorboard_histogram_freq,
            write_graph=True,
            write_images=False,  # Disable to save space
            update_freq='epoch',  # More efficient than 'batch'
            profile_batch=0,  # Disable profiling by default for performance
            embeddings_freq=0
        )
        callbacks_list.append(tensorboard)
    
    # Terminate on NaN
    if terminate_on_nan:
        callbacks_list.append(tf.keras.callbacks.TerminateOnNaN())
    
    return callbacks_list


# Utility function for quick callback setup
def get_standard_callbacks(
    task: str,
    base_dir: str | Path = "./experiments",
    **kwargs
) -> List[tf.keras.callbacks.Callback]:
    """
    Convenience function for standard callback setup.
    
    Args:
        task: Task name
        base_dir: Base directory for logs and checkpoints
        **kwargs: Additional arguments passed to get_callbacks
    
    Returns:
        List of configured callbacks
    """
    base_dir = Path(base_dir)
    log_dir = base_dir / task / "logs"
    ckpt_dir = base_dir / task / "checkpoints"
    
    return get_callbacks(
        task=task,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        **kwargs
    )