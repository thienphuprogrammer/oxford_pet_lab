"""
Optimized optimizer configurations for computer vision tasks.
"""

import tensorflow as tf
from typing import Dict, Any, Optional, Union


# Pre-defined configurations for different CV tasks
TASK_CONFIGS = {
    # Object Detection
    'yolo': {'lr': 0.01, 'wd': 0.0005, 'momentum': 0.937, 'warmup': 3},
    'rcnn': {'lr': 0.02, 'wd': 0.0001, 'momentum': 0.9, 'warmup': 1},
    'ssd': {'lr': 0.001, 'wd': 0.0005, 'momentum': 0.9, 'warmup': 2},
    
    # Segmentation
    'unet': {'lr': 0.001, 'wd': 0.0001, 'momentum': 0.9, 'warmup': 2, 'schedule': 'poly'},
    'deeplabv3': {'lr': 0.007, 'wd': 0.0001, 'momentum': 0.9, 'warmup': 1, 'schedule': 'poly'},
    
    # Classification
    'resnet': {'lr': 0.1, 'wd': 0.0001, 'momentum': 0.9, 'warmup': 5},
    'resnet50': {'lr': 0.1, 'wd': 0.0001, 'momentum': 0.9, 'warmup': 5},
    'efficientnet': {'lr': 0.016, 'wd': 0.00001, 'momentum': 0.9, 'warmup': 3},
    'vit': {'lr': 0.001, 'wd': 0.05, 'warmup': 10},
}

# Backbone learning rate multipliers
BACKBONE_LR_MULT = {
    'resnet50': 1.0, 'efficientnet': 0.8, 'vit': 0.1, 'swin': 0.05, 'hrnet': 0.1
}


def create_lr_schedule(
    initial_lr: float,
    total_epochs: int,
    schedule_type: str = 'cosine',
    warmup_epochs: int = 0,
    steps_per_epoch: int = 1000
) -> Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]:
    """Create learning rate schedule with warmup support."""
    
    if warmup_epochs == 0 and schedule_type == 'constant':
        return initial_lr
    
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Create base schedule
    if schedule_type == 'cosine':
        base_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=0.01
        )
    elif schedule_type == 'poly':
        base_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps - warmup_steps,
            power=0.9,
            end_learning_rate=initial_lr * 0.01
        )
    elif schedule_type == 'exponential':
        base_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps // 10,
            decay_rate=0.96,
            staircase=True
        )
    else:  # constant or step
        base_schedule = initial_lr
    
    # Add warmup if needed using a custom schedule to avoid nested schedules
    if warmup_epochs > 0:
        class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, init_lr, warmup_steps, after_schedule):
                super().__init__()
                self.init_lr = init_lr
                self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                self.after_schedule = after_schedule
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                warmup_lr = self.init_lr * (step + 1) / self.warmup_steps
                return tf.where(step < self.warmup_steps, warmup_lr, self.after_schedule(step - self.warmup_steps))
            def get_config(self):
                return {
                    'init_lr': self.init_lr,
                    'warmup_steps': int(self.warmup_steps.numpy() if isinstance(self.warmup_steps, tf.Tensor) else self.warmup_steps),
                }
        return WarmUpSchedule(initial_lr, warmup_steps, base_schedule)

    return base_schedule


def create_cv_optimizer(
    task: str,
    optimizer_type: str = 'sgd',
    backbone: Optional[str] = None,
    batch_size: int = 16,
    total_epochs: int = 100,
    custom_lr: Optional[float] = None,
    **kwargs
) -> tf.keras.optimizers.Optimizer:
    """
    Create optimized optimizer for computer vision tasks.
    
    Args:
        task: Task type (yolo, rcnn, unet, deeplabv3, resnet, efficientnet, vit)
        optimizer_type: Optimizer type (sgd, adam, adamw, radam, lion)
        backbone: Backbone architecture (optional, for lr adjustment)
        batch_size: Training batch size (for lr scaling)
        total_epochs: Total training epochs
        custom_lr: Custom learning rate (overrides default)
        **kwargs: Additional optimizer parameters
        
    Returns:
        Configured TensorFlow optimizer
    """
    
    # Get base configuration
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unsupported task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    
    config = TASK_CONFIGS[task].copy()
    
    # Calculate learning rate
    lr = custom_lr if custom_lr else config['lr']
    
    # Apply backbone multiplier
    if backbone and backbone in BACKBONE_LR_MULT:
        lr *= BACKBONE_LR_MULT[backbone]
    
    # Apply batch size scaling (linear scaling rule)
    if batch_size != 16:
        lr *= (batch_size / 16)
    
    # Create learning rate schedule
    schedule_type = config.get('schedule', 'cosine')
    warmup_epochs = config.get('warmup', 0)
    
    lr_schedule = create_lr_schedule(
        initial_lr=lr,
        total_epochs=total_epochs,
        schedule_type=schedule_type,
        warmup_epochs=warmup_epochs,
        steps_per_epoch=kwargs.get('steps_per_epoch', 1000)
    )
    
    # Common optimizer parameters
    common_params = {
        'learning_rate': lr_schedule,
        'clipnorm': kwargs.get('clipnorm', 10.0 if 'detection' in task or 'yolo' in task else 5.0)
    }
    
    # Create optimizer based on type
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD(
            momentum=config.get('momentum', 0.9),
            nesterov=True,
            **common_params
        )
    
    elif optimizer_type == 'adam':
        return tf.keras.optimizers.Adam(
            beta_1=kwargs.get('beta_1', 0.9),
            beta_2=kwargs.get('beta_2', 0.999),
            **common_params
        )
    
    elif optimizer_type == 'adamw':
        return tf.keras.optimizers.AdamW(
            weight_decay=config.get('wd', 0.0001),
            beta_1=kwargs.get('beta_1', 0.9),
            beta_2=kwargs.get('beta_2', 0.999),
            **common_params
        )
    
    elif optimizer_type == 'radam':
        # RAdamW optimizer (if available in your TF version)
        try:
            return tf.keras.optimizers.RMSprop(
                rho=kwargs.get('rho', 0.9),
                momentum=config.get('momentum', 0.0),
                **common_params
            )
        except AttributeError:
            # Fallback to Adam if RAdamW not available
            return tf.keras.optimizers.Adam(**common_params)
    
    elif optimizer_type == 'lion':
        # Lion optimizer (requires tensorflow-addons or custom implementation)
        try:
            import tensorflow_addons as tfa
            return tfa.optimizers.LAMB(
                weight_decay_rate=config.get('wd', 0.0001),
                **common_params
            )
        except ImportError:
            # Fallback to AdamW if Lion not available
            return tf.keras.optimizers.AdamW(
                weight_decay=config.get('wd', 0.0001),
                **common_params
            )
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def get_augmentation_config(task: str) -> Dict[str, Any]:
    """Get recommended data augmentation configuration for different tasks."""
    
    base_config = {
        'random_flip_horizontal': True,
        'random_rotation': 0.1,
        'random_brightness': 0.1,
        'random_contrast': 0.1,
    }
    
    if 'detection' in task or task in ['yolo', 'rcnn', 'ssd']:
        base_config.update({
            'random_crop': True,
            'mosaic': 0.5,
            'mixup': 0.1,
        })
    
    elif 'segment' in task or task in ['unet', 'deeplabv3']:
        base_config.update({
            'random_scale': (0.5, 2.0),
            'gaussian_blur': 0.1,
        })
    
    elif task in ['resnet', 'efficientnet', 'vit']:
        base_config.update({
            'random_crop': True,
            'cutmix': 0.1,
            'mixup': 0.2,
        })
    
    return base_config


# Convenience functions for common use cases
def create_detection_optimizer(model='yolo', **kwargs):
    """Create optimizer for object detection."""
    return create_cv_optimizer(task=model, **kwargs)


def create_segmentation_optimizer(model='unet', **kwargs):
    """Create optimizer for segmentation."""
    return create_cv_optimizer(task=model, **kwargs)


def create_classification_optimizer(model='resnet', **kwargs):
    """Create optimizer for classification."""
    return create_cv_optimizer(task=model, **kwargs)
