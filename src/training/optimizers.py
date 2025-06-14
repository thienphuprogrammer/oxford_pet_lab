"""
Optimizer configurations for different training scenarios.
"""

import tensorflow as tf
from typing import Dict, Any, Optional


class OptimizerFactory:
    """Factory class for creating optimizers with different configurations."""
    
    @staticmethod
    def create_adam_optimizer(
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: Optional[float] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create Adam optimizer with specified parameters.
        
        Args:
            learning_rate: Learning rate
            beta_1: First moment decay rate
            beta_2: Second moment decay rate
            epsilon: Small constant for numerical stability
            weight_decay: Weight decay rate
            clipnorm: Gradient clipping by norm
            clipvalue: Gradient clipping by value
            
        Returns:
            Configured Adam optimizer
        """
        if weight_decay is not None:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                weight_decay=weight_decay,
                clipnorm=clipnorm,
                clipvalue=clipvalue
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                clipnorm=clipnorm,
                clipvalue=clipvalue
            )
        return optimizer
    
    @staticmethod
    def create_sgd_optimizer(
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: Optional[float] = None,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create SGD optimizer with momentum.
        
        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            nesterov: Whether to use Nesterov momentum
            weight_decay: Weight decay rate
            clipnorm: Gradient clipping by norm
            clipvalue: Gradient clipping by value
            
        Returns:
            Configured SGD optimizer
        """
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
        return optimizer
    
    @staticmethod
    def create_rmsprop_optimizer(
        learning_rate: float = 0.001,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create RMSprop optimizer.
        
        Args:
            learning_rate: Learning rate
            rho: Discounting factor for the history/coming gradient
            momentum: Momentum factor
            epsilon: Small constant for numerical stability
            clipnorm: Gradient clipping by norm
            clipvalue: Gradient clipping by value
            
        Returns:
            Configured RMSprop optimizer
        """
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon,
            clipnorm=clipnorm,
            clipvalue=clipvalue
        )
        return optimizer


def create_learning_rate_schedule(
    schedule_type: str,
    initial_learning_rate: float,
    **kwargs
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Create learning rate schedule.
    
    Args:
        schedule_type: Type of schedule ('exponential', 'cosine', 'polynomial', 'step')
        initial_learning_rate: Initial learning rate
        **kwargs: Additional parameters for the schedule
        
    Returns:
        Learning rate schedule
    """
    if schedule_type == 'exponential':
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=kwargs.get('decay_steps', 1000),
            decay_rate=kwargs.get('decay_rate', 0.96),
            staircase=kwargs.get('staircase', False)
        )
    
    elif schedule_type == 'cosine':
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=kwargs.get('decay_steps', 1000),
            alpha=kwargs.get('alpha', 0.0)
        )
    
    elif schedule_type == 'polynomial':
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=kwargs.get('decay_steps', 1000),
            end_learning_rate=kwargs.get('end_learning_rate', 0.0001),
            power=kwargs.get('power', 1.0)
        )
    
    elif schedule_type == 'step':
        boundaries = kwargs.get('boundaries', [1000, 2000])
        values = kwargs.get('values', [initial_learning_rate, 
                                     initial_learning_rate * 0.1, 
                                     initial_learning_rate * 0.01])
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values
        )
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def get_optimizer_config(task_type: str, config: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    """
    Get optimizer configuration for specific task.
    
    Args:
        task_type: Type of task ('detection', 'segmentation', 'multitask')
        config: Configuration dictionary
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.get('optimizer', 'adam').lower()
    learning_rate = config.get('learning_rate', 0.001)
    
    # Create learning rate schedule if specified
    if 'lr_schedule' in config:
        learning_rate = create_learning_rate_schedule(
            schedule_type=config['lr_schedule']['type'],
            initial_learning_rate=learning_rate,
            **config['lr_schedule'].get('params', {})
        )
    
    # Task-specific optimizer configurations
    if task_type == 'detection':
        # Object detection typically uses lower learning rates
        if optimizer_name == 'adam':
            return OptimizerFactory.create_adam_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 1e-4),
                clipnorm=config.get('clipnorm', 10.0)
            )
        elif optimizer_name == 'sgd':
            return OptimizerFactory.create_sgd_optimizer(
                learning_rate=learning_rate,
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 1e-4),
                clipnorm=config.get('clipnorm', 10.0)
            )
    
    elif task_type == 'segmentation':
        # Segmentation often benefits from higher learning rates
        if optimizer_name == 'adam':
            return OptimizerFactory.create_adam_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 1e-5),
                clipnorm=config.get('clipnorm', 5.0)
            )
        elif optimizer_name == 'sgd':
            return OptimizerFactory.create_sgd_optimizer(
                learning_rate=learning_rate,
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 1e-5),
                clipnorm=config.get('clipnorm', 5.0)
            )
    
    elif task_type == 'multitask':
        # Multitask learning requires balanced optimization
        if optimizer_name == 'adam':
            return OptimizerFactory.create_adam_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 5e-5),
                clipnorm=config.get('clipnorm', 7.0)
            )
        elif optimizer_name == 'sgd':
            return OptimizerFactory.create_sgd_optimizer(
                learning_rate=learning_rate,
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 5e-5),
                clipnorm=config.get('clipnorm', 7.0)
            )
    
    # Default fallback
    return OptimizerFactory.create_adam_optimizer(learning_rate=learning_rate)


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup period."""
    
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_schedule: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None
    ):
        """
        Initialize warmup schedule.
        
        Args:
            initial_learning_rate: Target learning rate after warmup
            warmup_steps: Number of warmup steps
            decay_schedule: Optional decay schedule after warmup
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule = decay_schedule
        
    def __call__(self, step):
        """Get learning rate for given step."""
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        
        # Linear warmup
        warmup_lr = self.initial_learning_rate * step / warmup_steps
        
        if self.decay_schedule is not None:
            # Apply decay schedule after warmup
            decay_lr = self.decay_schedule(step - warmup_steps)
            lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        else:
            # Constant learning rate after warmup
            lr = tf.where(step < warmup_steps, warmup_lr, self.initial_learning_rate)
            
        return lr
    
    def get_config(self):
        """Get configuration for serialization."""
        config = {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps
        }
        if self.decay_schedule is not None:
            config['decay_schedule'] = self.decay_schedule.get_config()
        return config


def create_warmup_schedule(
    initial_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    decay_type: str = 'cosine'
) -> WarmupSchedule:
    """
    Create learning rate schedule with warmup and decay.
    
    Args:
        initial_learning_rate: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        decay_type: Type of decay after warmup ('cosine', 'exponential', 'linear')
        
    Returns:
        Warmup learning rate schedule
    """
    decay_steps = total_steps - warmup_steps
    
    if decay_type == 'cosine':
        decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=0.01
        )
    elif decay_type == 'exponential':
        decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps // 3,
            decay_rate=0.9
        )
    elif decay_type == 'linear':
        decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=initial_learning_rate * 0.01,
            power=1.0
        )
    else:
        decay_schedule = None
    
    return WarmupSchedule(
        initial_learning_rate=initial_learning_rate,
        warmup_steps=warmup_steps,
        decay_schedule=decay_schedule
    )
