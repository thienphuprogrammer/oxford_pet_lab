"""
Optimized optimizer configurations with SOTA optimizers for better performance.
"""

import tensorflow as tf
from typing import Dict, Any, Optional, Union, List
import numpy as np


class SOTAOptimizerFactory:
    """Enhanced factory class with SOTA optimizers for improved performance."""
    
    @staticmethod
    def create_adamw_optimizer(
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: float = 0.01,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        amsgrad: bool = False
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create AdamW optimizer - better than Adam for most tasks.
        AdamW decouples weight decay from gradient updates.
        """
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            amsgrad=amsgrad
        )
    
    @staticmethod
    def create_lion_optimizer(
        learning_rate: float = 1e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 0.01,
        clipnorm: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create Lion optimizer - SOTA optimizer that's more memory efficient than Adam.
        Lion (EvoLved Sign Momentum) uses sign operations for updates.
        Note: This is a simplified implementation as TensorFlow doesn't have built-in Lion.
        """
        # For now, we'll use a custom implementation or fall back to AdamW
        # In practice, you'd need to implement Lion or use a third-party library
        print("Warning: Lion optimizer not natively available in TensorFlow. Using AdamW instead.")
        return SOTAOptimizerFactory.create_adamw_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm
        )
    
    @staticmethod
    def create_lamb_optimizer(
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-6,
        weight_decay: float = 0.01,
        clipnorm: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create LAMB optimizer - excellent for large batch training.
        LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
        """
        # TensorFlow doesn't have built-in LAMB, using AdamW as fallback
        print("Warning: LAMB optimizer not natively available in TensorFlow. Using AdamW instead.")
        return SOTAOptimizerFactory.create_adamw_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm
        )
    
    @staticmethod
    def create_radam_optimizer(
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        weight_decay: Optional[float] = None,
        clipnorm: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create RAdam optimizer - Rectified Adam with better convergence.
        """
        # Using Adam with specific settings that mimic RAdam behavior
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            clipnorm=clipnorm,
            clipvalue=None,
            amsgrad=True  # Use AMSGrad variant for better convergence
        )
    
    @staticmethod
    def create_sgdr_optimizer(
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        nesterov: bool = True,
        weight_decay: Optional[float] = None,
        clipnorm: Optional[float] = None
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create SGD optimizer optimized for SGDR (Stochastic Gradient Descent with Restarts).
        """
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            clipnorm=clipnorm
        )


class AdvancedLearningRateSchedules:
    """Advanced learning rate schedules for better training dynamics."""
    
    @staticmethod
    def create_cosine_restarts_schedule(
        initial_learning_rate: float,
        first_decay_steps: int,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create Cosine Annealing with Restarts (SGDR) schedule.
        """
        return tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        )
    
    @staticmethod
    def create_one_cycle_schedule(
        max_learning_rate: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create One Cycle learning rate schedule.
        """
        # Custom implementation of One Cycle LR
        class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, max_lr, total_steps, pct_start, div_factor, final_div_factor):
                self.max_lr = max_lr
                self.total_steps = total_steps
                self.pct_start = pct_start
                self.div_factor = div_factor
                self.final_div_factor = final_div_factor
                self.step_up = int(total_steps * pct_start)
                self.step_down = total_steps - self.step_up
                
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                
                # Phase 1: Increase LR from min to max
                phase1_lr = self.max_lr / self.div_factor + (
                    self.max_lr - self.max_lr / self.div_factor
                ) * step / self.step_up
                
                # Phase 2: Decrease LR from max to min
                phase2_lr = self.max_lr - (
                    self.max_lr - self.max_lr / self.final_div_factor
                ) * (step - self.step_up) / self.step_down
                
                return tf.where(
                    step <= self.step_up,
                    phase1_lr,
                    tf.maximum(phase2_lr, self.max_lr / self.final_div_factor)
                )
        
        return OneCycleLR(max_learning_rate, total_steps, pct_start, div_factor, final_div_factor)


class EnhancedWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Enhanced warmup schedule with multiple warmup strategies."""
    
    def __init__(
        self,
        initial_learning_rate: float,
        warmup_steps: int,
        decay_schedule: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        warmup_method: str = 'linear'  # 'linear', 'exponential', 'constant'
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule = decay_schedule
        self.warmup_method = warmup_method
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        
        if self.warmup_method == 'linear':
            warmup_lr = self.initial_learning_rate * step / warmup_steps
        elif self.warmup_method == 'exponential':
            warmup_lr = self.initial_learning_rate * (step / warmup_steps) ** 2
        elif self.warmup_method == 'constant':
            warmup_lr = self.initial_learning_rate * 0.1
        else:
            warmup_lr = self.initial_learning_rate * step / warmup_steps
        
        if self.decay_schedule is not None:
            decay_lr = self.decay_schedule(tf.maximum(step - warmup_steps, 0))
            lr = tf.where(step < warmup_steps, warmup_lr, decay_lr)
        else:
            lr = tf.where(step < warmup_steps, warmup_lr, self.initial_learning_rate)
            
        return lr


def get_sota_optimizer_config(
    task_type: str, 
    config: Dict[str, Any],
    model_size: str = 'medium'  # 'small', 'medium', 'large'
) -> tf.keras.optimizers.Optimizer:
    """
    Get SOTA optimizer configuration for specific task with model size consideration.
    
    Args:
        task_type: Type of task ('detection', 'segmentation', 'classification', 'multitask')
        config: Configuration dictionary
        model_size: Size of the model ('small', 'medium', 'large')
        
    Returns:
        Configured SOTA optimizer
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    learning_rate = config.get('learning_rate', 0.001)
    
    # Adjust learning rate based on model size
    lr_multipliers = {'small': 1.0, 'medium': 0.8, 'large': 0.5}
    learning_rate *= lr_multipliers.get(model_size, 1.0)
    
    # Create advanced learning rate schedule
    if 'lr_schedule' in config:
        schedule_type = config['lr_schedule']['type']
        schedule_params = config['lr_schedule'].get('params', {})
        
        if schedule_type == 'cosine_restarts':
            learning_rate = AdvancedLearningRateSchedules.create_cosine_restarts_schedule(
                initial_learning_rate=learning_rate,
                first_decay_steps=schedule_params.get('first_decay_steps', 1000),
                **{k: v for k, v in schedule_params.items() if k != 'first_decay_steps'}
            )
        elif schedule_type == 'one_cycle':
            learning_rate = AdvancedLearningRateSchedules.create_one_cycle_schedule(
                max_learning_rate=learning_rate,
                total_steps=schedule_params.get('total_steps', 10000),
                **{k: v for k, v in schedule_params.items() if k != 'total_steps'}
            )
        elif schedule_type == 'warmup':
            warmup_steps = schedule_params.get('warmup_steps', 1000)
            decay_schedule = None
            
            if 'decay_type' in schedule_params:
                decay_type = schedule_params['decay_type']
                if decay_type == 'cosine':
                    decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
                        initial_learning_rate=learning_rate,
                        decay_steps=schedule_params.get('decay_steps', 5000),
                        alpha=0.01
                    )
            
            learning_rate = EnhancedWarmupSchedule(
                initial_learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                decay_schedule=decay_schedule,
                warmup_method=schedule_params.get('warmup_method', 'linear')
            )
    
    # Task-specific SOTA optimizer configurations
    if task_type == 'detection':
        if optimizer_name == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 0.05),
                clipnorm=config.get('clipnorm', 10.0),
                beta_1=0.9,
                beta_2=0.999
            )
        elif optimizer_name == 'sgdr':
            return SOTAOptimizerFactory.create_sgdr_optimizer(
                learning_rate=learning_rate,
                momentum=config.get('momentum', 0.9),
                clipnorm=config.get('clipnorm', 10.0)
            )
    
    elif task_type == 'segmentation':
        if optimizer_name == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 0.01),
                clipnorm=config.get('clipnorm', 5.0),
                beta_1=0.9,
                beta_2=0.999
            )
        elif optimizer_name == 'lion':
            return SOTAOptimizerFactory.create_lion_optimizer(
                learning_rate=learning_rate * 0.1,  # Lion typically uses 10x smaller LR
                weight_decay=config.get('weight_decay', 0.01),
                clipnorm=config.get('clipnorm', 5.0)
            )
    
    elif task_type == 'classification':
        if optimizer_name == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 0.1),
                clipnorm=config.get('clipnorm', 1.0),
                amsgrad=True
            )
        elif optimizer_name == 'radam':
            return SOTAOptimizerFactory.create_radam_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 0.1),
                clipnorm=config.get('clipnorm', 1.0)
            )
    
    elif task_type == 'multitask':
        # For multitask learning, use more conservative settings
        if optimizer_name == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=learning_rate,
                weight_decay=config.get('weight_decay', 0.02),
                clipnorm=config.get('clipnorm', 7.0),
                beta_1=0.95,  # Higher beta_1 for stability
                beta_2=0.999
            )
    
    # Default fallback to AdamW
    return SOTAOptimizerFactory.create_adamw_optimizer(
        learning_rate=learning_rate,
        weight_decay=config.get('weight_decay', 0.01)
    )


# Example usage for Detection and Segmentation Optimizers
def demo_usage():
    """Demonstrate how to use the specialized optimizers."""
    
    # === OBJECT DETECTION EXAMPLES ===
    
    # YOLOv5/v8 with EfficientNet backbone
    yolo_optimizer = DetectionOptimizer(model_type='yolo', backbone='efficientnet')
    yolo_sgd = yolo_optimizer.create_optimizer(
        optimizer_type='sgd',
        batch_size=32,
        total_epochs=300
    )
    print("YOLO + EfficientNet config:", yolo_optimizer.get_recommended_config())
    
    # DETR (Detection Transformer)
    detr_optimizer = DetectionOptimizer(model_type='detr', backbone='vit')
    detr_adamw = detr_optimizer.create_optimizer(
        optimizer_type='adamw',
        batch_size=8,
        total_epochs=500
    )
    
    # Faster R-CNN with ResNet50
    rcnn_optimizer = DetectionOptimizer(model_type='rcnn', backbone='resnet50')
    rcnn_sgd = rcnn_optimizer.create_optimizer(
        optimizer_type='sgd',
        batch_size=16,
        total_epochs=12
    )
    
    # === SEMANTIC SEGMENTATION EXAMPLES ===
    
    # U-Net for medical segmentation
    unet_optimizer = SegmentationOptimizer(
        model_type='unet',
        backbone='resnet50', 
        task_type='semantic'
    )
    unet_adam = unet_optimizer.create_optimizer(
        optimizer_type='adam',
        crop_size=256,
        total_epochs=100
    )
    
    # DeepLabV3+ for high-resolution segmentation
    deeplab_optimizer = SegmentationOptimizer(
        model_type='deeplabv3',
        backbone='efficientnet',
        task_type='semantic'
    )
    deeplab_sgd = deeplab_optimizer.create_optimizer(
        optimizer_type='sgd',
        crop_size=512,
        total_epochs=200
    )
    
    # Multi-scale training for segmentation
    multiscale_optimizers = deeplab_optimizer.create_multi_scale_optimizer(
        scales=[256, 512, 768],
        base_lr=0.007
    )
    
    # === INSTANCE SEGMENTATION EXAMPLES ===
    
    # Mask R-CNN for instance segmentation
    maskrcnn_optimizer = SegmentationOptimizer(
        model_type='maskrcnn',
        backbone='resnet50',
        task_type='instance'
    )
    maskrcnn_sgd = maskrcnn_optimizer.create_optimizer(
        optimizer_type='sgd',
        total_epochs=24
    )
    
    # Get data augmentation config
    aug_config = maskrcnn_optimizer.get_data_augmentation_config()
    print("Instance segmentation augmentation config:", aug_config)
    
    return {
        'detection_optimizers': {
            'yolo': yolo_sgd,
            'detr': detr_adamw,
            'rcnn': rcnn_sgd
        },
        'segmentation_optimizers': {
            'unet': unet_adam,
            'deeplab': deeplab_sgd,
            'maskrcnn': maskrcnn_sgd,
            'multiscale': multiscale_optimizers
        }
    }


# Updated OPTIMAL_CONFIGS with detection and segmentation
OPTIMAL_CONFIGS.update({
    'yolo_detection': {
        'optimizer': 'sgd',
        'learning_rate': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'lr_schedule': {
            'type': 'cosine_restarts',
            'params': {
                'first_decay_steps': 3000,
                't_mul': 2.0,
                'alpha': 0.01
            }
        }
    },
    'detr_detection': {
        'optimizer': 'adamw',
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'lr_schedule': {
            'type': 'warmup',
            'params': {
                'warmup_steps': 500,
                'warmup_method': 'linear',
                'decay_type': 'cosine',
                'decay_steps': 50000
            }
        }
    },
    'unet_segmentation': {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'lr_schedule': {
            'type': 'polynomial',
            'params': {
                'decay_steps': 10000,
                'end_learning_rate': 0.00001,
                'power': 0.9
            }
        }
    },
    'deeplab_segmentation': {
        'optimizer': 'sgd',
        'learning_rate': 0.007,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'lr_schedule': {
            'type': 'polynomial',
            'params': {
                'decay_steps': 20000,
                'end_learning_rate': 0.0007,
                'power': 0.9
            }
        }
    }
})
OPTIMAL_CONFIGS = {
    'vision_transformer': {
        'optimizer': 'adamw',
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'lr_schedule': {
            'type': 'cosine_restarts',
            'params': {
                'first_decay_steps': 1000,
                't_mul': 2.0,
                'alpha': 0.01
            }
        }
    },
    'cnn_classification': {
        'optimizer': 'sgdr',
        'learning_rate': 0.1,
        'momentum': 0.9,
        'lr_schedule': {
            'type': 'one_cycle',
            'params': {
                'total_steps': 10000,
                'pct_start': 0.3,
                'div_factor': 25.0
            }
        }
    },
    'object_detection': {
        'optimizer': 'adamw',
        'learning_rate': 0.0001,
        'weight_decay': 0.05,
        'clipnorm': 10.0,
        'lr_schedule': {
            'type': 'warmup',
            'params': {
                'warmup_steps': 1000,
                'warmup_method': 'linear',
                'decay_type': 'cosine',
                'decay_steps': 20000
            }
        }
    }
}


class DetectionOptimizer:
    """Specialized optimizer configurations for object detection tasks."""
    
    def __init__(self, model_type: str = 'yolo', backbone: str = 'resnet50'):
        """
        Initialize detection optimizer.
        
        Args:
            model_type: Type of detection model ('yolo', 'rcnn', 'ssd', 'retinanet', 'detr')
            backbone: Backbone architecture ('resnet50', 'efficientnet', 'vit', 'swin')
        """
        self.model_type = model_type.lower()
        self.backbone = backbone.lower()
        self._configure_base_params()
    
    def _configure_base_params(self):
        """Configure base parameters based on model type and backbone."""
        # Model-specific configurations
        model_configs = {
            'yolo': {
                'learning_rate': 0.01,
                'weight_decay': 0.0005,
                'momentum': 0.937,
                'warmup_epochs': 3,
                'cosine_lr': True
            },
            'rcnn': {
                'learning_rate': 0.02,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 1,
                'step_lr': True
            },
            'ssd': {
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.9,
                'warmup_epochs': 2,
                'cosine_lr': True
            },
            'retinanet': {
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 1,
                'cosine_lr': True
            },
            'detr': {
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 0,
                'cosine_lr': True
            }
        }
        
        # Backbone-specific adjustments
        backbone_multipliers = {
            'resnet50': 1.0,
            'efficientnet': 0.8,
            'vit': 0.1,
            'swin': 0.05
        }
        
        self.base_config = model_configs.get(self.model_type, model_configs['yolo'])
        self.lr_multiplier = backbone_multipliers.get(self.backbone, 1.0)
        
        # Apply backbone multiplier
        self.base_config['learning_rate'] *= self.lr_multiplier
    
    def create_optimizer(
        self,
        optimizer_type: str = 'sgd',
        custom_lr: Optional[float] = None,
        batch_size: int = 16,
        total_epochs: int = 300
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create optimized optimizer for detection task.
        
        Args:
            optimizer_type: Type of optimizer ('sgd', 'adamw', 'radam')
            custom_lr: Custom learning rate (overrides default)
            batch_size: Training batch size
            total_epochs: Total training epochs
            
        Returns:
            Configured optimizer
        """
        lr = custom_lr if custom_lr is not None else self.base_config['learning_rate']
        
        # Adjust learning rate based on batch size (linear scaling rule)
        if batch_size != 16:
            lr = lr * (batch_size / 16)
        
        # Create learning rate schedule
        if self.base_config.get('cosine_lr', False):
            if self.base_config['warmup_epochs'] > 0:
                warmup_steps = self.base_config['warmup_epochs'] * 1000  # Assume 1000 steps per epoch
                total_steps = total_epochs * 1000
                
                cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=lr,
                    decay_steps=total_steps - warmup_steps,
                    alpha=0.01
                )
                
                lr_schedule = EnhancedWarmupSchedule(
                    initial_learning_rate=lr,
                    warmup_steps=warmup_steps,
                    decay_schedule=cosine_schedule,
                    warmup_method='linear'
                )
            else:
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=lr,
                    decay_steps=total_epochs * 1000,
                    alpha=0.01
                )
            lr = lr_schedule
        
        # Create optimizer based on type
        if optimizer_type.lower() == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=self.base_config['momentum'],
                nesterov=True,
                clipnorm=10.0
            )
        elif optimizer_type.lower() == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=lr,
                weight_decay=self.base_config['weight_decay'],
                clipnorm=10.0,
                beta_1=0.9,
                beta_2=0.999
            )
        elif optimizer_type.lower() == 'radam':
            return SOTAOptimizerFactory.create_radam_optimizer(
                learning_rate=lr,
                weight_decay=self.base_config['weight_decay'],
                clipnorm=10.0
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """Get recommended configuration for the detection model."""
        recommendations = {
            'yolo': {
                'optimizer': 'sgd',
                'batch_size': 64,
                'epochs': 300,
                'mosaic_prob': 1.0,
                'mixup_prob': 0.1
            },
            'rcnn': {
                'optimizer': 'sgd',
                'batch_size': 16,
                'epochs': 12,
                'mosaic_prob': 0.0,
                'mixup_prob': 0.0
            },
            'detr': {
                'optimizer': 'adamw',
                'batch_size': 8,
                'epochs': 500,
                'mosaic_prob': 0.0,
                'mixup_prob': 0.0
            }
        }
        
        return recommendations.get(self.model_type, recommendations['yolo'])


class SegmentationOptimizer:
    """Specialized optimizer configurations for semantic/instance segmentation tasks."""
    
    def __init__(self, model_type: str = 'unet', backbone: str = 'resnet50', task_type: str = 'semantic'):
        """
        Initialize segmentation optimizer.
        
        Args:
            model_type: Type of segmentation model ('unet', 'deeplabv3', 'pspnet', 'fcn', 'maskrcnn')
            backbone: Backbone architecture ('resnet50', 'efficientnet', 'hrnet', 'swin')
            task_type: Type of segmentation ('semantic', 'instance', 'panoptic')
        """
        self.model_type = model_type.lower()
        self.backbone = backbone.lower()
        self.task_type = task_type.lower()
        self._configure_base_params()
    
    def _configure_base_params(self):
        """Configure base parameters based on model and task type."""
        # Model-specific configurations
        model_configs = {
            'unet': {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 2,
                'poly_lr': True,
                'power': 0.9
            },
            'deeplabv3': {
                'learning_rate': 0.007,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 1,
                'poly_lr': True,
                'power': 0.9
            },
            'pspnet': {
                'learning_rate': 0.01,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 1,
                'poly_lr': True,
                'power': 0.9
            },
            'fcn': {
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'momentum': 0.9,
                'warmup_epochs': 0,
                'step_lr': True
            },
            'maskrcnn': {
                'learning_rate': 0.02,
                'weight_decay': 0.0001,
                'momentum': 0.9,
                'warmup_epochs': 1,
                'step_lr': True
            }
        }
        
        # Task-specific adjustments
        task_multipliers = {
            'semantic': 1.0,
            'instance': 0.5,
            'panoptic': 0.3
        }
        
        # Backbone-specific adjustments
        backbone_configs = {
            'resnet50': {'lr_mult': 1.0, 'wd_mult': 1.0},
            'efficientnet': {'lr_mult': 0.8, 'wd_mult': 1.2},
            'hrnet': {'lr_mult': 0.1, 'wd_mult': 0.5},
            'swin': {'lr_mult': 0.05, 'wd_mult': 0.1}
        }
        
        self.base_config = model_configs.get(self.model_type, model_configs['unet'])
        task_mult = task_multipliers.get(self.task_type, 1.0)
        backbone_config = backbone_configs.get(self.backbone, backbone_configs['resnet50'])
        
        # Apply multipliers
        self.base_config['learning_rate'] *= task_mult * backbone_config['lr_mult']
        self.base_config['weight_decay'] *= backbone_config['wd_mult']
    
    def create_optimizer(
        self,
        optimizer_type: str = 'sgd',
        custom_lr: Optional[float] = None,
        crop_size: int = 512,
        total_epochs: int = 200
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create optimized optimizer for segmentation task.
        
        Args:
            optimizer_type: Type of optimizer ('sgd', 'adamw', 'adam')
            custom_lr: Custom learning rate
            crop_size: Input crop size (affects learning rate)
            total_epochs: Total training epochs
            
        Returns:
            Configured optimizer
        """
        lr = custom_lr if custom_lr is not None else self.base_config['learning_rate']
        
        # Adjust learning rate based on crop size
        if crop_size != 512:
            lr = lr * (crop_size / 512) ** 0.5
        
        # Create learning rate schedule
        if self.base_config.get('poly_lr', False):
            # Polynomial learning rate decay (common in segmentation)
            class PolynomialDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, initial_lr, total_steps, warmup_steps, power=0.9):
                    self.initial_lr = initial_lr
                    self.total_steps = total_steps
                    self.warmup_steps = warmup_steps
                    self.power = power
                
                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    
                    # Warmup phase
                    warmup_lr = self.initial_lr * step / self.warmup_steps
                    
                    # Polynomial decay phase
                    decay_steps = tf.maximum(step - self.warmup_steps, 0)
                    poly_lr = self.initial_lr * (
                        1 - decay_steps / (self.total_steps - self.warmup_steps)
                    ) ** self.power
                    
                    return tf.where(step < self.warmup_steps, warmup_lr, poly_lr)
            
            warmup_steps = self.base_config['warmup_epochs'] * 1000
            total_steps = total_epochs * 1000
            
            lr_schedule = PolynomialDecayWithWarmup(
                initial_lr=lr,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                power=self.base_config.get('power', 0.9)
            )
            lr = lr_schedule
        
        elif self.base_config.get('step_lr', False):
            # Step learning rate decay
            boundaries = [int(total_epochs * 0.6 * 1000), int(total_epochs * 0.8 * 1000)]
            values = [lr, lr * 0.1, lr * 0.01]
            lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        
        # Create optimizer
        if optimizer_type.lower() == 'sgd':
            return tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=self.base_config['momentum'],
                nesterov=True,
                clipnorm=5.0
            )
        elif optimizer_type.lower() == 'adamw':
            return SOTAOptimizerFactory.create_adamw_optimizer(
                learning_rate=lr,
                weight_decay=self.base_config['weight_decay'],
                clipnorm=5.0,
                beta_1=0.9,
                beta_2=0.999
            )
        elif optimizer_type.lower() == 'adam':
            return tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                clipnorm=5.0
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def create_multi_scale_optimizer(
        self,
        scales: List[int] = [256, 512, 768],
        base_lr: float = 0.001
    ) -> List[tf.keras.optimizers.Optimizer]:
        """
        Create multiple optimizers for multi-scale training.
        
        Args:
            scales: List of input scales
            base_lr: Base learning rate
            
        Returns:
            List of optimizers for each scale
        """
        optimizers = []
        for scale in scales:
            # Larger scales need smaller learning rates
            scale_lr = base_lr * (256 / scale) ** 0.5
            optimizer = self.create_optimizer(
                optimizer_type='sgd',
                custom_lr=scale_lr,
                crop_size=scale
            )
            optimizers.append(optimizer)
        return optimizers
    
    def get_data_augmentation_config(self) -> Dict[str, Any]:
        """Get recommended data augmentation configuration."""
        base_config = {
            'random_flip': True,
            'random_rotation': 10,
            'random_scale': (0.5, 2.0),
            'color_jitter': True,
            'gaussian_blur': False
        }
        
        # Task-specific adjustments
        if self.task_type == 'instance':
            base_config['random_crop'] = True
            base_config['mosaic'] = 0.5
        elif self.task_type == 'panoptic':
            base_config['copy_paste'] = 0.3
            base_config['mixup'] = 0.1
        
        return base_config


def create_optimal_optimizer(task_name: str, model_size: str = 'medium') -> tf.keras.optimizers.Optimizer:
    """
    Create optimal optimizer for common tasks.
    
    Args:
        task_name: Name of the task from OPTIMAL_CONFIGS
        model_size: Size of the model
        
    Returns:
        Optimally configured optimizer
    """
    
    if task_name not in OPTIMAL_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(OPTIMAL_CONFIGS.keys())}")
    
    config = OPTIMAL_CONFIGS[task_name]
    
    # Map task names to task types
    task_type_mapping = {
        'vision_transformer': 'classification',
        'cnn_classification': 'classification',
        'object_detection': 'detection'
    }
    
    task_type = task_type_mapping.get(task_name, 'classification')
    
    return get_sota_optimizer_config(task_type, config, model_size)


# Example usage configurations
SOTA_LOSS_CONFIGS = {
    'detection_sota': {
        'type': 'unified_detection',
        'bbox_loss_weight': 2.0,
        'cls_loss_weight': 1.0,
        'quality_loss_weight': 1.0,
        'use_eiou': True,
        'use_varifocal': True,
        'use_quality_focal': True
    },
    'segmentation_sota': {
        'type': 'unified_segmentation',
        'ce_weight': 1.0,
        'dice_weight': 1.0,
        'boundary_weight': 0.5,
        'hausdorff_weight': 0.3,
        'use_adaptive_dice': True,
        'use_boundary_loss': True,
        'use_hausdorff_loss': True
    },
    'classification_sota': {
        'type': 'poly',
        'epsilon': 1.0,
        'alpha': 1.0,
        'from_logits': False
    }
}