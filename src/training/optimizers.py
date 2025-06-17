import tensorflow as tf

class SOTAOptimizers:
    """SOTA Optimizers sử dụng built-in optimizers"""
    
    # ========== SEGMENTATION OPTIMIZERS ==========
    @staticmethod
    def get_segmentation_optimizer(learning_rate=1e-3, model_size='medium'):
        """Best optimizers cho segmentation tasks"""
        
        configs = {
            'small': {
                'optimizer': tf.keras.optimizers.Adam,
                'lr': 1e-3,
                'params': {'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-7}
            },
            'medium': {
                'optimizer': tf.keras.optimizers.AdamW,
                'lr': 1e-3, 
                'params': {'weight_decay': 1e-4, 'beta_1': 0.9, 'beta_2': 0.999}
            },
            'large': {
                'optimizer': tf.keras.optimizers.Lamb,
                'lr': 1e-3,
                'params': {'weight_decay_rate': 0.01, 'beta_1': 0.9, 'beta_2': 0.999}
            }
        }
        
        config = configs[model_size]
        return config['optimizer'](
            learning_rate=learning_rate or config['lr'],
            **config['params']
        )
    
    # ========== OBJECT DETECTION OPTIMIZERS ==========
    @staticmethod
    def get_detection_optimizer(learning_rate=1e-4, framework='yolo'):
        """Best optimizers cho object detection"""
        
        frameworks = {
            'yolo': lambda lr: tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.937,
                nesterov=True
            ),
            
            'rcnn': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            'ssd': lambda lr: tf.keras.optimizers.RectifiedAdam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999,
                weight_decay=1e-4
            ),
            
            'efficientdet': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=4e-5,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            'detr': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999
            )
        }
        
        return frameworks[framework](learning_rate)
    
    # ========== CLASSIFICATION OPTIMIZERS ==========
    @staticmethod
    def get_classification_optimizer(learning_rate=1e-3, architecture='resnet'):
        """Best optimizers cho classification"""
        
        architectures = {
            'resnet': lambda lr: tf.keras.optimizers.SGD(
                learning_rate=lr,
                momentum=0.9,
                nesterov=True
            ),
            
            'efficientnet': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=1e-5,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            'vit': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=0.3,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            'mobilenet': lambda lr: tf.keras.optimizers.RectifiedAdam(
                learning_rate=lr,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            'convnext': lambda lr: tf.keras.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=0.05,
                beta_1=0.9,
                beta_2=0.999
            )
        }
        
        return architectures[architecture](learning_rate)
    
    # ========== MULTITASK OPTIMIZERS ==========
    @staticmethod
    def get_multitask_optimizer(learning_rate=1e-3, strategy='adaptive'):
        """Optimizers cho multitask learning"""
        
        strategies = {
            # Adaptive learning rates
            'adaptive': tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            # Gradient centralization
            'centralized': tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            ),
            
            # Large batch training
            'lamb': tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-3,
                beta_1=0.9,
                beta_2=0.999
            ),
            
            # Lookahead wrapper
            'lookahead': 
                tf.keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=1e-4
                ),
        }
        
        return strategies[strategy]
    
    # ========== ADVANCED OPTIMIZERS ==========
    @staticmethod
    def get_advanced_optimizer(optimizer_name='adamw', learning_rate=1e-3, **kwargs):
        """Advanced optimizers với custom configs"""
        
        optimizers = {
            # Standard with improvements
            'adamw': lambda: tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=kwargs.get('weight_decay', 1e-4),
                **{k: v for k, v in kwargs.items() if k != 'weight_decay'}
            ),
            
            # Rectified Adam
            'radam': lambda: tf.keras.optimizers.RectifiedAdam(
                learning_rate=learning_rate,
                **kwargs
            ),
            
            # LAMB for large batches
            'lamb': lambda: tf.keras.optimizers.LAMB(
                learning_rate=learning_rate,
                weight_decay_rate=kwargs.get('weight_decay_rate', 0.01),
                **{k: v for k, v in kwargs.items() if k != 'weight_decay_rate'}
            ),
            
            # Lookahead wrapper
            'lookahead_adam': lambda: tf.keras.optimizers.Lookahead(
                tf.keras.optimizers.Adam(learning_rate=learning_rate),
                sync_period=kwargs.get('sync_period', 5),
                slow_step_size=kwargs.get('slow_step_size', 0.5)
            ),
            
            # Stochastic Weight Averaging
            'swa': lambda: tf.keras.optimizers.SWA(
                tf.keras.optimizers.Adam(learning_rate=learning_rate),
                start_averaging=kwargs.get('start_averaging', 0),
                average_period=kwargs.get('average_period', 10)
            ),
            
            # Gradient Centralization
            'gc_sgd': lambda: tf.keras.optimizers.CentralizedGradientDescent(
                learning_rate=learning_rate,
                momentum=kwargs.get('momentum', 0.9),
                use_nesterov=kwargs.get('use_nesterov', True)
            )
        }
        
        return optimizers[optimizer_name]()

# ==========================================================
# QUICK SETUP FUNCTIONS - CỰC KỲ ĐƠN GIẢN
# ==========================================================

def get_segmentation_setup(model_size='medium', lr=1e-3):
    """Setup optimizer cho segmentation"""
    return {
        'optimizer': SOTAOptimizers.get_segmentation_optimizer(lr, model_size),
        'recommended_lr': lr,
        'scheduler': 'cosine'  # Recommend dùng với cosine scheduler
    }

def get_detection_setup(framework='yolo', lr=1e-4):
    """Setup optimizer cho detection"""
    return {
        'optimizer': SOTAOptimizers.get_detection_optimizer(lr, framework),
        'recommended_lr': lr,
        'scheduler': 'plateau'  # Recommend dùng với plateau scheduler
    }

def get_classification_setup(architecture='resnet', lr=1e-3):
    """Setup optimizer cho classification"""
    return {
        'optimizer': SOTAOptimizers.get_classification_optimizer(lr, architecture),
        'recommended_lr': lr,
        'scheduler': 'step'  # Recommend dùng với step scheduler
    }

def get_multitask_setup(strategy='adaptive', lr=1e-3):
    """Setup optimizer cho multitask"""
    return {
        'optimizer': SOTAOptimizers.get_multitask_optimizer(lr, strategy),
        'recommended_lr': lr,
        'scheduler': 'cosine'  # Recommend dùng với cosine scheduler
    }
