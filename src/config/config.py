# config/config.py
import os
from pathlib import Path

class Config:
    """Global configuration for the Oxford Pet Lab project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    PLOTS_DIR = RESULTS_DIR / "plots"
    PREDICTIONS_DIR = RESULTS_DIR / "predictions"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, PREDICTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration
    DATASET_NAME = "oxford_iiit_pet:4.0.0"
    DATASET_VERSION = "4.0.0"
    
    # Data preprocessing
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # Training configuration
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    # Model configuration
    NUM_CLASSES = 37  # 37 different pet breeds
    LEARNING_RATE = 1e-4
    
    # Bounding box configuration
    BBOX_FORMAT = "xyxy"  # x1, y1, x2, y2 format
    
    # Augmentation configuration
    ENABLE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.2
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Logging
    LOG_LEVEL = "INFO"
    TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
    
    # Model checkpointing
    CHECKPOINT_MONITOR = "val_loss"
    CHECKPOINT_MODE = "min"
    SAVE_BEST_ONLY = True
    
    # Visualization
    PLOT_STYLE = "seaborn-v0_8"
    FIGURE_SIZE = (12, 8)
    DPI = 100

# config/model_configs.py
class ModelConfigs:
    """Model-specific configurations."""
    
    # Detection model configurations
    DETECTION_MODELS = {
        "resnet50": {
            "backbone": "ResNet50",
            "input_shape": (224, 224, 3),
            "pretrained_weights": "imagenet",
            "freeze_backbone": False,
            "detection_head_units": [256, 128, 64],
            "bbox_output_units": 4,  # x1, y1, x2, y2
            "class_output_units": 37,
        },
        "mobilenetv2": {
            "backbone": "MobileNetV2",
            "input_shape": (224, 224, 3),
            "pretrained_weights": "imagenet",
            "freeze_backbone": False,
            "detection_head_units": [256, 128, 64],
            "bbox_output_units": 4,
            "class_output_units": 37,
        }
    }
    
    # Segmentation model configurations
    SEGMENTATION_MODELS = {
        "unet_resnet50": {
            "backbone": "ResNet50",
            "input_shape": (224, 224, 3),
            "pretrained_weights": "imagenet",
            "freeze_backbone": False,
            "decoder_filters": [512, 256, 128, 64, 32],
            "num_classes": 3,  # background, foreground, unknown
        },
        "unet_mobilenetv2": {
            "backbone": "MobileNetV2",
            "input_shape": (224, 224, 3),
            "pretrained_weights": "imagenet",
            "freeze_backbone": False,
            "decoder_filters": [512, 256, 128, 64, 32],
            "num_classes": 3,
        }
    }
    
    # Multitask model configuration
    MULTITASK_MODEL = {
        "backbone": "ResNet50",
        "input_shape": (224, 224, 3),
        "pretrained_weights": "imagenet",
        "freeze_backbone": False,
        "shared_features": 512,
        "detection_head_units": [256, 128, 64],
        "segmentation_decoder_filters": [512, 256, 128, 64, 32],
        "bbox_output_units": 4,
        "class_output_units": 37,
        "seg_output_units": 3,
        "loss_weights": {
            "detection": 1.0,
            "segmentation": 1.0,
            "bbox": 1.0,
            "classification": 1.0
        }
    }
    
    # Loss function configurations
    LOSS_CONFIGS = {
        "detection": {
            "bbox_loss": "smooth_l1",
            "classification_loss": "sparse_categorical_crossentropy",
            "bbox_loss_weight": 1.0,
            "classification_loss_weight": 1.0,
        },
        "segmentation": {
            "loss": "sparse_categorical_crossentropy",
            "metrics": ["accuracy", "iou"],
        },
        "multitask": {
            "detection_weight": 1.0,
            "segmentation_weight": 1.0,
        }
    }
    
    # Optimizer configurations
    OPTIMIZER_CONFIGS = {
        "adam": {
            "learning_rate": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-7,
        },
        "sgd": {
            "learning_rate": 1e-3,
            "momentum": 0.9,
            "nesterov": True,
        }
    }