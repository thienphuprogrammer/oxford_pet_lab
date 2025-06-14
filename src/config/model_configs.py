# config/model_configs.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfigs:
    """Model-specific configurations."""
    
    # Detection model configurations
    DETECTION_MODELS: Dict[str, Dict[str, Any]] = None
    
    # Segmentation model configurations
    SEGMENTATION_MODELS: Dict[str, Dict[str, Any]] = None
    
    # Multitask model configurations
    MULTITASK_MODELS: Dict[str, Dict[str, Any]] = None

    # Loss function configurations
    LOSS_CONFIGS: Dict[str, Dict[str, Any]] = None

    # Optimizer configurations
    OPTIMIZER_CONFIGS: Dict[str, Dict[str, Any]] = None
    
    
    def __post_init__(self):
        if self.DETECTION_MODELS is None:
            self.DETECTION_MODELS = {
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
                },
                "efficientnetb0": {
                    "backbone": "EfficientNetB0",
                    "input_shape": (224, 224, 3),
                    "pretrained_weights": "imagenet",
                    "freeze_backbone": False,
                    "detection_head_units": [256, 128, 64],
                    "bbox_output_units": 4,
                    "class_output_units": 37,
                }
            }
    
        # Segmentation model configurations
        if self.SEGMENTATION_MODELS is None:
            self.SEGMENTATION_MODELS = {
                    "simple_unet": {
                        "backbone": "ResNet50",
                        "input_shape": (224, 224, 3),
                        "pretrained_weights": "imagenet",
                        "freeze_backbone": False,
                        "encoder_filters": [64, 128, 256, 512, 1024],
                        "decoder_filters": [512, 256, 128, 64, 32],
                        "num_classes": 3,  # background, foreground, unknown
                    },
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
                    },
                    "deeplabv3plus": {
                        "backbone": "ResNet50",
                        "input_shape": (224, 224, 3),
                        "pretrained_weights": "imagenet",
                        "freeze_backbone": False,
                        "decoder_filters": [512, 256, 128, 64, 32],
                        "num_classes": 3,
                    }
                }
        
        # Multitask model configuration
        if self.MULTITASK_MODELS is None:
            self.MULTITASK_MODELS = {
                "resnet50": {
                    "input_shape": (224, 224, 3),
                    "feature_dim": 2048,
                    "pretrained_weights": "imagenet",
                    "freeze_backbone": False,
                    "shared_features": 512,
                    "detection_head_units": [512, 256, 128, 64],
                    "detection_head_dropout": 0.5,
                    "classification_head_units": [512, 256, 128, 64],
                    "classification_head_dropout": 0.5,
                    "segmentation_head_units": [512, 256, 128, 64],
                    "segmentation_head_dropout": 0.3,
                    "bbox_output_units": 4,
                    "class_output_units": 37,
                    "seg_output_units": 3,
                    "loss_weights": {
                        "detection": 1.0,
                        "segmentation": 1.0,
                        "bbox": 1.0,
                        "classification": 1.0
                        }
                },
                "efficientnetb0": {
                    "input_shape": (224, 224, 3),
                    "feature_dim": 1280,
                    "pretrained_weights": "imagenet",
                    "freeze_backbone": False,
                    "shared_features": 512,
                    "detection_head_units": [512, 256, 128, 64],
                    "detection_head_dropout": 0.5,
                    "classification_head_units": [512, 256, 128, 64],
                    "classification_head_dropout": 0.5,
                    "segmentation_head_units": [512, 256, 128, 64],
                    "segmentation_head_dropout": 0.3,
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
            }

        
        
        # Loss function configurations
        if self.LOSS_CONFIGS is None:
            self.LOSS_CONFIGS = {
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
        if self.OPTIMIZER_CONFIGS is None:
            self.OPTIMIZER_CONFIGS = {
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