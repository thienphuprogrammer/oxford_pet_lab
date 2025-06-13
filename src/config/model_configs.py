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