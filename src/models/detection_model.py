import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from src.config.model_configs import ModelConfigs
from src.models.base_model import BaseDetectionModel

class SimpleDetectionModel(BaseDetectionModel):
    """Simple CNN-based detection model without pretrained backbone"""

    def __init__(
        self,
        num_classes: int,
        config: ModelConfigs = None,
        **kwargs,
    ):
        super().__init__(num_classes, **kwargs)
        self.config = config
        self.backbone = self._build_backbone()
        self.head = self._build_detection_head()
        self.classification_head = self._build_classification_head()
        

    def _build_backbone(self):  
        """Build the backbone of the model."""
        backbone = keras.Sequential([
            # Block 1
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Block 2
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Block 3
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Block 4
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            
            # Block 5
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(512, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])
        
        return backbone
    

    def _build_detection_head(self):
        """Build the detection head of the model."""
        detection_head = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4, name='bbox_output')  # x_min, y_min, x_max, y_max
        ])
        
        return detection_head
    
    def _build_classification_head(self):
        """Build the classification head of the model."""
        classification_head = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax', name='class_output')
        ])
        
        return classification_head


class PretrainedDetectionModel(BaseDetectionModel):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'resnet50',
        config: ModelConfigs = None,
        pretrained: bool = True,
        **kwargs,
    ):
        """Initialize detection model with specified backbone."""
        super().__init__(num_classes, **kwargs)
        self.config = config
        self.backbone_name = backbone_name

        self.pretrained = pretrained

        self.backbone = self._build_backbone()
        self.head = self._build_detection_head()
        self.classification_head = self._build_classification_head()

    def _build_backbone(self):
        """Build the backbone of the model."""
        params_model = self.config.DETECTION_MODELS[self.backbone_name.lower()]
        input_shape = params_model['input_shape']
        pretrained_weights = params_model['pretrained_weights']
        freeze_backbone = params_model['freeze_backbone']
        detection_head_units = params_model['detection_head_units']
        bbox_output_units = params_model['bbox_output_units']
        class_output_units = params_model['class_output_units']
        
        if self.backbone_name.lower() == 'resnet50':
            backbone = ResNet50(
                include_top=False,
                weights=pretrained_weights,
                input_shape=input_shape,
            )
        elif self.backbone_name.lower() == 'mobilenetv2':
            backbone = MobileNetV2(
                include_top=False,
                weights=pretrained_weights,
                input_shape=input_shape,
            )
        elif self.backbone_name.lower() == 'efficientnetb0':
            backbone = EfficientNetB0(
                include_top=False,
                weights=pretrained_weights,
                input_shape=input_shape,
            )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        if freeze_backbone:
            for layer in backbone.layers:
                layer.trainable = False

        return backbone 
    
    def _build_detection_head(self):
        """Build the detection head of the model."""
        params_model = self.config.DETECTION_MODELS[self.backbone_name.lower()]
        detection_head_units = params_model['detection_head_units']
        bbox_output_units = params_model['bbox_output_units']

        # Predict Sequence
        predict_sequence = [
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]

        detection_head = keras.Sequential([
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Bounding box prediction
            *[
                (
                    layers.Conv2D(units, (3, 3), padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                )
                for units in detection_head_units
            ],
            # Classification prediction
            layers.GlobalAveragePooling2D(),
            layers.Dense(bbox_output_units, name='bbox_output'),

            # Classification prediction
        ])
        
        return detection_head
    
    def _build_detection_head(self, x, num_classes):
        """Build detection head for bounding box and classification."""
        # Shared features
        x = layers.Conv2D(512, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Bounding box prediction
        bbox_head = x
        for units in self.config.DETECTION_MODELS[self.model_name]['detection_head_units']:
            bbox_head = layers.Conv2D(units, (3, 3), padding='same')(bbox_head)
            bbox_head = layers.BatchNormalization()(bbox_head)
            bbox_head = layers.ReLU()(bbox_head)
        bbox_head = layers.GlobalAveragePooling2D()(bbox_head)
        bbox_output = layers.Dense(4, name='bbox_output')(bbox_head)
        
        # Classification prediction
        class_head = x
        for units in self.config.DETECTION_MODELS[self.model_name]['detection_head_units']:
            class_head = layers.Conv2D(units, (3, 3), padding='same')(class_head)
            class_head = layers.BatchNormalization()(class_head)
            class_head = layers.ReLU()(class_head)
        class_head = layers.GlobalAveragePooling2D()(class_head)
        class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(class_head)
        
        return bbox_output, class_output

    def _build_model(self):
        """Build the complete detection model."""
        # Get backbone configuration
        backbone_config = self.config.DETECTION_MODELS[self.model_name]
        
        # Load backbone
        if self.model_name == 'resnet50':
            backbone = ResNet50(
                include_top=False,
                weights='imagenet' if self.pretrained else None,
                input_shape=backbone_config['input_shape'],
            )
        elif self.model_name == 'mobilenetv2':
            backbone = MobileNetV2(
                include_top=False,
                weights='imagenet' if self.pretrained else None,
                input_shape=backbone_config['input_shape'],
            )
        
        # Freeze backbone if specified
        if backbone_config['freeze_backbone']:
            for layer in backbone.layers:
                layer.trainable = False
        
        # Get output from backbone
        x = backbone.output
        
        # Add detection head
        bbox_output, class_output = self._build_detection_head(x, backbone_config['class_output_units'])
        
        # Create model
        model = models.Model(
            inputs=backbone.input,
            outputs=[bbox_output, class_output],
            name=f'detection_{self.model_name}'
        )
        
        return model
