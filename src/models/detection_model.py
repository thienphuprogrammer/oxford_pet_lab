import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from src.config.model_configs import ModelConfigs
from src.config.config import Config
from src.models.base_model import BaseDetectionModel

class SimpleDetectionModel(BaseDetectionModel):
    """Simple CNN-based detection model without pretrained backbone"""

    def __init__(
        self,
        num_classes: int,
        config: Config = None,
        models_config: ModelConfigs = None,
        **kwargs,
    ):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
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
        config: Config = None,
        models_config: ModelConfigs = None,
        pretrained: bool = True,
        **kwargs,
    ):
        """Initialize detection model with specified backbone."""
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        self.backbone_name = backbone_name

        self.pretrained = pretrained

        self.backbone = self._build_backbone()
        self.detection_head = self._build_detection_head()
        self.classification_head = self._build_classification_head()

    def _build_backbone(self):
        """Build the backbone of the model."""
        params_model = self.models_config.DETECTION_MODELS[self.backbone_name.lower()]
        input_shape = params_model['input_shape']
        pretrained_weights = params_model['pretrained_weights'] if self.pretrained else None
        freeze_backbone = params_model['freeze_backbone']

        
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
            for layer in backbone.layers[:-10]:
                layer.trainable = False

        return backbone 
    
    def _build_detection_head(self):
        """Build the detection head of the model."""
        params_model = self.models_config.DETECTION_MODELS[self.backbone_name.lower()]
        detection_head_units = params_model['detection_head_units']
        bbox_output_units = params_model['bbox_output_units']

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
            layers.GlobalAveragePooling2D(),
            layers.Dense(bbox_output_units, name='bbox_output'),
        ])
        
        return detection_head
    
    def _build_classification_head(self):
        """Build classification head."""
        classification_head = keras.Sequential([
            layers.Conv2D(512, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            *[
                (
                    layers.Conv2D(units, (3, 3), padding='same'),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                )
                for units in self.models_config.DETECTION_MODELS[self.backbone_name.lower()]['detection_head_units']
            ],
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.num_classes, activation='softmax', name='class_output')
        ])

        return classification_head



    def call(self, inputs, training=None, mask=None):
        """Forward pass of the model."""
        features = self.backbone(inputs, training=training)
        
        bbox_output = self.detection_head(features)
        class_output = self.classification_head(features)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output
        }


class YOLOv3Model(DetectionModel):
    """Simple YOLO-like detection model"""
    
    def __init__(self, num_classes: int, config: Config = None, models_config: ModelConfigs = None, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        
        # Darknet-like backbone
        self.backbone = self._build_darknet_backbone()
        
        # YOLO detection head
        self.detection_layers = self._build_detection_layers()
        
    def _build_darknet_backbone(self):
        """Build Darknet-like backbone"""
        backbone = keras.Sequential([
            # Initial conv
            layers.Conv2D(32, 3, padding='same', activation='leaky_relu'),
            layers.BatchNormalization(),
            
            # Downsample blocks
            self._conv_block(64, 2),
            self._conv_block(128, 2),
            self._conv_block(256, 2),
            self._conv_block(512, 2),
            self._conv_block(1024, 2),
        ])
        
        return backbone
    
    def _conv_block(self, filters: int, blocks: int):
        """Convolutional block"""
        block = keras.Sequential()
        
        # Downsample
        block.add(layers.Conv2D(filters, 3, strides=2, padding='same', activation='leaky_relu'))
        block.add(layers.BatchNormalization())
        
        # Residual blocks
        for _ in range(blocks):
            block.add(layers.Conv2D(filters // 2, 1, padding='same', activation='leaky_relu'))
            block.add(layers.BatchNormalization())
            block.add(layers.Conv2D(filters, 3, padding='same', activation='leaky_relu'))
            block.add(layers.BatchNormalization())
        
        return block
    
    def _build_detection_layers(self):
        """Build YOLO detection layers"""
        # Number of anchors per scale
        num_anchors = len(self.config.ASPECT_RATIOS)
        
        # Output channels: (bbox + objectness + classes) * anchors
        output_channels = (4 + 1 + self.num_classes) * num_anchors
        
        detection_layers = keras.Sequential([
            layers.Conv2D(512, 3, padding='same', activation='leaky_relu'),
            layers.BatchNormalization(),
            layers.Conv2D(output_channels, 1, padding='same'),
        ])
        
        return detection_layers
    
    def call(self, inputs, training=None):
        """Forward pass"""
        features = self.backbone(inputs, training=training)
        detections = self.detection_layers(features, training=training)
        
        # Reshape output
        batch_size = tf.shape(inputs)[0]
        grid_size = tf.shape(detections)[1]
        num_anchors = len(self.config.ASPECT_RATIOS)
        
        detections = tf.reshape(
            detections,
            [batch_size, grid_size, grid_size, num_anchors, 4 + 1 + self.num_classes]
        )
        
        # Split outputs
        bbox_pred = detections[..., :4]
        objectness = detections[..., 4:5]
        class_pred = detections[..., 5:]
        
        return {
            'bbox_output': bbox_pred,
            'objectness': objectness,
            'class_output': class_pred
        }