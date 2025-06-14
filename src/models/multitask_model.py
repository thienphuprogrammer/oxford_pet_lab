import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from typing import Dict, Tuple, Optional

from src.models.base_model import BaseMultitaskModel
from models.detecttion.detection_model import PretrainedDetectionModel, SimpleDetectionModel
from models.segmentation.segmentation_model import PretrainedUNet, SimpleUNet, DeepLabV3Plus
from src.training.losses import DetectionLoss, MultiTaskLoss
from src.config import ModelConfigs, Config

class MultitaskModel(BaseMultitaskModel):
    """
    Multitask model for simultaneous object detection and semantic segmentation
    """
    
    def __init__(
        self,
        num_detection_classes: int,
        num_segmentation_classes: int,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        config: Optional[Config] = None,
        models_config: Optional[ModelConfigs] = None,
        **kwargs
    ):
        super().__init__(num_detection_classes, num_segmentation_classes, **kwargs)
        
        # Configs
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        
        # Model parameters
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        # Build model components
        self._build_backbone()
        self._build_detection_head()
        self._build_classification_head()
        self._build_segmentation_head()
        
        # Build the model
        self._build_model()
    
    def _build_backbone(self):
        """Build the backbone network"""
        params = self.models_config.MULTITASK_MODELS[self.backbone_name.lower()]
        pretrained_weights = params['pretrained_weights'] if self.pretrained else None
        freeze_backbone = params['freeze_backbone']
        input_shape = params['input_shape']
        feature_dim = params['feature_dim']
        
        if self.backbone_name == 'resnet50':
            self.backbone = ResNet50(
                weights=pretrained_weights,
                include_top=False,
                input_shape=input_shape
            )
            self.feature_dim = feature_dim
            
        elif self.backbone_name == 'efficientnetb0':
            self.backbone = EfficientNetB0(
                weights=pretrained_weights,
                include_top=False,
                input_shape=input_shape
            )
            self.feature_dim = feature_dim
            
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.backbone.trainable = False
    
    @staticmethod
    def _dense_block(units: list[int], dropout: float) -> list[layers.Layer]:
        """Utility to create Dense+Dropout blocks."""
        block: list[layers.Layer] = []
        for n in units:
            block.extend([layers.Dense(n, activation="relu"),
                          layers.Dropout(dropout)])
        return block

    def _build_detection_head(self):
        """Build the detection head for bounding box regression"""
        params = self.models_config.MULTITASK_MODELS[self.backbone_name.lower()]
        detection_head_units = params['detection_head_units']
        detection_head_dropout = params['detection_head_dropout']
        bbox_output_units = params['bbox_output_units']

        self.detection_head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            *[
                (
                    layers.Dense(units, activation='relu'),
                    layers.Dropout(detection_head_dropout)
                ) for units in detection_head_units
            ],
            layers.Dense(bbox_output_units, activation='linear', name='bbox_output')  # [x, y, w, h]
        ], name='detection_head')
    
    def _build_classification_head(self):
        """Build the classification head for object detection"""
        params = self.models_config.MULTITASK_MODELS[self.backbone_name.lower()]
        classification_head_units = params['classification_head_units']
        classification_head_dropout = params['classification_head_dropout']
        class_output_units = params['class_output_units']
        
        self.classification_head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            *[
                (
                    layers.Dense(units, activation='relu'),
                    layers.Dropout(classification_head_dropout)
                ) for units in classification_head_units
            ],
            layers.Dense(class_output_units, activation='softmax', name='class_output')
        ], name='classification_head')
    
    def _build_segmentation_head(self):
        """Build the segmentation head for semantic segmentation"""
        params = self.models_config.MULTITASK_MODELS[self.backbone_name.lower()]
        segmentation_head_units = params['segmentation_head_units']
        segmentation_head_dropout = params['segmentation_head_dropout']
        seg_output_units = params['seg_output_units']
        decoder_filters = params['decoder_filters']
        # Decoder for upsampling
        self.segmentation_head = tf.keras.Sequential([
            *[
                (
                    layers.Conv2D(units, 3, padding='same', activation='relu'),
                    layers.BatchNormalization(),
                    layers.UpSampling2D(size=(2, 2))
                ) for units in decoder_filters
            ],
            # Upsampling to original image size
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.UpSampling2D(size=(2, 2)),
            
            # Final segmentation output
            layers.Conv2D(seg_output_units, 1, activation='softmax', name='segmentation_output')
        ], name='segmentation_head')
    
    def _build_model(self):
        """Build the complete multitask model"""
        # Create input layer
        self.input_layer = layers.Input(shape=self.input_shape)
        
        # Forward pass through backbone
        backbone_features = self.backbone(self.input_layer)
        
        # Detection outputs
        bbox_output = self.detection_head(backbone_features)
        class_output = self.classification_head(backbone_features)
        
        # Segmentation output
        segmentation_output = self.segmentation_head(backbone_features)
        
        # Create the model
        self.model = Model(
            inputs=self.input_layer,
            outputs={
                'bbox_output': bbox_output,
                'class_output': class_output,
                'segmentation_output': segmentation_output
            }
        )
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass for multitask models"""
        return self.model(inputs, training=training)
    
    def compile_model(
        self,
        detection_loss_weight: float = 1.0,
        classification_loss_weight: float = 1.0,
        segmentation_loss_weight: float = 1.0,
        optimizer: str = 'adam',
        learning_rate: float = 0.001
    ):
        """Compile the multitask model with appropriate losses"""
        
        # Define optimizer
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Define losses
        losses = {
            'bbox_output': 'mse',  # Mean Squared Error for bounding box regression
            'class_output': 'sparse_categorical_crossentropy',  # Classification loss
            'segmentation_output': 'sparse_categorical_crossentropy'  # Segmentation loss
        }
        
        # Define loss weights
        loss_weights = {
            'bbox_output': detection_loss_weight,
            'class_output': classification_loss_weight,
            'segmentation_output': segmentation_loss_weight
        }
        
        # Define metrics
        metrics = {
            'bbox_output': ['mae'],  # Mean Absolute Error for bbox
            'class_output': ['accuracy'],  # Accuracy for classification
            'segmentation_output': ['accuracy']  # Accuracy for segmentation
        }
        
        # Compile the model
        self.model.compile(
            optimizer=opt,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary"""
        print(f"Multitask Model Summary:")
        print(f"Backbone: {self.backbone_name}")
        print(f"Detection classes: {self.num_detection_classes}")
        print(f"Segmentation classes: {self.num_segmentation_classes}")
        print(f"Input shape: {self.input_shape}")
        print("\nModel Architecture:")
        self.model.summary()
    
    def get_feature_maps(self, inputs, layer_names=None):
        """Extract feature maps from specified layers"""
        if layer_names is None:
            layer_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
        
        # Create a model that outputs feature maps
        feature_extractor = Model(
            inputs=self.backbone.input,
            outputs=[self.backbone.get_layer(name).output for name in layer_names if name in [l.name for l in self.backbone.layers]]
        )
        
        # Extract features
        backbone_features = self.backbone(inputs)
        feature_maps = feature_extractor(inputs)
        
        return feature_maps
    
    def predict_with_nms(
        self,
        image: tf.Tensor,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ) -> Dict[str, tf.Tensor]:
        """Predict with Non-Maximum Suppression for detection"""
        predictions = self.predict_single_image(image)
        
        # Apply confidence threshold
        class_probs = predictions['class_output']
        max_prob = tf.reduce_max(class_probs)
        
        if max_prob < confidence_threshold:
            # No confident detections
            return {
                'bbox_output': tf.constant([]),
                'class_output': tf.constant([]),
                'segmentation_output': predictions['segmentation_output']
            }
        
        # For simplicity, return original predictions
        # In practice, you would implement proper NMS here
        return predictions
    
    def save_model(self, filepath: str):
        """Save the multitask model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained multitask model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


class LightweightMultitaskModel(BaseMultitaskModel):
    """
    Lightweight version of multitask model for faster inference
    """
    
    def __init__(
        self,
        num_detection_classes: int,
        num_segmentation_classes: int,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs
    ):
        super().__init__(num_detection_classes, num_segmentation_classes, **kwargs)
        self.input_shape = input_shape
        
        # Build lightweight components
        self._build_backbone()
        self._build_detection_head()
        self._build_classification_head()
        self._build_segmentation_head()
        
        # Build the model
        self._build_model()
    
    def _build_backbone(self):
        """Build lightweight backbone"""
        self.backbone = tf.keras.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
        ], name='lightweight_backbone')
    
    def _build_detection_head(self):
        """Build lightweight detection head"""
        self.detection_head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='linear', name='bbox_output')
        ], name='detection_head')
    
    def _build_classification_head(self):
        """Build lightweight classification head"""
        self.classification_head = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_detection_classes, activation='softmax', name='class_output')
        ], name='classification_head')
    
    def _build_segmentation_head(self):
        """Build lightweight segmentation head"""
        self.segmentation_head = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.UpSampling2D(size=(2, 2)),
            
            layers.Conv2D(self.num_segmentation_classes, 1, activation='softmax', name='segmentation_output')
        ], name='segmentation_head')
    
    def _build_model(self):
        """Build the lightweight multitask model"""
        input_layer = layers.Input(shape=self.input_shape)
        
        # Forward pass
        features = self.backbone(input_layer)
        
        # Outputs
        bbox_output = self.detection_head(features)
        class_output = self.classification_head(features)
        segmentation_output = self.segmentation_head(features)
        
        # Create model
        self.model = Model(
            inputs=input_layer,
            outputs={
                'bbox_output': bbox_output,
                'class_output': class_output,
                'segmentation_output': segmentation_output
            }
        )
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass"""
        return self.model(inputs, training=training)