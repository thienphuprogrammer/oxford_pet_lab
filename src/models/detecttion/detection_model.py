import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetV2B1, EfficientNetV2B0, EfficientNetV2B2

from src.config import ModelConfigs, Config
from src.models.base_model import BaseDetectionModel
from src.models.detecttion.module import AttentionBlock, FPN, BiFPN

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
        self.model_type = 'simple_detection_model'
        self._init_params()
        self.backbone = self._build_backbone()
        self.head = self._build_detection_head()
        self.classification_head = self._build_classification_head()
        

    def _init_params(self):
        """Initialize parameters"""
        params_model = self.models_config.DETECTION_MODELS[self.model_type]
        self.backbone_name = params_model.get('backbone_name', 'resnet50')
        self.backbone_units = params_model.get('backbone_units', [64, 128, 256, 512])
        self.detection_head_units = params_model.get('detection_head_units', [256, 128, 64])
        self.classification_head_units = params_model.get('classification_head_units', [256, 128, 64])
        self.bbox_output_units = params_model.get('bbox_output_units', 4)
        self.class_output_units = params_model.get('class_output_units', 3)
        self.pretrained_weights = params_model.get('pretrained_weights', True)
        self.freeze_backbone = params_model.get('freeze_backbone', True)

        self.use_attention = params_model.get('use_attention', True)
        self.use_fpn = params_model.get('use_fpn', True)
        self.use_deep_supervision = params_model.get('use_deep_supervision', True)
        self.use_auxiliary_head = params_model.get('use_auxiliary_head', True)
        self.use_classification_head = params_model.get('use_classification_head', True)
        self.use_detection_head = params_model.get('use_detection_head', True)
        

    def _build_backbone(self):  
        """Build the backbone of the model."""
        def residual_block(filters, strides=1, use_attention=True):
            def block(x):
                # Main path
                shortcut = x
                
                # First conv
                x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                
                # Second conv
                x = layers.Conv2D(filters, 3, padding='same')(x)
                x = layers.BatchNormalization()(x)
                
                # Shortcut connection
                if strides != 1 or shortcut.shape[-1] != filters:
                    shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
                    shortcut = layers.BatchNormalization()(shortcut)
                
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)
                
                # Add attention
                if use_attention and self.use_attention:
                    x = AttentionBlock(filters)(x)
                
                return x
            return block
        
        # Input layer
        inputs = layers.Input(shape=(224, 224, 3))
        
        # Stem
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Feature extraction with residual blocks
        feature_maps = []
        
        # Stage 1
        x = residual_block(64)(x)
        x = residual_block(64)(x)
        feature_maps.append(x)
        
        # Stage 2
        x = residual_block(128, strides=2)(x)
        x = residual_block(128)(x)
        feature_maps.append(x)
        
        # Stage 3
        x = residual_block(256, strides=2)(x)
        x = residual_block(256)(x)
        feature_maps.append(x)
        
        # Stage 4
        x = residual_block(512, strides=2)(x)
        x = residual_block(512)(x)
        feature_maps.append(x)
        
        backbone = keras.Model(inputs, feature_maps if self.use_fpn else x)
        return backbone
    

    def _build_detection_head(self):
        """Build the detection head of the model."""
        if self.use_fpn:
            # Multi-scale detection head
            def detection_head(fpn_features):
                bbox_outputs = []
                for i, feature in enumerate(fpn_features):
                    # Detection subnet
                    x = feature
                    for _ in range(4):  # 4 conv layers
                        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
                    
                    # Bbox regression
                    bbox_out = layers.Conv2D(4, 3, padding='same')(x)
                    bbox_out = layers.GlobalAveragePooling2D()(bbox_out)
                    bbox_outputs.append(bbox_out)
                
                # Combine multi-scale outputs
                combined_bbox = layers.Average()(bbox_outputs)
                return combined_bbox
            return detection_head
        else:
            # Single scale detection head
            detection_head = keras.Sequential([
                layers.GlobalAveragePooling2D(),
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(4, name='bbox_output')
            ])
            return detection_head
    
    def _build_classification_head(self):
        """Build the classification head of the model."""
        if self.use_fpn:
            def classification_head(fpn_features):
                class_outputs = []
                for i, feature in enumerate(fpn_features):
                    # Classification subnet
                    x = feature
                    for _ in range(4):  # 4 conv layers
                        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
                    
                    # Classification
                    class_out = layers.Conv2D(self.num_classes, 3, padding='same')(x)
                    class_out = layers.GlobalAveragePooling2D()(class_out)
                    class_outputs.append(class_out)
                
                # Combine multi-scale outputs
                combined_class = layers.Average()(class_outputs)
                combined_class = layers.Activation('softmax', name='class_output')(combined_class)
                return combined_class
            return classification_head
        else:
            classification_head = keras.Sequential([
                layers.GlobalAveragePooling2D(),
                layers.Dense(1024, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax', name='class_output')
            ])
            return classification_head
        
    def call(self, inputs, training=None):
        features = self.backbone(inputs, training=training)
        
        if self.use_fpn:
            fpn_features = self.fpn(features)
            bbox_output = self.detection_head(fpn_features)
            class_output = self.classification_head(fpn_features)
        else:
            bbox_output = self.detection_head(features, training=training)
            class_output = self.classification_head(features, training=training)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output
        }


class PretrainedDetectionModel(BaseDetectionModel):
    def __init__(
        self,
        num_classes: int,
        config: Config = None,
        models_config: ModelConfigs = None,
        **kwargs,
    ):
        """Initialize detection model with specified backbone."""
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        self.model_type = 'pretrained_detection_model'
        self._init_params()
        
        self.backbone = self._build_backbone()
        self.detection_head = self._build_detection_head()
        self.classification_head = self._build_classification_head()

    def _init_params(self):
        """Initialize parameters"""
        params_model = self.models_config.DETECTION_MODELS[self.model_type]
        self.backbone_name = params_model.get('backbone_name', 'resnet50')
        self.pretrained = params_model.get('pretrained', True)
        self.pretrained_weights = params_model.get('pretrained_weights', 'imagenet') if self.pretrained else None
        self.input_shape = params_model.get('input_shape', (224, 224, 3))

    def _build_backbone(self):
        """Build the backbone of the model."""
    
        if self.backbone_name.lower() == 'resnet50':
            backbone = ResNet50(
                include_top=False,
                weights=self.pretrained_weights,
                input_shape=self.input_shape,
            )
            layer_names = [
                'conv1_relu', 
                'conv2_block3_out', 
                'conv3_block4_out', 
                'conv4_block23_out', 
                'conv5_block3_out'
            ]
        elif self.backbone_name.lower() == 'mobilenetv2':
            backbone = MobileNetV2(
                include_top=False,
                weights=self.pretrained_weights,
                input_shape=self.input_shape,
            )
            layer_names = [
                'block_1_expand_relu',
                'block_3_expand_relu',
                'block_6_expand_relu',
                'block_13_expand_relu',
                'block_16_project',
            ]
        elif self.backbone_name.lower() == 'efficientnetv2b0':
            backbone = EfficientNetV2B0(
                include_top=False,
                weights=self.pretrained_weights,
                input_shape=self.input_shape,
            )
            layer_names = [
                'block3a_expand_activation',
                'block4a_expand_activation',
                'block6a_expand_activation',
                'top_activation',
            ]
        elif self.backbone_name.lower() == 'efficientnetv2b1':
            backbone = EfficientNetV2B1(
                include_top=False,
                weights=self.pretrained_weights,
                input_shape=self.input_shape,
            )
            layer_names = [
                'block2b_add',
                'block4a_expand_activation',
                'block6a_expand_activation',
                'top_activation',
            ]
        elif self.backbone_name.lower() == 'efficientnetv2b2':
            backbone = EfficientNetV2B2(
                include_top=False,
                weights=self.pretrained_weights,
                input_shape=self.input_shape,
            )   
            layer_names = [
                'block3a_add',
                'block4a_expand_activation',
                'block6a_expand_activation',
                'block8a_expand_activation',
                'top_activation',
            ]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        
        outputs = [backbone.get_layer(name).output for name in layer_names if name in [layer.name for layer in backbone.layers]]
        
        # If some layers don't exist, use alternative approach
        if len(outputs) < 3:
            outputs = []
            for i, layer in enumerate(backbone.layers):
                if 'block' in layer.name and 'activation' in layer.name:
                    outputs.append(layer.output)
            outputs = outputs[-4:]  # Take last 4 feature maps
        
        feature_extractor = keras.Model(backbone.input, outputs)
        return feature_extractor
    
    def _build_bifpn(self):
        """Build Bidirectional Feature Pyramid Network"""
        return BiFPN(feature_dim=64)
    
    def _build_detection_head(self):
        """Build detection head for multi-scale features"""
        def detection_head(features):
            bbox_outputs = []
            for feature in features:
                x = feature
                # Detection subnet
                for _ in range(3):
                    x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
                
                bbox_out = layers.Conv2D(4, 3, padding='same')(x)
                bbox_out = layers.GlobalAveragePooling2D()(bbox_out)
                bbox_outputs.append(bbox_out)
            
            # Weighted combination
            combined = layers.Average()(bbox_outputs)
            return combined
        
        return detection_head
    
        
    def _build_classification_head(self):
        """Build classification head for multi-scale features"""
        def classification_head(features):
            class_outputs = []
            for feature in features:
                x = feature
                # Classification subnet
                for _ in range(3):
                    x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
                
                class_out = layers.Conv2D(self.num_classes, 3, padding='same')(x)
                class_out = layers.GlobalAveragePooling2D()(class_out)
                class_outputs.append(class_out)
            
            # Weighted combination
            combined = layers.Average()(class_outputs)
            combined = layers.Activation('softmax', name='class_output')(combined)
            return combined
        
        return classification_head
    
    def call(self, inputs, training=None):
        # Extract multi-scale features
        backbone_features = self.backbone(inputs, training=training)
        
        # BiFPN feature fusion
        bifpn_features = self.bifpn(backbone_features)
        
        # Detection and classification
        bbox_output = self.detection_head(bifpn_features)
        class_output = self.classification_head(bifpn_features)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output
        }


class YOLOv5InspiredModel(BaseDetectionModel):
    """Simple YOLO-like detection model"""
    
    def __init__(self, num_classes: int, config: Config = None, models_config: ModelConfigs = None, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        self.model_type = 'yolo_inspired_model'
        self._init_params()
        
        # Darknet-like backbone
        self.backbone = self._build_darknet_backbone()
        
        # YOLO detection head
        self.detection_layers = self._build_detection_layers()

    def _init_params(self):
        """Initialize parameters"""
        params_model = self.models_config.DETECTION_MODELS[self.model_type]
        self.backbone_name = params_model.get('backbone_name', 'resnet50')
        self.pretrained = params_model.get('pretrained', True)
        self.pretrained_weights = params_model.get('pretrained_weights', 'imagenet') if self.pretrained else None
        self.input_shape = params_model.get('input_shape', (224, 224, 3))
        
    def _build_csp_backbone(self):
        """Build CSPDarknet backbone"""
        def csp_block(filters, num_blocks=1):
            def block(x):
                # Split
                x1 = layers.Conv2D(filters // 2, 1, padding='same')(x)
                x1 = layers.BatchNormalization()(x1)
                x1 = layers.Activation('swish')(x1)
                
                x2 = layers.Conv2D(filters // 2, 1, padding='same')(x)
                x2 = layers.BatchNormalization()(x2)
                x2 = layers.Activation('swish')(x2)
                
                # Process x2 through residual blocks
                for _ in range(num_blocks):
                    shortcut = x2
                    x2 = layers.Conv2D(filters // 2, 1, padding='same')(x2)
                    x2 = layers.BatchNormalization()(x2)
                    x2 = layers.Activation('swish')(x2)
                    x2 = layers.Conv2D(filters // 2, 3, padding='same')(x2)
                    x2 = layers.BatchNormalization()(x2)
                    x2 = layers.Activation('swish')(x2)
                    x2 = layers.Add()([x2, shortcut])
                
                # Concatenate and process
                x = layers.Concatenate()([x1, x2])
                x = layers.Conv2D(filters, 1, padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('swish')(x)
                
                return x
            return block
        
        inputs = layers.Input(shape=(224, 224, 3))
        
        # Focus layer (simulate space-to-depth)
        x = layers.Conv2D(64, 6, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # CSP stages
        feature_maps = []
        
        x = csp_block(128, 1)(x)
        x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
        feature_maps.append(x)
        
        x = csp_block(256, 2)(x)
        x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
        feature_maps.append(x)
        
        x = csp_block(512, 8)(x)
        x = layers.Conv2D(512, 3, strides=2, padding='same')(x)
        feature_maps.append(x)
        
        x = csp_block(1024, 4)(x)
        feature_maps.append(x)
        
        backbone = keras.Model(inputs, feature_maps)
        return backbone
    
    def _build_pafpn_neck(self):
        """Build Path Aggregation FPN neck"""
        class PAFPN(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                
            def build(self, input_shape):
                # Lateral connections
                self.lateral_convs = []
                for i in range(len(input_shape)):
                    conv = layers.Conv2D(256, 1, padding='same', name=f'lateral_{i}')
                    self.lateral_convs.append(conv)
                
                # Output convs
                self.output_convs = []
                for i in range(len(input_shape)):
                    conv = layers.Conv2D(256, 3, padding='same', name=f'output_{i}')
                    self.output_convs.append(conv)
                    
                super().build(input_shape)
            
            def call(self, inputs):
                # Top-down path
                laterals = [conv(inp) for conv, inp in zip(self.lateral_convs, inputs)]
                
                # FPN top-down
                fpn_features = [laterals[-1]]
                for i in range(len(laterals) - 2, -1, -1):
                    target_shape = tf.shape(laterals[i])
                    upsampled = tf.image.resize(fpn_features[0], [target_shape[1], target_shape[2]])
                    merged = laterals[i] + upsampled
                    fpn_features.insert(0, merged)
                
                # Bottom-up path (Path Aggregation)
                pan_features = [fpn_features[0]]
                for i in range(1, len(fpn_features)):
                    target_shape = tf.shape(fpn_features[i])
                    downsampled = layers.MaxPooling2D(2)(pan_features[-1])
                    downsampled = tf.image.resize(downsampled, [target_shape[1], target_shape[2]])
                    merged = fpn_features[i] + downsampled
                    pan_features.append(merged)
                
                # Apply output convs
                outputs = [conv(feat) for conv, feat in zip(self.output_convs, pan_features)]
                return outputs
        
        return PAFPN()
    
    def _build_yolo_head(self):
        """Build YOLO detection head"""
        def yolo_head(features):
            outputs = []
            num_anchors = 3  # 3 anchors per scale
            
            for feature in features:
                # Detection head
                x = layers.Conv2D(256, 3, padding='same', activation='swish')(feature)
                x = layers.Conv2D(256, 3, padding='same', activation='swish')(x)
                
                # Output: [batch, grid_h, grid_w, anchors * (4 + 1 + num_classes)]
                output_dim = num_anchors * (4 + 1 + self.num_classes)
                detection = layers.Conv2D(output_dim, 1)(x)
                outputs.append(detection)
            
            return outputs
        
        return yolo_head
    
    def call(self, inputs, training=None):
        # Backbone
        backbone_features = self.backbone(inputs, training=training)
        
        # Neck
        neck_features = self.neck(backbone_features)
        
        # Head
        detections = self.head(neck_features)
        
        # For compatibility, return aggregated outputs
        # In practice, you'd want to handle multi-scale outputs separately
        bbox_outputs = []
        class_outputs = []
        
        for detection in detections:
            # Reshape and split
            batch_size = tf.shape(detection)[0]
            grid_size = tf.shape(detection)[1]
            
            detection = tf.reshape(detection, [batch_size, grid_size, grid_size, 3, 5 + self.num_classes])
            
            bbox_out = detection[..., :4]
            bbox_out = tf.reduce_mean(bbox_out, axis=[1, 2, 3])  # Global average
            bbox_outputs.append(bbox_out)
            
            class_out = detection[..., 5:]
            class_out = tf.reduce_mean(class_out, axis=[1, 2, 3])  # Global average
            class_outputs.append(class_out)
        
        # Combine multi-scale outputs
        bbox_output = tf.reduce_mean(tf.stack(bbox_outputs), axis=0)
        class_output = tf.reduce_mean(tf.stack(class_outputs), axis=0)
        class_output = tf.nn.softmax(class_output)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output
        }

# Model factory function
def create_optimized_model(
    model_type: str,
    num_classes: int,
    config: Config = None,
    models_config: ModelConfigs = None,
    **kwargs
):
    """Factory function to create optimized detection models"""
    
    model_type = model_type.lower()
    model_map = {
        'simple_detection_model': SimpleDetectionModel,
        'pretrained_detection_model': PretrainedDetectionModel,
        'yolo_inspired_model': YOLOv5InspiredModel,
    }
    if model_type in model_map:
        return model_map[model_type](
            num_classes=num_classes,
            config=config,
            models_config=models_config,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
