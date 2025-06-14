import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import List, Tuple, Optional, Callable
from src.config import ModelConfigs, Config
from src.models.base_model import BaseSegmentationModel

try:
    import segmentation_models as sm
    sm.set_framework('tf.keras')
except ImportError:  # pragma: no cover
    sm = None

class SimpleUNet(BaseSegmentationModel):
    """Simple U-Net architecture without pretrained backbone"""
    
    def __init__(
        self, 
        num_classes: int, 
        config: Optional[Config] = None, 
        models_config: Optional[ModelConfigs] = None, 
        **kwargs
    ):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        
        # Encoder (downsampling path)
        self.encoder_blocks = self._build_encoder()
        
        # Decoder (upsampling path)  
        self.decoder_blocks = self._build_decoder()
        
        # Final classification layer
        self.final_conv = layers.Conv2D(
            num_classes, 
            1, 
            activation='softmax' if num_classes > 1 else 'sigmoid',
            name='segmentation_output'
        )
        
    def _build_encoder(self) -> List[keras.Sequential]:
        """Build encoder blocks"""
        params = self.models_config.SEGMENTATION_MODELS[self.model_name.lower()]
        filters = params['encoder_filters']
        encoder_blocks = []
        
        for i, filter_count in enumerate(filters):
            block = keras.Sequential([
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
            ])
            encoder_blocks.append(block)
            
        return encoder_blocks
    
    def _build_decoder(self) -> List[keras.Sequential]:
        """Build decoder blocks"""
        params = self.models_config.SEGMENTATION_MODELS[self.model_name.lower()]
        filters = params['decoder_filters']
        decoder_blocks = []
        
        for i, filter_count in enumerate(filters):
            block = keras.Sequential([
                layers.Conv2DTranspose(filter_count, 2, strides=2, padding='same'),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
            ])
            decoder_blocks.append(block)
            
        return decoder_blocks
    
    def call(self, inputs, training=None) -> tf.Tensor:
        """Forward pass with skip connections"""
        # Encoder path
        skip_connections = []
        x = inputs
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, training=training)
            
            if i < len(self.encoder_blocks) - 1:  # Skip the last layer
                skip_connections.append(x)
                x = layers.MaxPooling2D(2)(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse order
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, training=training)
            
            if i < len(skip_connections):
                # Concatenate with skip connection
                skip = skip_connections[i]
                x = layers.Concatenate()([x, skip])
        
        # Final output
        output = self.final_conv(x)
        
        return output

class PretrainedUNet(BaseSegmentationModel):
    """U-Net with pretrained encoder backbone"""
    
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True,
        backbone_name: str = 'ResNet50', 
        config: Optional[Config] = None, 
        models_config: Optional[ModelConfigs] = None, 
        **kwargs
    ):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        self.backbone_name = backbone_name

        self.pretrained = pretrained
        
        # Pretrained encoder
        self.encoder = self._build_pretrained_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # Final classification layer
        self.final_conv = layers.Conv2D(
            num_classes,
            1,
            activation='softmax' if num_classes > 1 else 'sigmoid',
            name='segmentation_output'
        )
        
    def _build_pretrained_encoder(self) -> keras.Model:
        """Build pretrained encoder"""
        params_model = self.models_config.SEGMENTATION_MODELS[self.backbone_name.lower()]
        input_shape = params_model['input_shape']
        pretrained_weights = params_model['pretrained_weights'] if self.pretrained else None
        freeze_backbone = params_model['freeze_backbone']

        if self.backbone_name == 'ResNet50':
            backbone = keras.applications.ResNet50(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            # Extract features at different scales
            layer_names = [
                'conv1_relu',           # 112x112
                'conv2_block3_out',     # 56x56  
                'conv3_block4_out',     # 28x28
                'conv4_block6_out',     # 14x14
                'conv5_block3_out'      # 7x7
            ]
        elif self.backbone_name == 'EfficientNetB0':
            backbone = keras.applications.EfficientNetB0(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            layer_names = [
                'block2a_expand_activation',  # 112x112
                'block3a_expand_activation',  # 56x56
                'block4a_expand_activation',  # 28x28
                'block6a_expand_activation',  # 14x14
                'top_activation'              # 7x7
            ]
        elif self.backbone_name == 'MobileNetV2':
            backbone = keras.applications.MobileNetV2(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            layer_names = [
                'block_1_expand_relu',  # 112x112
                'block_3_expand_relu',  # 56x56
                'block_6_expand_relu',  # 28x28
                'block_13_expand_relu',  # 14x14
                'block_16_project'       # 7x7
            ]
        elif self.backbone_name == 'MobileNetV3':
            backbone = keras.applications.MobileNetV3(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            layer_names = [
                'block_1_expand_relu',  # 112x112
                'block_3_expand_relu',  # 56x56
                'block_6_expand_relu',  # 28x28
                'block_13_expand_relu',  # 14x14
                'block_16_project'       # 7x7
            ]

        elif self.backbone_name == 'DenseNet121':
            backbone = keras.applications.DenseNet121(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            layer_names = [
                'conv1_relu',           # 112x112
                'conv2_block3_out',     # 56x56  
                'conv3_block4_out',     # 28x28
                'conv4_block6_out',     # 14x14
                'conv5_block3_out'      # 7x7
            ]
        elif self.backbone_name == 'DenseNet169':
            backbone = keras.applications.DenseNet169(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            layer_names = [
                'conv1_relu',           # 112x112
                'conv2_block3_out',     # 56x56  
                'conv3_block4_out',     # 28x28
                'conv4_block6_out',     # 14x14
                'conv5_block3_out'      # 7x7
            ]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Extract intermediate outputs
        outputs = [backbone.get_layer(name).output for name in layer_names]
        encoder = keras.Model(backbone.input, outputs)
        
        # Freeze early layers
        if freeze_backbone:
            for layer in encoder.layers[:-10]:
                layer.trainable = False
            
        return encoder
    
    def _build_decoder(self) -> List[keras.Sequential]:
        """Build decoder with skip connections"""
        decoder_blocks = []
        params_model = self.models_config.SEGMENTATION_MODELS[self.backbone_name.lower()]
        filters = params_model['decoder_filters']
        
        for filter_count in filters:
            block = keras.Sequential([
                layers.Conv2DTranspose(filter_count, 2, strides=2, padding='same'),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
            ])
            decoder_blocks.append(block)
            
        return decoder_blocks
    
    def call(self, inputs, training=None) -> tf.Tensor:
        """Forward pass"""
        # Encoder - extract multi-scale features
        encoder_outputs = self.encoder(inputs, training=training)
        
        # Start from the deepest features
        x = encoder_outputs[-1]
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block(x, training=training)
            
            # Add skip connection if available
            if i < len(encoder_outputs) - 1:
                skip_idx = len(encoder_outputs) - 2 - i
                skip = encoder_outputs[skip_idx]
                
                # Resize skip connection to match current resolution
                target_shape = tf.shape(x)[1:3]
                skip = tf.image.resize(skip, target_shape)
                
                x = layers.Concatenate()([x, skip])
        
        # Final output
        output = self.final_conv(x)
        
        return output

class DeepLabV3Plus(BaseSegmentationModel):
    """DeepLabV3+ implementation"""
    
    def __init__(
        self, 
        num_classes: int, 
        pretrained: bool = True,
        backbone_name: str = 'ResNet50', 
        config: Optional[Config] = None, 
        models_config: Optional[ModelConfigs] = None,
        **kwargs
    ):
        super().__init__(num_classes, **kwargs)
        self.config = config or Config()
        self.models_config = models_config or ModelConfigs()
        self.backbone_name = backbone_name
        
        self.pretrained = pretrained
        
        # Pretrained backbone
        self.backbone = self._build_backbone()
        
        # ASPP module
        self.aspp = self._build_aspp()
        
        # Decoder
        self.decoder = self._build_decoder()
        
        # Final classification layer
        self.final_conv = layers.Conv2D(
            num_classes,
            1,
            activation='softmax' if num_classes > 1 else 'sigmoid',
            name='segmentation_output'
        )
    
    def _build_backbone(self) -> keras.Model:
        """Build backbone network"""
        params_model = self.models_config.SEGMENTATION_MODELS[self.backbone_name.lower()]
        input_shape = params_model['input_shape']
        pretrained_weights = params_model['pretrained_weights'] if self.pretrained else None
        freeze_backbone = params_model['freeze_backbone']
        
        if self.backbone_name == 'ResNet50':
            backbone = keras.applications.ResNet50(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            # Use output from conv4_block6_out for low-level features
            # and conv5_block3_out for high-level features
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'MobileNetV2':
            backbone = keras.applications.MobileNetV2(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            low_level_layer = 'block_3_expand_relu'
            high_level_layer = 'block_16_project'
        elif self.backbone_name == 'MobileNetV3':
            backbone = keras.applications.MobileNetV3(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            low_level_layer = 'block_3_expand_relu'
            high_level_layer = 'block_16_project'
        elif self.backbone_name == 'DenseNet121':
            backbone = keras.applications.DenseNet121(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'DenseNet169':
            backbone = keras.applications.DenseNet169(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'EfficientNetB0':
            backbone = keras.applications.EfficientNetB0(
                input_shape=input_shape,
                weights=pretrained_weights,
                include_top=False
            )
            low_level_layer = 'block2a_expand_activation'
            high_level_layer = 'top_activation'
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        low_level_output = backbone.get_layer(low_level_layer).output
        high_level_output = backbone.get_layer(high_level_layer).output
        
        model = keras.Model(
            backbone.input,
            [low_level_output, high_level_output]
        )
        
        return model
    
    def _build_aspp(self) -> Callable[[tf.Tensor], tf.Tensor]:
        """Build Atrous Spatial Pyramid Pooling module"""
        def aspp_block(x, filters=256, rate=1):
            if rate == 1:
                conv = layers.Conv2D(filters, 1, padding='same', activation='relu')
            else:
                conv = layers.Conv2D(filters, 3, padding='same', dilation_rate=rate, activation='relu')
            
            x = conv(x)
            x = layers.BatchNormalization()(x)
            return x
        
        def aspp_module(inputs):
            # Different dilation rates
            aspp1 = aspp_block(inputs, rate=1)
            aspp2 = aspp_block(inputs, rate=6)
            aspp3 = aspp_block(inputs, rate=12)
            aspp4 = aspp_block(inputs, rate=18)
            
            # Global average pooling
            image_pooling = layers.GlobalAveragePooling2D()(inputs)
            image_pooling = layers.Dense(256, activation='relu')(image_pooling)
            image_pooling = layers.Reshape((1, 1, 256))(image_pooling)
            image_pooling = layers.UpSampling2D(
                size=(tf.shape(inputs)[1], tf.shape(inputs)[2]),
                interpolation='bilinear'
            )(image_pooling)
            
            # Concatenate all branches
            concat = layers.Concatenate()([aspp1, aspp2, aspp3, aspp4, image_pooling])
            
            # Final conv
            output = layers.Conv2D(256, 1, padding='same', activation='relu')(concat)
            output = layers.BatchNormalization()(output)
            
            return output
        
        return aspp_module
    
    def _build_decoder(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Build decoder module"""
        def decoder_module(high_level_features, low_level_features):
            # Upsample high-level features
            upsampled = layers.UpSampling2D(size=4, interpolation='bilinear')(high_level_features)
            
            # Process low-level features
            low_level = layers.Conv2D(48, 1, padding='same', activation='relu')(low_level_features)
            low_level = layers.BatchNormalization()(low_level)
            
            # Concatenate
            concat = layers.Concatenate()([upsampled, low_level])
            
            # Refine features
            x = layers.Conv2D(256, 3, padding='same', activation='relu')(concat)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            
            return x
        
        return decoder_module
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Extract features
        low_level_features, high_level_features = self.backbone(inputs, training=training)
        
        # Apply ASPP
        aspp_features = self.aspp(high_level_features)
        
        # Decode features
        decoded = self.decoder(aspp_features, low_level_features)
        
        # Upsample to original resolution
        upsampled = layers.UpSampling2D(size=4, interpolation='bilinear')(decoded)
        
        # Final classification
        output = self.final_conv(upsampled)
        
        return output

class SMKerasWrapper(BaseSegmentationModel):
    """Wrap a segmentation-models Keras model so it plugs into our BaseSegmentationModel API."""

    def __init__(self, keras_model: keras.Model, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
        self._model = keras_model

    def call(self, inputs, training=None, mask=None):  # type: ignore[override]
        return self._model(inputs, training=training)

def create_segmentation_model(
    model_type: str,
    num_classes: int,
    backbone: str = 'ResNet50',
    pretrained: bool = True,
    config: Optional[Config] = None,
    models_config: Optional[ModelConfigs] = None,
    **kwargs
) -> BaseSegmentationModel:
    """Factory function to create segmentation models"""
    if model_type == 'simple_unet':
        return SimpleUNet(
            num_classes=num_classes,
            config=config,
            models_config=models_config,
            **kwargs
        )
    elif model_type == 'pretrained_unet':
        return PretrainedUNet(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            config=config,
            models_config=models_config,
            **kwargs
        )
    elif model_type == 'deeplabv3plus':
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            config=config,
            models_config=models_config,
            **kwargs
        )
    elif model_type in ['unetpp', 'unetplusplus', 'unet_plus_plus', 'fpn', 'pspnet', 'linknet']:
        if sm is None:
            raise ImportError("segmentation-models package is required for model_type '{}'".format(model_type))
        activation = 'softmax' if num_classes > 1 else 'sigmoid'
        model_builder = {
            'unetpp': sm.UnetPlusPlus,
            'unetplusplus': sm.UnetPlusPlus,
            'unet_plus_plus': sm.UnetPlusPlus,
            'fpn': sm.FPN,
            'pspnet': sm.PSPNet,
            'linknet': sm.Linknet,
        }[model_type]
        keras_model = model_builder(
            backbone_name=backbone.lower(),
            classes=num_classes,
            activation=activation,
            encoder_weights='imagenet' if pretrained else None,
        )
        return SMKerasWrapper(keras_model, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")