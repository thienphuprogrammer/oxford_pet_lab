from __future__ import annotations

from typing import List, Optional, Callable

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import ModelConfigs, Config
from src.models.base_model import BaseSegmentationModel
from src.models.segmentation.module import ResidualBlock, CBAM

try:
    import segmentation_models as sm
    sm.set_framework('tf.keras')
except ImportError:  # pragma: no cover
    sm = None

class PretrainedUNet(BaseSegmentationModel):
    """U-Net with pretrained encoder backbone"""
    
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
        self.model_type = 'pretrained_unet'
        # Pretrained encoder
        self.encoder = self._build_pretrained_encoder()
        
        # Decoder
        self.decoder = self._build_decoder()

        self.fpn_convs = self._build_fpn_layers()

        # Final classification layer
        self.final_conv = layers.Conv2D(
            num_classes,
            1,
            activation='softmax' if num_classes > 1 else 'sigmoid',
            name='segmentation_output'
        )

        self._init_params()

        if self.use_deep_supervision:
            self.aux_heads = self._build_auxiliary_heads()

    def _init_params(self):
        params_model = self.models_config.SEGMENTATION_MODELS[self.model_type.lower()]
        self.pretrained = params_model.get('pretrained', False)
        self.input_shape = params_model.get('input_shape', (224, 224, 3))
        self.backbone_name = params_model.get('backbone', 'ResNet50')
        self.use_attention = params_model.get('use_attention', False)
        self.use_deep_supervision = params_model.get('use_deep_supervision', False)
        self.decoder_filters = params_model.get('decoder_filters', [256, 128, 64, 32, 16])
        self.fpn_filters = params_model.get('fpn_filters', [256, 256, 256, 256, 256])
        self.pretrained_weights = params_model.get('pretrained_weights', None) if self.pretrained else None
        self.aux_heads = params_model.get('aux_heads', 3)
    
    def _build_modern_encoder(self) -> keras.Model:
        """Build pretrained encoder"""
        if self.backbone_name.startswith('EfficientNetV2'):
            if self.backbone_name == 'EfficientNetV2B0':
                backbone = keras.applications.EfficientNetV2B0(
                    input_shape=self.input_shape,
                    weights=self.pretrained_weights,
                    include_top=False
                )
                feature_layers = [
                    'block1a_project_activation',  # 112x112
                    'block2b_add',                 # 56x56
                    'block4a_expand_activation',   # 28x28
                    'block6a_expand_activation',   # 14x14
                    'top_activation'               # 7x7
                ]
            elif self.backbone_name == 'EfficientNetV2B1':
                backbone = keras.applications.EfficientNetV2B1(
                    input_shape=self.input_shape,
                    weights=self.pretrained_weights,
                    include_top=False
                )
                feature_layers = [
                    'block1a_project_activation',  # 112x112
                    'block2b_add',                 # 56x56
                    'block4a_expand_activation',   # 28x28
                    'block6a_expand_activation',   # 14x14
                    'top_activation'               # 7x7
                ]
        elif self.backbone_name == 'ConvNeXtTiny':
            backbone = keras.applications.ConvNeXtTiny(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            feature_layers = [
                'convnext_tiny_stage_1_block_0_add',
                'convnext_tiny_stage_2_block_0_add', 
                'convnext_tiny_stage_3_block_0_add',
                'convnext_tiny_stage_4_block_0_add',
                'convnext_tiny_head_layernorm'
            ]
        elif self.backbone_name == 'ResNet50':
            backbone = keras.applications.ResNet50(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            feature_layers = [
                'conv1_relu',
                'conv2_block3_out',
                'conv3_block4_out', 
                'conv4_block6_out',
                'conv5_block3_out'
            ]
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        # Extract intermediate outputs
        outputs = [backbone.get_layer(name).output for name in feature_layers]
        encoder = keras.Model(backbone.input, outputs, name='encoder')
        
        return encoder
    
    def _build_decoder(self) -> List[keras.Sequential]:
        """Build decoder with skip connections"""
        decoder_blocks = []
        
        for i, filter_count in enumerate(self.decoder_filters):
            block = keras.Sequential([
                layers.Conv2DTranspose(filter_count, 2, strides=2, padding='same'),
                ResidualBlock(filter_count, use_attention=self.use_attention),
                ResidualBlock(filter_count, use_attention=self.use_attention),
            ], name=f'decoder_block_{i}')
            decoder_blocks.append(block)
            
        return decoder_blocks
    
    def _build_fpn_layers(self):
        """Build Feature Pyramid Network layers"""
        fpn_convs = []
        for i, filter_count in enumerate(self.fpn_filters):
            fpn_convs.append(layers.Conv2D(filter_count, 1, padding='same', name=f'fpn_conv_{i}'))
        return fpn_convs
    
    def _build_auxiliary_heads(self):
        """Build auxiliary heads for deep supervision"""
        aux_heads = []
        for i in range(self.aux_heads):  # 3 auxiliary outputs
            head = keras.Sequential([
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(self.num_classes, 1, activation='softmax' if self.num_classes > 1 else 'sigmoid')
            ], name=f'aux_head_{i}')
            aux_heads.append(head)
        return aux_heads
    
    def call(self, inputs, training=None) -> tf.Tensor:
        """Forward pass"""
        # Encoder
        encoder_outputs = self.encoder(inputs, training=training)
        
        # FPN-style feature processing
        fpn_features = []
        for i, (feature, fpn_conv) in enumerate(zip(encoder_outputs, self.fpn_convs)):
            fpn_feature = fpn_conv(feature)
            fpn_features.append(fpn_feature)
        
        # Decoder with skip connections
        x = fpn_features[-1]  # Start from deepest features
        aux_outputs = []
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, training=training)
            
            # Skip connection with attention
            if i < len(fpn_features) - 1:
                skip_idx = len(fpn_features) - 2 - i
                skip = fpn_features[skip_idx]
                
                # Resize to match
                target_shape = tf.shape(x)[1:3]
                skip = tf.image.resize(skip, target_shape)
                
                if self.use_attention:
                    attention = layers.MultiHeadAttention(
                        num_heads=8, key_dim=32, name=f'cross_attention_{i}'
                    )
                    x_reshaped = tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])
                    skip_reshaped = tf.reshape(skip, [tf.shape(skip)[0], -1, tf.shape(skip)[-1]])
                    attended = attention(x_reshaped, skip_reshaped, training=training)
                    attended = tf.reshape(attended, tf.shape(x))
                    x = layers.Add()([x, attended])
                else:
                    x = layers.Concatenate()([x, skip])
            
            # Auxiliary outputs for deep supervision
            if self.use_deep_supervision and i < 3 and training:
                aux_out = self.aux_heads[i](x)
                aux_out = tf.image.resize(aux_out, tf.shape(inputs)[1:3])
                aux_outputs.append(aux_out)
        
        # Final output
        output = self.final_conv(x)
        
        if self.use_deep_supervision and training:
            return [output] + aux_outputs
        return output


class UNet3Plus(BaseSegmentationModel):
    """DeepLabV3+ implementation"""
    
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
        self.model_type = 'unet3plus'

        self._init_params()

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.cls_head = self._build_classification_head()
        self.final_conv = layers.Conv2D(
            num_classes,
            1,
            activation='softmax' if num_classes > 1 else 'sigmoid',
            name='segmentation_output'
        )

    def _init_params(self):
        params_model = self.models_config.SEGMENTATION_MODELS[self.model_type.lower()]
        self.encoder_filters = params_model['encoder_filters']
        self.decoder_filters = params_model['decoder_filters']
        self.use_attention = params_model['use_attention']
        self.use_deep_supervision = params_model['use_deep_supervision']

    def _build_encoder(self):
        """Build encoder with residual blocks"""
        encoder_blocks = []
        
        for i, filter_count in enumerate(self.encoder_filters):
            block = keras.Sequential([
                ResidualBlock(filter_count, use_attention=True),
                ResidualBlock(filter_count, use_attention=True),
            ], name=f'encoder_block_{i}')
            encoder_blocks.append(block)
        return encoder_blocks
    
    def _build_decoder(self) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """Build decoder module"""
        decoder_blocks = []

        for i, filter_count in enumerate(self.decoder_filters):
            block = keras.Sequential([
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
                CBAM(),
                layers.Conv2D(filter_count, 3, padding='same', activation='relu'),
                layers.BatchNormalization(),
            ], name=f'decoder_block_{i}')
            decoder_blocks.append(block)
        return decoder_blocks
    
    def _build_classification_head(self):
        """Classification head for guided attention"""
        return keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='sigmoid')
        ])
    
    
    def call(self, inputs, training=None):
        # Encoder
        encoder_features = []
        x = inputs
        
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x, training=training)
            encoder_features.append(x)
            if i < len(self.encoder_blocks) - 1:
                x = layers.MaxPooling2D(2)(x)
        
        # UNet3+ decoder with full-scale connections
        decoder_features = []
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Collect features from all scales
            level_features = []
            
            # From encoder (different scales)
            for j, enc_feat in enumerate(encoder_features[:-1]):  # Exclude bottleneck
                if j <= i:  # Upsample from higher resolution
                    feat = enc_feat
                    scale_factor = 2 ** (i - j)
                    if scale_factor > 1:
                        feat = layers.MaxPooling2D(scale_factor)(feat)
                else:  # Downsample from lower resolution  
                    feat = enc_feat
                    scale_factor = 2 ** (j - i)
                    feat = layers.UpSampling2D(scale_factor, interpolation='bilinear')(feat)
                
                # Standardize channels
                feat = layers.Conv2D(64, 1, activation='relu')(feat)
                level_features.append(feat)
            
            # From previous decoder levels
            for prev_feat in decoder_features:
                feat = layers.UpSampling2D(2, interpolation='bilinear')(prev_feat)
                level_features.append(feat)
            
            # From bottleneck
            if i == 0:
                bottleneck = encoder_features[-1]
                bottleneck = layers.UpSampling2D(2, interpolation='bilinear')(bottleneck)
                bottleneck = layers.Conv2D(64, 1, activation='relu')(bottleneck)
                level_features.append(bottleneck)
            
            # Concatenate and process
            if level_features:
                concat_feat = layers.Concatenate()(level_features)
                decoded = decoder_block(concat_feat, training=training)
                decoder_features.append(decoded)
        
        # Final output
        final_features = decoder_features[-1] if decoder_features else encoder_features[-1]
        
        # Upsample to original size
        output = layers.UpSampling2D(size=4, interpolation='bilinear')(final_features)
        output = self.final_conv(output)
        
        return output


class DeepLabV3Plus(BaseSegmentationModel):
    """DeepLabV3+ implementation"""
    
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
        self.model_type = 'deeplabv3plus'

        self._init_params()
        
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
    
    def _init_params(self):
        params_model = self.models_config.SEGMENTATION_MODELS[self.model_type.lower()]
        self.backbone_name = params_model.get('backbone', 'ResNet50')
        self.pretrained = params_model.get('pretrained', False)
        self.input_shape = params_model.get('input_shape', (224, 224, 3))
        self.pretrained_weights = params_model.get('pretrained_weights', None) if self.pretrained else None
        self.freeze_backbone = params_model.get('freeze_backbone', False)
        self.encoder_filters = params_model.get('encoder_filters', [64, 128, 256, 512, 1024])
        self.decoder_filters = params_model.get('decoder_filters', [64, 64, 64, 64])
        self.use_attention = params_model.get('use_attention', True)
        self.use_deep_supervision = params_model.get('use_deep_supervision', True)
    
    def _build_backbone(self) -> keras.Model:
        """Build backbone network"""
        if self.backbone_name == 'ResNet50':
            backbone = keras.applications.ResNet50(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            # Use output from conv4_block6_out for low-level features
            # and conv5_block3_out for high-level features
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'MobileNetV2':
            backbone = keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            low_level_layer = 'block_3_expand_relu'
            high_level_layer = 'block_16_project'
        elif self.backbone_name == 'MobileNetV3':
            backbone = keras.applications.MobileNetV3(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            low_level_layer = 'block_3_expand_relu'
            high_level_layer = 'block_16_project'
        elif self.backbone_name == 'DenseNet121':
            backbone = keras.applications.DenseNet121(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'DenseNet169':
            backbone = keras.applications.DenseNet169(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
                include_top=False
            )
            low_level_layer = 'conv2_block3_out'
            high_level_layer = 'conv5_block3_out'
        elif self.backbone_name == 'EfficientNetB0':
            backbone = keras.applications.EfficientNetB0(
                input_shape=self.input_shape,
                weights=self.pretrained_weights,
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


class TransUNet(BaseSegmentationModel):
    """Transformer-based UNet for segmentation"""
    
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
        self.model_type = 'transunet'

        self._init_params()

        # CNN encoder
        self.cnn_encoder = self._build_cnn_encoder()
        
        # Transformer encoder
        self.transformer = self._build_transformer()

        
        # Hybrid decoder
        self.decoder = self._build_hybrid_decoder()
        
        self.final_conv = layers.Conv2D(
            num_classes, 1, activation='softmax' if num_classes > 1 else 'sigmoid'
        )
    
    def _init_params(self):
        params_model = self.models_config.SEGMENTATION_MODELS[self.model_type.lower()]
        self.encoder_filters = params_model['encoder_filters']
        self.transformer_num_heads = params_model['transformer_num_heads']
        self.transformer_key_dim = params_model['transformer_key_dim']
        self.transformer_embed_dim = params_model['transformer_embed_dim']
        self.transformer_num_layers = params_model['transformer_num_layers']
        self.transformer_dropout = params_model['transformer_dropout']
        self.transformer_attention_dropout = params_model['transformer_attention_dropout']
        self.transformer_ffn_dropout = params_model['transformer_ffn_dropout']

    def _build_cnn_encoder(self):
        """Build CNN encoder for low-level features"""
        return keras.Sequential([
            *[
                (
                    layers.Conv2D(filter_count, 3, padding='same', activation='relu'), 
                    layers.BatchNormalization()
                )
                for filter_count in self.encoder_filters
            ],
            layers.MaxPooling2D(2),
            *[
                (
                    ResidualBlock(filter_count),
                    layers.MaxPooling2D(2)
                )
                for filter_count in self.encoder_filters
            ]
        ])
    
    def _build_transformer(self):
        """Build transformer encoder"""
        return keras.Sequential([
            layers.MultiHeadAttention(num_heads=self.transformer_num_heads, key_dim=self.transformer_key_dim),
            layers.LayerNormalization(),
            layers.Dense(self.transformer_embed_dim * 4, activation='gelu'),
            layers.Dense(self.transformer_embed_dim),
            layers.LayerNormalization(),
        ])
    
    def _build_hybrid_decoder(self):
        """Build hybrid CNN-Transformer decoder"""
        return keras.Sequential([
            layers.Conv2DTranspose(256, 2, strides=2, padding='same'),
            ResidualBlock(256, use_attention=True),
            layers.Conv2DTranspose(128, 2, strides=2, padding='same'), 
            ResidualBlock(128, use_attention=True),
            layers.Conv2DTranspose(64, 2, strides=2, padding='same'),
            ResidualBlock(64, use_attention=True),
        ])
    
    def call(self, inputs, training=None):
        # CNN features
        cnn_features = self.cnn_encoder(inputs, training=training)
        
        # Patch embedding for transformer
        patches = self._extract_patches(cnn_features)
        
        # Transformer processing
        transformer_features = self.transformer(patches, training=training)
        
        # Reshape back to spatial format
        spatial_features = self._patches_to_spatial(transformer_features, cnn_features)
        
        # Hybrid decoding
        decoded = self.decoder(spatial_features, training=training)
        
        # Final prediction
        output = self.final_conv(decoded)
        
        return output
    
    def _extract_patches(self, x):
        """Extract patches for transformer"""
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape to sequence
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        
        return patches
    
    def _patches_to_spatial(self, patches, reference):
        """Convert patches back to spatial format"""
        batch_size = tf.shape(reference)[0]
        height, width, channels = reference.shape[1:]
        
        # Reshape patches back to spatial
        spatial = tf.reshape(patches, [batch_size, height, width, channels])
        
        return spatial

def create_segmentation_model(
    model_type: str,
    num_classes: int,
    config: Optional[Config] = None,
    models_config: Optional[ModelConfigs] = None,
    **kwargs
) -> BaseSegmentationModel:
    """Factory function to create segmentation models"""

    model_map = {
        'unet3plus': UNet3Plus,
        'transunet': TransUNet,
        'enhanced_deeplabv3plus': DeepLabV3Plus,  # Use existing but can be enhanced
        'pretrained_unet': PretrainedUNet,
    }

    if model_type in model_map:
        if model_type == 'unet3plus':
            return UNet3Plus(
                num_classes=num_classes,
                config=config,
                models_config=models_config,
                **kwargs
            )
        elif model_type == 'transunet':
            return TransUNet(
                num_classes=num_classes,
                config=config,
                models_config=models_config,
                **kwargs
            )
        elif model_type == 'enhanced_deeplabv3plus':
            return DeepLabV3Plus(
                num_classes=num_classes,
                config=config,
                models_config=models_config,
                **kwargs
            )
        elif model_type == 'pretrained_unet':
            return PretrainedUNet(
                num_classes=num_classes,
                config=config,
                models_config=models_config,
                **kwargs
            )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
