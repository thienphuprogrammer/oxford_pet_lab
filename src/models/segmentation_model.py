import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from src.config.model_configs import ModelConfigs

class SegmentationModel:
    def __init__(self, model_name='unet_resnet50', pretrained=True):
        """Initialize segmentation model with specified backbone."""
        self.config = ModelConfigs()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = self._build_model()

    def _build_unet_decoder(self, encoder_outputs, num_classes):
        """Build U-Net decoder architecture."""
        # Get decoder configuration
        decoder_config = self.config.SEGMENTATION_MODELS[self.model_name]
        
        # Get encoder outputs in reverse order
        encoder_outputs.reverse()
        
        # Start with the last encoder output
        x = encoder_outputs[0]
        
        # Build decoder blocks
        for i, filters in enumerate(decoder_config['decoder_filters']):
            # Skip connections
            skip = encoder_outputs[i + 1] if i + 1 < len(encoder_outputs) else None
            
            # Upsampling
            x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            # Concatenate with skip connection
            if skip is not None:
                x = layers.concatenate([x, skip])
            
            # Convolutional blocks
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        
        # Final output layer
        output = layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
        
        return output

    def _build_model(self):
        """Build the complete segmentation model."""
        # Get backbone configuration
        backbone_config = self.config.SEGMENTATION_MODELS[self.model_name]
        
        # Load backbone
        if 'resnet' in self.model_name:
            backbone = ResNet50(
                include_top=False,
                weights='imagenet' if self.pretrained else None,
                input_shape=backbone_config['input_shape'],
            )
        elif 'mobilenet' in self.model_name:
            backbone = MobileNetV2(
                include_top=False,
                weights='imagenet' if self.pretrained else None,
                input_shape=backbone_config['input_shape'],
            )
        
        # Freeze backbone if specified
        if backbone_config['freeze_backbone']:
            for layer in backbone.layers:
                layer.trainable = False
        
        # Get encoder outputs at different levels
        encoder_outputs = []
        for layer in backbone.layers:
            if 'conv' in layer.name:
                encoder_outputs.append(layer.output)
        
        # Add U-Net decoder
        output = self._build_unet_decoder(encoder_outputs, backbone_config['num_classes'])
        
        # Create model
        model = models.Model(
            inputs=backbone.input,
            outputs=output,
            name=f'segmentation_{self.model_name}'
        )
        
        return model

    def compile(self, learning_rate=1e-4):
        """Compile the model with appropriate loss functions."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=3)]
        )

    def get_model(self):
        """Return the built model."""
        return self.model
