import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from ..config.model_configs import ModelConfigs

class MultiTaskModel:
    def __init__(self, pretrained=True):
        """Initialize multi-task model with shared backbone."""
        self.config = ModelConfigs()
        self.pretrained = pretrained
        self.model = self._build_model()

    def _build_shared_backbone(self):
        """Build shared backbone network."""
        backbone_config = self.config.MULTITASK_MODEL
        
        # Load ResNet50 as backbone
        backbone = ResNet50(
            include_top=False,
            weights='imagenet' if self.pretrained else None,
            input_shape=backbone_config['input_shape'],
        )
        
        # Freeze backbone if specified
        if backbone_config['freeze_backbone']:
            for layer in backbone.layers:
                layer.trainable = False
        
        return backbone

    def _build_detection_head(self, x):
        """Build detection head for bounding box and classification."""
        # Shared features
        x = layers.Conv2D(512, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        # Bounding box prediction
        bbox_head = x
        for units in self.config.MULTITASK_MODEL['detection_head_units']:
            bbox_head = layers.Conv2D(units, (3, 3), padding='same')(bbox_head)
            bbox_head = layers.BatchNormalization()(bbox_head)
            bbox_head = layers.ReLU()(bbox_head)
        bbox_head = layers.GlobalAveragePooling2D()(bbox_head)
        bbox_output = layers.Dense(4, name='bbox_output')(bbox_head)
        
        # Classification prediction
        class_head = x
        for units in self.config.MULTITASK_MODEL['detection_head_units']:
            class_head = layers.Conv2D(units, (3, 3), padding='same')(class_head)
            class_head = layers.BatchNormalization()(class_head)
            class_head = layers.ReLU()(class_head)
        class_head = layers.GlobalAveragePooling2D()(class_head)
        class_output = layers.Dense(self.config.MULTITASK_MODEL['class_output_units'], 
                                  activation='softmax', name='class_output')(class_head)
        
        return bbox_output, class_output

    def _build_segmentation_head(self, encoder_outputs):
        """Build segmentation head with U-Net architecture."""
        # Get decoder configuration
        decoder_config = self.config.MULTITASK_MODEL
        
        # Get encoder outputs in reverse order
        encoder_outputs.reverse()
        
        # Start with the last encoder output
        x = encoder_outputs[0]
        
        # Build decoder blocks
        for i, filters in enumerate(decoder_config['segmentation_decoder_filters']):
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
        seg_output = layers.Conv2D(self.config.MULTITASK_MODEL['seg_output_units'], 
                                  (1, 1), activation='softmax', name='seg_output')(x)
        
        return seg_output

    def _build_model(self):
        """Build the complete multi-task model."""
        # Build shared backbone
        backbone = self._build_shared_backbone()
        
        # Get encoder outputs at different levels
        encoder_outputs = []
        for layer in backbone.layers:
            if 'conv' in layer.name:
                encoder_outputs.append(layer.output)
        
        # Build detection head
        bbox_output, class_output = self._build_detection_head(encoder_outputs[-1])
        
        # Build segmentation head
        seg_output = self._build_segmentation_head(encoder_outputs)
        
        # Create model
        model = models.Model(
            inputs=backbone.input,
            outputs=[bbox_output, class_output, seg_output],
            name='multitask_model'
        )
        
        return model

    def compile(self, learning_rate=1e-4):
        """Compile the model with appropriate loss functions."""
        bbox_loss = tf.keras.losses.MeanSquaredError()
        class_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        seg_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        # Get loss weights from config
        loss_weights = self.config.MULTITASK_MODEL['loss_weights']
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'bbox_output': bbox_loss,
                'class_output': class_loss,
                'seg_output': seg_loss
            },
            loss_weights=loss_weights,
            metrics={
                'class_output': ['accuracy'],
                'seg_output': [tf.keras.metrics.MeanIoU(num_classes=3)]
            }
        )

    def get_model(self):
        """Return the built model."""
        return self.model
