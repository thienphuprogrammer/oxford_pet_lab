import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2
from ..config.model_configs import ModelConfigs

class DetectionModel:
    def __init__(self, model_name='resnet50', pretrained=True):
        """Initialize detection model with specified backbone."""
        self.config = ModelConfigs()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = self._build_model()

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

    def compile(self, learning_rate=1e-4):
        """Compile the model with appropriate loss functions."""
        bbox_loss = tf.keras.losses.MeanSquaredError()
        class_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                'bbox_output': bbox_loss,
                'class_output': class_loss
            },
            metrics={
                'class_output': ['accuracy']
            }
        )

    def get_model(self):
        """Return the built model."""
        return self.model
