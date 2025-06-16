import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict, Tuple, Any
from abc import abstractmethod 

class BaseModel(Model):
  def __init__(self, num_classes: int, **kwargs):
    super().__init__(**kwargs)
    self.num_classes = num_classes

  def call(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False,
      mask: tf.Tensor = None,
  ) -> Dict[str, tf.Tensor]:
    raise NotImplementedError("Subclasses must implement the call method.")

  def summary_custom(self):
    """Custom summary method"""
    print(f"Model: {self.__class__.__name__}")
    print(f"Number of classes: {self.num_classes}")
    super().summary()

  def get_config(self):
    """Get model configuration"""
    config = super().get_config()
    config.update({
        'num_classes': self.num_classes
    })
    return config


class BaseDetectionModel(BaseModel):
    """Base class for object detection models"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
    
    @abstractmethod
    def _build_backbone(self):
        """Build the backbone network"""
        pass
    
    @abstractmethod
    def _build_detection_head(self):
        """Build the detection head"""
        pass
    
    @abstractmethod
    def _build_classification_head(self):
        """Build the classification head"""
        pass
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass for detection models"""
        # Extract features using backbone
        features = self.backbone(inputs, training=training)
        
        # Get bounding box predictions
        bbox_output = self.detection_head(features, training=training)
        
        # Get classification predictions
        class_output = self.classification_head(features, training=training)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output
        }
    
    def predict_single_image(self, image: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Predict on a single image"""
        # Add batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Get predictions
        predictions = self(image, training=False)
        
        # Remove batch dimension
        predictions = {
            key: tf.squeeze(value, 0) for key, value in predictions.items()
        }
        
        return predictions


class BaseSegmentationModel(BaseModel):
    """Base class for segmentation models"""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
    
    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        """Forward pass for segmentation models"""
        pass
    
    def predict_single_image(self, image: tf.Tensor) -> tf.Tensor:
        """Predict segmentation mask for a single image"""
        # Add batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Get prediction
        prediction = self(image, training=False)
        
        # Remove batch dimension
        prediction = tf.squeeze(prediction, 0)
        
        return prediction
    
    def predict_with_postprocessing(self, image: tf.Tensor) -> tf.Tensor:
        """Predict with postprocessing (argmax)"""
        prediction = self.predict_single_image(image)
        
        # Apply argmax to get class predictions
        mask = tf.argmax(prediction, axis=-1)
        
        return mask


class BaseMultitaskModel(BaseModel):
    """Base class for multitask models (detection + segmentation)"""
    
    def __init__(self, num_detection_classes: int, num_segmentation_classes: int, **kwargs):
        # Use detection classes as primary num_classes
        super().__init__(num_detection_classes, **kwargs)
        self.num_detection_classes = num_detection_classes
        self.num_segmentation_classes = num_segmentation_classes
    
    @abstractmethod
    def _build_backbone(self):
        """Build the backbone network"""
        pass
    
    @abstractmethod
    def _build_detection_head(self):
        """Build the detection head"""
        pass
    
    @abstractmethod
    def _build_segmentation_head(self):
        """Build the segmentation head"""
        pass
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass for multitask models"""
        # Extract features using backbone
        features = self.backbone(inputs, training=training)
        
        # Get detection predictions
        bbox_output = self.detection_head(features, training=training)
        class_output = self.classification_head(features, training=training)
        
        # Get segmentation predictions
        segmentation_output = self.segmentation_head(features, training=training)
        
        return {
            'bbox_output': bbox_output,
            'class_output': class_output,
            'segmentation_output': segmentation_output
        }
    
    def predict_single_image(self, image: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Predict on a single image"""
        # Add batch dimension
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        
        # Get predictions
        predictions = self(image, training=False)
        
        # Remove batch dimension
        predictions = {
            key: tf.squeeze(value, 0) if key != 'segmentation_output' 
            else tf.squeeze(value, 0)
            for key, value in predictions.items()
        }
        
        return predictions
    
    def get_config(self):
        """Get model configuration"""
        config = super().get_config()
        config.update({
            'num_detection_classes': self.num_detection_classes,
            'num_segmentation_classes': self.num_segmentation_classes
        })
        return config


class ModelBuilder:
    """Factory class for building models"""
    
    @staticmethod
    def build_detection_model(
        model_type: str, 
        num_classes: int, 
        config: Any = None,
        **kwargs
    ) -> BaseDetectionModel:
        """Build detection model"""
        from src.models.detecttion.detection_model import SimpleDetectionModel, PretrainedDetectionModel
        
        if model_type == 'simple':
            return SimpleDetectionModel(num_classes, config=config, **kwargs)
        elif model_type == 'pretrained':
            return PretrainedDetectionModel(
                num_classes, config=config, **kwargs
            )
        else:
            raise ValueError(f"Unknown detection model type: {model_type}")
    
    @staticmethod
    def build_segmentation_model(
        model_type: str, 
        num_classes: int, 
        config: Any = None,
        **kwargs
    ) -> BaseSegmentationModel:
        """Build segmentation model"""
        from src.models.segmentation.segmentation_model import PretrainedUNet, DeepLabV3Plus
        
        if model_type == 'simple_unet':
            return SimpleUNet(num_classes, config=config, **kwargs)
        elif model_type == 'pretrained_unet':
            return PretrainedUNet(
                num_classes, config=config, **kwargs
            )
        elif model_type == 'deeplabv3plus':
            return DeepLabV3Plus(
                num_classes, config=config, **kwargs
            )
        else:
            # Delegate to segmentation_model factory for additional SOTA architectures
            from src.models.segmentation.segmentation_model import create_segmentation_model
            try:
                return create_segmentation_model(model_type, num_classes, config=config, **kwargs)
            except Exception as e:
                raise ValueError(f"Unknown segmentation model type: {model_type}. {e}")
    
    @staticmethod
    def build_multitask_model(
        model_type: str,
        num_detection_classes: int,
        num_segmentation_classes: int,
        config: Any = None,
        **kwargs
    ) -> BaseMultitaskModel:
        """Build multitask model"""
        from src.models.multitask_model import MultitaskModel
        
        if model_type == 'resnet50_multitask':
            return MultitaskModel(
                num_detection_classes=num_detection_classes,
                num_segmentation_classes=num_segmentation_classes,
                backbone_name='resnet50',
                config=config,
                **kwargs
            )
        elif model_type == 'efficientnet_multitask':
            return MultitaskModel(
                num_detection_classes=num_detection_classes,
                num_segmentation_classes=num_segmentation_classes,
                backbone_name='efficientnetb0',
                config=config,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown multitask model type: {model_type}")