
# data/augmentation.py
import tensorflow as tf
from typing import Dict, Any
from config.config import Config

class DataAugmentor:
    """Data augmentation for detection and segmentation tasks."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.strength = self.config.AUGMENTATION_STRENGTH
    
    def random_flip_horizontal(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random horizontal flip to image, bbox, and mask."""
        if tf.random.uniform([]) > 0.5:
            # Flip image and mask
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(tf.expand_dims(mask, -1))
            mask = tf.squeeze(mask, -1)
            
            # Flip bounding box coordinates
            xmin, ymin, xmax, ymax = tf.unstack(bbox)
            width = 1.0  # Since bbox is normalized
            
            new_xmin = width - xmax
            new_xmax = width - xmin
            
            bbox = tf.stack([new_xmin, ymin, new_xmax, ymax])
        
        return image, bbox, mask
    
    def random_brightness(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random brightness adjustment."""
        return tf.image.random_brightness(image, max_delta=self.strength)
    
    def random_contrast(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random contrast adjustment."""
        return tf.image.random_contrast(image, lower=1-self.strength, upper=1+self.strength)
    
    def random_saturation(self, image: tf.Tensor) -> tf.Tensor:
        """Apply random saturation adjustment."""
        return tf.image.random_saturation(image, lower=1-self.strength, upper=1+self.strength)
    
    def augment_sample(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply augmentation to a sample."""
        if not self.config.ENABLE_AUGMENTATION:
            return image, bbox, mask
        
        # Geometric augmentations (affect all modalities)
        image, bbox, mask = self.random_flip_horizontal(image, bbox, mask)
        
        # Photometric augmentations (only affect image)
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_saturation(image)
        
        # Ensure image values stay in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, bbox, mask