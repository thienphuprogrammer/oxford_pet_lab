# data/preprocessing.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
from config.config import Config

class DataPreprocessor:
    """Data preprocessing utilities for Oxford Pet dataset."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """Normalize image to [0, 1] range."""
        return tf.cast(image, tf.float32) / 255.0
    
    def resize_image(self, image: tf.Tensor, size: Tuple[int, int] = None) -> tf.Tensor:
        """Resize image to target size."""
        size = size or self.config.IMG_SIZE
        return tf.image.resize(image, size)
    
    def process_bbox(self, bbox: tf.Tensor, image_shape: tf.Tensor) -> tf.Tensor:
        """
        Process bounding box coordinates.
        Oxford Pet dataset provides normalized coordinates [ymin, xmin, ymax, xmax].
        Convert to [xmin, ymin, xmax, ymax] format and denormalize.
        """
        # bbox format in dataset: [ymin, xmin, ymax, xmax] (normalized)
        ymin, xmin, ymax, xmax = tf.unstack(bbox)
        
        # Convert to [xmin, ymin, xmax, ymax] format
        bbox_reordered = tf.stack([xmin, ymin, xmax, ymax])
        
        # Denormalize coordinates
        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)
        
        scale = tf.stack([width, height, width, height])
        bbox_denorm = bbox_reordered * scale
        
        return bbox_denorm
    
    def process_segmentation_mask(self, mask: tf.Tensor) -> tf.Tensor:
        """
        Process segmentation mask.
        Oxford Pet dataset provides trimap: 1=foreground, 2=background, 3=not classified
        Convert to: 0=background, 1=foreground, 2=unknown
        """
        # Resize mask to target size
        mask = tf.image.resize(
            tf.expand_dims(mask, -1), 
            self.config.IMG_SIZE, 
            method='nearest'
        )
        mask = tf.squeeze(mask, -1)
        
        # Convert from 1,2,3 to 0,1,2
        mask = mask - 1
        mask = tf.clip_by_value(mask, 0, 2)
        
        return tf.cast(mask, tf.int32)
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """
        Preprocess a single sample from the dataset.
        
        Args:
            sample: Dictionary containing 'image', 'label', 'bbox', 'segmentation_mask'
            
        Returns:
            Preprocessed sample dictionary
        """
        # Get original image shape for bbox processing
        original_shape = tf.shape(sample['image'])
        
        # Process image
        image = self.resize_image(sample['image'])
        image = self.normalize_image(image)
        
        # Process label (already in correct format)
        label = sample['label']
        
        # Process bounding box
        bbox = self.process_bbox(sample['bbox'], original_shape)
        # Normalize bbox to [0, 1] range for the resized image
        bbox_normalized = bbox / tf.cast(tf.stack([
            self.config.IMG_WIDTH, self.config.IMG_HEIGHT,
            self.config.IMG_WIDTH, self.config.IMG_HEIGHT
        ]), tf.float32)
        
        # Process segmentation mask
        seg_mask = self.process_segmentation_mask(sample['segmentation_mask'])
        
        return {
            'image': image,
            'label': label,
            'bbox': bbox_normalized,
            'segmentation_mask': seg_mask,
            'original_bbox': bbox,  # Keep original for visualization
        }
    
    def create_detection_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Create targets for detection task."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'label': processed_sample['label'],
        }
        return image, targets
    
    def create_segmentation_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create targets for segmentation task."""
        image = processed_sample['image']
        mask = processed_sample['segmentation_mask']
        return image, mask
    
    def create_multitask_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Create targets for multitask learning."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'label': processed_sample['label'],
            'segmentation_mask': processed_sample['segmentation_mask'],
        }
        return image, targets