# data/preprocessing.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
from src.config.config import Config

class DataPreprocessor:
    """Data preprocessing utilities for Oxford Pet dataset."""
    
    def __init__(self, config: Config = None, suffer_buffer: int = 1000):
        self.config = config or Config()
        self.suffer_buffer = suffer_buffer
    
    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        """Normalize image to [0, 1] range."""
        return tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=True)
    
    def resize_image(self, image: tf.Tensor, size: Tuple[int, int] = None) -> tf.Tensor:
        """Resize image to target size."""
        size = size or self.config.IMG_SIZE
        return tf.image.resize(image, size, antialias=True)
    
    def process_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """
        Process bounding box coordinates.
        Oxford Pet dataset provides normalized coordinates [ymin, xmin, ymax, xmax].
        Convert to [xmin, ymin, xmax, ymax] format and denormalize.
        """
        # bbox format in dataset: [ymin, xmin, ymax, xmax] (normalized)
        ymin, xmin, ymax, xmax = tf.unstack(bbox)
        return tf.stack([xmin, ymin, xmax, ymax])
    
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
        orig_h = tf.cast(tf.shape(sample['image'])[0], tf.float32)
        orig_w = tf.cast(tf.shape(sample['image'])[1], tf.float32)
        
        # Process image
        image = self.resize_image(sample['image'])
        image = self.normalize_image(image)
        
        # Process label (already in correct format)
        label = sample['label']
        
        # Process bounding box
        bbox_normalized = self.process_bbox(sample['bbox'])
        # Normalize bbox to [0, 1] range for the resized image
        bbox_denormalized = bbox_normalized * tf.stack([orig_w, orig_h, orig_w, orig_h])
        
        # Process segmentation mask
        seg_mask = self.process_segmentation_mask(sample['segmentation_mask'])
        
        return {
            'image': image,
            'label': label,
            'bbox': bbox_normalized,
            'segmentation_mask': seg_mask,
            'original_bbox': bbox_denormalized,  # Keep original for visualization
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

    def _format_for_task(self, processed_sample: Dict[str, tf.Tensor], task: str):
        """Return (x, y) depending on the chosen task."""
        if task == "detection":
            return self.create_detection_target(processed_sample)
        if task == "segmentation":
            return self.create_segmentation_target(processed_sample)
        # Default → multitask
        return self.create_multitask_target(processed_sample)
    
    def prepare_dataset(
        self,
        ds: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        task: str = "multitask",
    ) -> tf.data.Dataset:
        """
        Convert raw TFDS split into batched, pre-processed dataset
        ready for model.{fit,evaluate,predict}.
        """
        if shuffle:
            ds = ds.shuffle(self.suffer_buffer, seed=self.config.RANDOM_SEED)

        ds = ds.map(
            lambda s: self._format_for_task(self.preprocess_sample(s), task),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )

        return (
            ds.batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )