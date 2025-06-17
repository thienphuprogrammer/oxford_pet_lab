import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.config.config import Config

class DataPreprocessor:
    """Tối ưu hóa preprocessing cho Oxford Pet dataset: chuẩn hóa input và hỗ trợ nhiều task."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.img_size = getattr(self.config, 'IMG_SIZE', (224, 224))
        self.target_h, self.target_w = self.img_size
        self.use_imagenet_norm = getattr(self.config, 'USE_IMAGENET_NORM', True)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def ensure_image(self, image: tf.Tensor) -> tf.Tensor:
        """Chuẩn hóa ảnh về uint8/float32 [0,1] + 3 channel."""
        # Nếu ảnh là uint8: OK; Nếu float: chuẩn hóa [0,1]
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Đảm bảo 3 channel
        shape = tf.shape(image)
        if image.shape.rank == 2:
            image = tf.stack([image, image, image], axis=-1)
        elif image.shape.rank == 3 and image.shape[-1] == 1:
            image = tf.tile(image, [1, 1, 3])
        elif image.shape.rank == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def ensure_mask(self, mask: tf.Tensor) -> tf.Tensor:
        """Chuẩn hóa mask về [H, W, 1], kiểu float32 (có thể scale về 0/1/2 cho segmentation)."""
        mask = tf.cast(mask, tf.float32)
        if mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]
        elif mask.shape.rank == 3 and mask.shape[-1] != 1:
            mask = mask[..., 0:1]
        return mask

    def ensure_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """Đảm bảo bbox đúng shape (4,) và nằm trong [0,1]."""
        bbox = tf.cast(bbox, tf.float32)
        bbox = tf.reshape(bbox, [4])
        bbox = tf.clip_by_value(bbox, 0., 1.)
        return bbox

    def ensure_label(self, label) -> tf.Tensor:
        return tf.cast(label, tf.int64)

    def ensure_species(self, species) -> tf.Tensor:
        return tf.cast(species, tf.int64)

    def ensure_file_name(self, file_name) -> tf.Tensor:
        return tf.cast(file_name, tf.string)

    def resize_image(self, image: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(image, [self.target_h, self.target_w])

    def resize_mask(self, mask: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(mask, [self.target_h, self.target_w], method='nearest')

    def normalize_image(self, image: tf.Tensor) -> tf.Tensor:
        if self.use_imagenet_norm:
            image = (image - self.mean) / self.std
        else:
            image = tf.clip_by_value(image, 0., 1.)
        return image

    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        image = self.ensure_image(sample['image'])
        mask = self.ensure_mask(sample['segmentation_mask'])
        bbox = self.ensure_bbox(sample['head_bbox'])
        label = self.ensure_label(sample['label'])
        species = self.ensure_species(sample['species'])
        file_name = self.ensure_file_name(sample['file_name'])

        image = self.resize_image(image)
        mask = self.resize_mask(mask)
        image = self.normalize_image(image)

        return {
            'image': image,
            'head_bbox': bbox,
            'segmentation_mask': mask,
            'label': label,
            'species': species,
            'file_name': file_name,
        }

    def for_detection(self, sample: Dict[str, Any]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        data = self.preprocess_sample(sample)
        image = data['image']
        target = {
            'bbox': data['head_bbox'],
            'label': data['label'],
        }
        return image, target

    def for_segmentation(self, sample: Dict[str, Any]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        data = self.preprocess_sample(sample)
        image = data['image']
        target = {
            'mask': data['segmentation_mask'],
            'label': data['label'],
        }
        return image, target

    def for_classification(self, sample: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
        data = self.preprocess_sample(sample)
        return data['image'], data['label']

    def get_signature(self):
        """Lấy TensorSpec cho output (dùng cho tf.data)."""
        return {
            'image': tf.TensorSpec(shape=(self.target_h, self.target_w, 3), dtype=tf.float32),
            'head_bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'segmentation_mask': tf.TensorSpec(shape=(self.target_h, self.target_w, 1), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64),
            'species': tf.TensorSpec(shape=(), dtype=tf.int64),
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string),
        }
