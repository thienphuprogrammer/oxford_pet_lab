
# data/augmentation.py
import tensorflow as tf
from typing import Tuple
from src.config.config import Config

class DataAugmentor:
    """GPU-friendly data augmentation for detection / segmentation."""
    
    def __init__(
        self,
        config: Config = None,
        prob_geo: float = 0.5,
        prob_photo: float = 0.5,
    ):
        self.config = config or Config()
        self.prob_geo = prob_geo
        self.prob_photo = prob_photo
        self.strength = self.config.AUGMENTATION_STRENGTH

    def  _flip_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """Flip bounding box coordinates."""
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        width = 1.0  # Since bbox is normalized
        new_xmin = width - xmax
        new_xmax = width - xmin
        return tf.stack([new_xmin, ymin, new_xmax, ymax])
    
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
    
    def __call__(
        self,
        image: tf.Tensor,
        bbox: tf.Tensor,
        mask: tf.Tensor,
        seed: int = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply augmentation to a sample."""
        if not self.config.ENABLE_AUGMENTATION:
            return image, bbox, mask

        # Split seeds once – ensures determinism while keeping stochasticity.
        s1, s2, s3 = tf.random.experimental.stateless_split(seed, 3)

        # ---------- Geometric: Horizontal flip --------------------------------
        should_flip = tf.random.stateless_uniform([], seed=s1) < self.prob_geo
        image = tf.cond(
            should_flip, lambda: tf.image.flip_left_right(image), lambda: image
        )
        mask = tf.cond(
            should_flip,
            lambda: tf.squeeze(
                tf.image.flip_left_right(tf.expand_dims(mask, -1)), -1
            ),
            lambda: mask,
        )
        bbox = tf.cond(should_flip, lambda: self._flip_bbox_h(bbox), lambda: bbox)

        # ---------- Photometric: brightness, contrast, saturation -------------
        apply_photo = tf.random.stateless_uniform([], seed=s2) < self.prob_photo
        def _photo_aug(img: tf.Tensor) -> tf.Tensor:
            img = tf.image.stateless_random_brightness(
                img, max_delta=self.strength, seed=s2
            )
            img = tf.image.stateless_random_contrast(
                img, lower=1 - self.strength, upper=1 + self.strength, seed=s3
            )
            img = tf.image.random_saturation(  # no stateless variant yet
                img, lower=1 - self.strength, upper=1 + self.strength
            )
            return img

        image = tf.cond(apply_photo, lambda: _photo_aug(image), lambda: image)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, bbox, mask