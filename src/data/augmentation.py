import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
from src.config.config import Config

class DataAugmentor:
    """Optimized GPU-friendly data augmentation for detection/segmentation tasks."""
    
    def __init__(
        self,
        config: Config = None,
        prob_geo: float = 0.5,
        prob_photo: float = 0.7,
        prob_mixup: float = 0.3,
        prob_cutout: float = 0.2,
        prob_mosaic: float = 0.3,
        target_height: int = 224,
        target_width: int = 224,
    ):
        self.config = config or Config()
        self.prob_geo = prob_geo
        self.prob_photo = prob_photo
        self.prob_mixup = prob_mixup
        self.prob_cutout = prob_cutout
        self.prob_mosaic = prob_mosaic
        self.strength = getattr(self.config, 'AUGMENTATION_STRENGTH', 0.2)
        self.target_height = target_height
        self.target_width = target_width
        
        # Enhanced augmentation parameters
        self.rotation_range = 15.0
        self.zoom_range = (0.8, 1.2)
        self.shear_range = 0.1
        self.translate_range = 0.1
        
        # Photometric parameters
        self.brightness_range = 0.2
        self.contrast_range = 0.2
        self.saturation_range = 0.2
        self.hue_shift = 0.05

    def _normalize_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """Ensure bbox coordinates are within [0, 1] range."""
        bbox = tf.clip_by_value(bbox, 0.0, 1.0)
        
        # Ensure xmin < xmax and ymin < ymax
        bbox.set_shape([4])
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        xmin, xmax = tf.minimum(xmin, xmax), tf.maximum(xmin, xmax)
        ymin, ymax = tf.minimum(ymin, ymax), tf.maximum(ymin, ymax)
        
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    def _flip_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        """Flip bounding box coordinates horizontally."""
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        new_xmin = 1.0 - xmax
        new_xmax = 1.0 - xmin
        return tf.stack([new_xmin, ymin, new_xmax, ymax], axis=-1)
    
    def _ensure_mask_shape(self, mask: tf.Tensor) -> tf.Tensor:
        """Simplified mask shape handling with better performance."""
        # Convert to float32 first
        mask = tf.cast(mask, tf.float32)
        
        # Handle common cases efficiently
        mask_shape = tf.shape(mask)
        
        # If 2D, add channel dimension
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, -1)
        elif len(mask.shape) == 3 and mask.shape[-1] > 1:
            # Take first channel if multiple channels
            mask = tf.expand_dims(mask[..., 0], -1)
        elif len(mask.shape) > 3:
            # Squeeze extra dimensions and ensure 3D
            mask = tf.squeeze(mask)
            if len(mask.shape) == 2:
                mask = tf.expand_dims(mask, -1)
        
        # Resize if needed
        current_h, current_w = mask_shape[0], mask_shape[1]
        if current_h != self.target_height or current_w != self.target_width:
            mask = tf.image.resize(
                mask, 
                [self.target_height, self.target_width], 
                method='nearest'
            )
        
        # Ensure exact shape - THIS IS THE KEY FIX
        mask = tf.reshape(mask, [self.target_height, self.target_width, 1])
        return mask

    def _create_transform_matrix(self, angle: tf.Tensor = 0.0, 
                               scale_x: tf.Tensor = 1.0, scale_y: tf.Tensor = 1.0,
                               tx: tf.Tensor = 0.0, ty: tf.Tensor = 0.0,
                               shear_x: tf.Tensor = 0.0, shear_y: tf.Tensor = 0.0) -> tf.Tensor:
        """Create a combined transformation matrix for tf.raw_ops.ImageProjectiveTransformV3."""
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        
        # Create transformation matrix in the format expected by TensorFlow
        # The transform is applied as: [x', y'] = [[a, b], [c, d]] * [x, y] + [tx, ty]
        # We need to return the 8 coefficients: [a, b, c, d, e, f, g, h] where:
        # x' = (ax + by + e) / (gx + hy + 1)
        # y' = (cx + dy + f) / (gx + hy + 1)
        
        # For affine transformations, g=0, h=0, so:
        # x' = ax + by + e
        # y' = cx + dy + f
        
        a = scale_x * cos_a - shear_y * sin_a
        b = -scale_x * sin_a - shear_y * cos_a  
        c = scale_y * sin_a + shear_x * cos_a
        d = scale_y * cos_a - shear_x * sin_a
        e = tx * tf.cast(self.target_width, tf.float32)
        f = ty * tf.cast(self.target_height, tf.float32)
        g = 0.0
        h = 0.0
        
        return tf.stack([a, b, c, d, e, f, g, h])

    def _apply_transform_with_projective(self, image: tf.Tensor, mask: tf.Tensor, 
                                       transform_params: tf.Tensor, 
                                       interpolation_image: str = 'BILINEAR',
                                       interpolation_mask: str = 'NEAREST') -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply transformation using tf.raw_ops.ImageProjectiveTransformV3."""
        # Add batch dimension if needed
        image_batched = tf.expand_dims(image, 0) if len(image.shape) == 3 else image
        mask_batched = tf.expand_dims(mask, 0) if len(mask.shape) == 3 else mask
        
        # Add batch dimension to transform params
        transform_batched = tf.expand_dims(transform_params, 0)
        
        # Apply transformation to image
        transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=image_batched,
            transforms=transform_batched,
            output_shape=[self.target_height, self.target_width],
            interpolation=interpolation_image,
            fill_mode='REFLECT',
            fill_value=0.0
        )
        
        # Apply transformation to mask
        transformed_mask = tf.raw_ops.ImageProjectiveTransformV3(
            images=mask_batched,
            transforms=transform_batched,
            output_shape=[self.target_height, self.target_width],
            interpolation=interpolation_mask,
            fill_mode='CONSTANT',
            fill_value=0.0
        )
        
        # Remove batch dimension
        transformed_image = tf.squeeze(transformed_image, 0)
        transformed_mask = tf.squeeze(transformed_mask, 0)
        
        return transformed_image, transformed_mask

    def _apply_transform(self, image: tf.Tensor, mask: tf.Tensor, bbox: tf.Tensor,
                        transform_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply transformation to image, mask, and bbox efficiently."""
        # Apply transformation using the corrected projective transform
        transformed_image, transformed_mask = self._apply_transform_with_projective(
            image, mask, transform_matrix
        )
        
        # Transform bbox using the matrix approach
        transformed_bbox = self._transform_bbox_with_matrix(bbox, transform_matrix)
        
        # Return in (image, bbox, mask) order to match callers
        return transformed_image, transformed_bbox, transformed_mask

    def _transform_bbox_with_matrix(self, bbox: tf.Tensor, transform_params: tf.Tensor) -> tf.Tensor:
        """Transform bounding box using transformation parameters."""
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        
        # Convert normalized coordinates to pixel coordinates
        h_f = tf.cast(self.target_height, tf.float32)
        w_f = tf.cast(self.target_width, tf.float32)
        
        x1, y1 = xmin * w_f, ymin * h_f
        x2, y2 = xmax * w_f, ymax * h_f
        
        # Get all four corners of the bbox
        corners_x = tf.stack([x1, x2, x2, x1])
        corners_y = tf.stack([y1, y1, y2, y2])
        
        # Extract transformation parameters
        a, b, c, d, e, f = transform_params[0], transform_params[1], transform_params[2], transform_params[3], transform_params[4], transform_params[5]
        
        # Apply transformation to corners
        transformed_x = a * corners_x + b * corners_y + e
        transformed_y = c * corners_x + d * corners_y + f
        
        # Get new bounding box from transformed corners
        new_xmin = tf.reduce_min(transformed_x) / w_f
        new_xmax = tf.reduce_max(transformed_x) / w_f
        new_ymin = tf.reduce_min(transformed_y) / h_f
        new_ymax = tf.reduce_max(transformed_y) / h_f
        
        return self._normalize_bbox(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax], axis=-1))

    def _transform_bbox(self, bbox: tf.Tensor, transform_matrix: tf.Tensor) -> tf.Tensor:
        """Transform bounding box using transformation matrix (alternative method)."""
        return self._transform_bbox_with_matrix(bbox, transform_matrix)

    def random_flip_horizontal(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                             seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random horizontal flip."""
        should_flip = tf.random.stateless_uniform([], seed=seed) < 0.5
        
        def flip_fn():
            flipped_image = tf.image.flip_left_right(image)
            flipped_mask = tf.image.flip_left_right(mask)
            flipped_bbox = self._flip_bbox(bbox)
            return flipped_image, flipped_bbox, flipped_mask
        
        return tf.cond(should_flip, flip_fn, lambda: (image, bbox, mask))

    def random_geometric_transform(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                                 seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply combined geometric transformations for better performance."""
        seeds = tf.random.experimental.stateless_split(seed, 6)
        
        # Random rotation
        angle = tf.random.stateless_uniform(
            [], seed=seeds[0], 
            minval=-self.rotation_range * tf.constant(3.14159265359) / 180.0,
            maxval=self.rotation_range * tf.constant(3.14159265359) / 180.0
        )
        
        # Random scale
        scale = tf.random.stateless_uniform(
            [], seed=seeds[1], 
            minval=self.zoom_range[0], 
            maxval=self.zoom_range[1]
        )
        
        # Random translation
        tx = tf.random.stateless_uniform(
            [], seed=seeds[2], 
            minval=-self.translate_range, 
            maxval=self.translate_range
        )
        ty = tf.random.stateless_uniform(
            [], seed=seeds[3], 
            minval=-self.translate_range, 
            maxval=self.translate_range
        )
        
        # Random shear
        shear_x = tf.random.stateless_uniform(
            [], seed=seeds[4], 
            minval=-self.shear_range, 
            maxval=self.shear_range
        )
        shear_y = tf.random.stateless_uniform(
            [], seed=seeds[5], 
            minval=-self.shear_range, 
            maxval=self.shear_range
        )
        
        # Create combined transformation matrix
        transform_matrix = self._create_transform_matrix(
            angle=angle, scale_x=scale, scale_y=scale, 
            tx=tx, ty=ty, shear_x=shear_x, shear_y=shear_y
        )
        
        return self._apply_transform(image, mask, bbox, transform_matrix)

    def random_cutout(self, image: tf.Tensor, mask: tf.Tensor, seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Optimized random cutout implementation."""
        seeds = tf.random.experimental.stateless_split(seed, 4)
        
        # Random cutout parameters
        cutout_area = tf.random.stateless_uniform([], seed=seeds[0], minval=0.02, maxval=0.3)
        aspect_ratio = tf.random.stateless_uniform([], seed=seeds[1], minval=0.3, maxval=3.3)
        
        # Calculate cutout dimensions
        h_f = tf.cast(self.target_height, tf.float32)
        w_f = tf.cast(self.target_width, tf.float32)
        
        cutout_height = tf.sqrt(cutout_area * h_f * w_f / aspect_ratio)
        cutout_width = cutout_height * aspect_ratio
        
        # Ensure cutout fits in image
        cutout_height = tf.minimum(cutout_height, h_f)
        cutout_width = tf.minimum(cutout_width, w_f)
        
        # Random position
        cutout_x = tf.random.stateless_uniform([], seed=seeds[2], minval=0.0, maxval=w_f - cutout_width)
        cutout_y = tf.random.stateless_uniform([], seed=seeds[3], minval=0.0, maxval=h_f - cutout_height)
        
        # Convert to integers
        x1, y1 = tf.cast(cutout_x, tf.int32), tf.cast(cutout_y, tf.int32)
        x2, y2 = tf.cast(cutout_x + cutout_width, tf.int32), tf.cast(cutout_y + cutout_height, tf.int32)
        
        # Create cutout using tensor_scatter_nd_update
        cutout_mask = tf.ones([self.target_height, self.target_width, 1], dtype=tf.float32)
        
        # Create indices for the cutout region
        y_indices, x_indices = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing='ij')
        indices = tf.stack([
            tf.reshape(y_indices, [-1]),
            tf.reshape(x_indices, [-1]),
            tf.zeros(tf.size(y_indices), dtype=tf.int32)
        ], axis=1)
        
        # Zero out the cutout region
        updates = tf.zeros([tf.shape(indices)[0]], dtype=tf.float32)
        cutout_mask = tf.tensor_scatter_nd_update(cutout_mask, indices, updates)
        
        # Apply cutout
        cutout_image = image * cutout_mask
        
        return cutout_image, mask  # Keep segmentation mask intact

    def enhance_photometric_augmentation(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Optimized photometric augmentation pipeline."""
        seeds = tf.random.experimental.stateless_split(seed, 6)
        
        # Apply multiple augmentations in sequence
        image = tf.image.stateless_random_brightness(image, 0.2, seed=seeds[0])
        image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seed=seeds[1])
        image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seed=seeds[2])
        image = tf.image.stateless_random_hue(image, 0.05, seed=seeds[3])
        
        # Gamma correction
        gamma = tf.random.stateless_uniform([], seed=seeds[4], minval=0.8, maxval=1.2)
        image = tf.image.adjust_gamma(image, gamma)
        
        # Optional Gaussian noise
        if tf.random.stateless_uniform([], seed=seeds[5]) < 0.3:
            noise = tf.random.stateless_normal(tf.shape(image), seed=seeds[5], stddev=0.02)
            image = image + noise
        
        return tf.clip_by_value(image, 0.0, 1.0)

    def _preprocess_inputs(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Efficient input preprocessing."""
        # Normalize image to [0, 1] if needed
        image = tf.cast(image, tf.float32)
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0
        
        # Resize image
        image = tf.image.resize(image, [self.target_height, self.target_width])
        
        # Normalize bbox
        bbox = tf.cast(bbox, tf.float32)
        bbox = self._normalize_bbox(bbox)
        
        # Process mask
        mask = self._ensure_mask_shape(mask)
        
        return image, bbox, mask

    def augment_sample(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                      seed: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Streamlined augmentation pipeline."""
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            return self._preprocess_inputs(image, bbox, mask)
        
        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        elif tf.rank(seed) == 0:
            seed = tf.stack([seed, seed + 1])
        
        # Preprocess inputs
        image, bbox, mask = self._preprocess_inputs(image, bbox, mask)
        
        # Split seeds for different augmentations
        seeds = tf.random.experimental.stateless_split(seed, 6)
        
        # Horizontal flip
        if tf.random.stateless_uniform([], seed=seeds[0]) < self.prob_geo:
            image, bbox, mask = self.random_flip_horizontal(image, bbox, mask, seeds[1])
        
        # Combined geometric transformations
        if tf.random.stateless_uniform([], seed=seeds[2]) < self.prob_geo * 0.8:
            image, bbox, mask = self.random_geometric_transform(image, bbox, mask, seeds[3])
        
        # Photometric augmentations
        if tf.random.stateless_uniform([], seed=seeds[4]) < self.prob_photo:
            image = self.enhance_photometric_augmentation(image, seeds[4])
        
        # Cutout augmentation
        if tf.random.stateless_uniform([], seed=seeds[5]) < self.prob_cutout:
            image, mask = self.random_cutout(image, mask, seeds[5])
        
        # Final cleanup
        image = tf.clip_by_value(image, 0.0, 1.0)
        bbox = self._normalize_bbox(bbox)
        
        return image, bbox, mask

    def __call__(self, sample: Dict[str, tf.Tensor], seed: Optional[int] = None) -> Dict[str, tf.Tensor]:
        """Apply augmentation to a sample while maintaining output format."""
        # Handle augmentation toggle
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            image, bbox, mask = self._preprocess_inputs(
                sample['image'], 
                sample['head_bbox'], 
                sample['segmentation_mask']
            )
        else:
            # Prepare seed
            if seed is None:
                seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
            else:
                if isinstance(seed, int):
                    seed = tf.stack([tf.constant(seed, dtype=tf.int32), tf.constant(seed + 1, dtype=tf.int32)])
                else:
                    seed = tf.cast(seed, tf.int32)
                    if tf.rank(seed) == 0:
                        seed = tf.stack([seed, seed + 1])

            # Apply augmentation
            image, bbox, mask = self.augment_sample(
                sample['image'], 
                sample['head_bbox'], 
                sample['segmentation_mask'], 
                seed
            )

        # Ensure final shapes are explicit
        image = tf.ensure_shape(image, [self.target_height, self.target_width, 3])
        bbox = tf.ensure_shape(bbox, [4])
        mask = tf.ensure_shape(mask, [self.target_height, self.target_width, 1])
        
        # FIX 1: Cast label and species to match input dtypes
        label = tf.cast(sample['label'], tf.int64)  # Keep as int64
        species = tf.cast(sample['species'], tf.int64)  # Keep as int64
        
        # Return sample with same structure as input
        return {
            'image': image,
            'head_bbox': bbox,
            'segmentation_mask': mask,
            'label': label,  # Fixed: Keep as int64
            'species': species,  # Fixed: Keep as int64  
            'file_name': sample['file_name']
        }

    def get_output_signature(self) -> Dict[str, tf.TensorSpec]:
        """Get the output signature for dataset mapping - FIXED to match input signature."""
        return {
            'image': tf.TensorSpec(shape=(self.target_height, self.target_width, 3), dtype=tf.float32),
            'head_bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'segmentation_mask': tf.TensorSpec(shape=(self.target_height, self.target_width, 1), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64),  # Fixed: Changed from int32 to int64
            'species': tf.TensorSpec(shape=(), dtype=tf.int64),  # Fixed: Changed from string to int64
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string)
        }

    # Simplified batch methods for common use cases
    def create_augmented_dataset(self, dataset: tf.data.Dataset, 
                               augmentation_factor: int = 2) -> tf.data.Dataset:
        """Create augmented dataset with multiple versions of each sample."""
        def augment_fn(sample):
            base_seed = tf.random.uniform([], maxval=2**30, dtype=tf.int32)
            
            # Create multiple augmented versions
            augmented_samples = []
            for i in range(augmentation_factor):
                seed = base_seed + i
                aug_sample = self.__call__(sample, seed)
                augmented_samples.append(aug_sample)
            
            return tf.data.Dataset.from_tensor_slices({
                key: tf.stack([s[key] for s in augmented_samples])
                for key in augmented_samples[0].keys()
            })
        
        return dataset.flat_map(augment_fn)

    # Optional advanced methods (kept for compatibility)
    def mixup_augmentation(self, batch_data: Dict[str, tf.Tensor], seed: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Apply MixUp augmentation to a batch."""
        batch_size = tf.shape(batch_data['image'])[0]
        indices = tf.random.stateless_shuffle(tf.range(batch_size), seed=seed)
        alpha = tf.random.stateless_uniform([], seed=seed, minval=0.2, maxval=0.8)
        
        mixed_images = alpha * batch_data['image'] + (1 - alpha) * tf.gather(batch_data['image'], indices)
        mixed_masks = alpha * batch_data['segmentation_mask'] + (1 - alpha) * tf.gather(batch_data['segmentation_mask'], indices)
        
        return {
            'image': mixed_images,
            'head_bbox': batch_data['head_bbox'],  # Keep original bboxes
            'segmentation_mask': mixed_masks,
            'label': batch_data['label'],
            'species': batch_data['species'],
            'file_name': batch_data['file_name']
        }

    def mosaic_augmentation(self, images: tf.Tensor, bboxes: tf.Tensor, masks: tf.Tensor, 
                           seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create mosaic augmentation from 4 images."""
        half_h, half_w = self.target_height // 2, self.target_width // 2
        
        resized_images = tf.image.resize(images, [half_h, half_w])
        resized_masks = tf.image.resize(masks, [half_h, half_w])
        
        # Create mosaic layout
        top_row = tf.concat([resized_images[0], resized_images[1]], axis=1)
        bottom_row = tf.concat([resized_images[2], resized_images[3]], axis=1)
        mosaic_image = tf.concat([top_row, bottom_row], axis=0)
        
        top_mask_row = tf.concat([resized_masks[0], resized_masks[1]], axis=1)
        bottom_mask_row = tf.concat([resized_masks[2], resized_masks[3]], axis=1)
        mosaic_mask = tf.concat([top_mask_row, bottom_mask_row], axis=0)
        
        # Adjust bbox for top-left quadrant
        bbox = bboxes[0]
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        mosaic_bbox = tf.stack([xmin * 0.5, ymin * 0.5, xmax * 0.5, ymax * 0.5], axis=-1)
        
        return mosaic_image, mosaic_bbox, mosaic_mask