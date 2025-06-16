import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
from src.config.config import Config

class DataAugmentor:
    """Enhanced GPU-friendly data augmentation for detection/segmentation tasks."""
    
    def __init__(
        self,
        config: Config = None,
        prob_geo: float = 0.5,
        prob_photo: float = 0.7,
        prob_mixup: float = 0.3,
        prob_cutout: float = 0.2,
        prob_mosaic: float = 0.3,
        target_height: int = 512,  # Add explicit target dimensions
        target_width: int = 512,
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
        """Ensure mask has consistent 3D shape [H, W, 1] with defensive handling."""
        
        # Convert to float32 first
        mask = tf.cast(mask, tf.float32)
        
        # Get dynamic shape information
        mask_shape = tf.shape(mask)
        mask_rank = tf.rank(mask)
        
        # Handle 1D case - assume it's flattened
        def handle_1d():
            total_elements = mask_shape[0]
            # Try to make it square
            side_length = tf.cast(tf.sqrt(tf.cast(total_elements, tf.float32)), tf.int32)
            mask_2d = tf.reshape(mask, [side_length, side_length])
            return tf.expand_dims(mask_2d, -1)
        
        # Handle 2D case - add channel dimension
        def handle_2d():
            return tf.expand_dims(mask, -1)
        
        # Handle 3D case - check channels
        def handle_3d():
            channels = mask_shape[2]
            return tf.cond(
                channels > 1,
                lambda: tf.expand_dims(mask[..., 0], -1),  # Take first channel
                lambda: mask  # Already has 1 channel
            )
        
        # Handle 4D+ case - squeeze extra dimensions
        def handle_higher_d():
            # Squeeze out singleton dimensions except the last 3
            squeezed = tf.squeeze(mask)
            squeezed_rank = tf.rank(squeezed)
            
            return tf.cond(
                tf.equal(squeezed_rank, 2),
                lambda: tf.expand_dims(squeezed, -1),
                lambda: tf.cond(
                    tf.equal(squeezed_rank, 3),
                    lambda: squeezed,
                    lambda: tf.expand_dims(tf.reshape(squeezed, [self.target_height, self.target_width]), -1)
                )
            )
        
        # Apply appropriate handler based on rank
        mask_3d = tf.case([
            (tf.equal(mask_rank, 1), handle_1d),
            (tf.equal(mask_rank, 2), handle_2d),
            (tf.equal(mask_rank, 3), handle_3d),
        ], default=handle_higher_d)
        
        # Now handle resizing with known 3D shape
        current_shape = tf.shape(mask_3d)
        current_h, current_w = current_shape[0], current_shape[1]
        
        # Check if we need to resize
        needs_resize = tf.logical_or(
            tf.not_equal(current_h, self.target_height),
            tf.not_equal(current_w, self.target_width)
        )
        
        def do_resize():
            # Set shape hints to help TensorFlow
            mask_with_hints = tf.ensure_shape(mask_3d, [None, None, 1])
            return tf.image.resize(
                mask_with_hints, 
                [self.target_height, self.target_width], 
                method='nearest'
            )
        
        def no_resize():
            return mask_3d
        
        # Apply resize if needed
        final_mask = tf.cond(needs_resize, do_resize, no_resize)
        
        # Final safety reshape to guarantee exact dimensions
        final_mask = tf.reshape(final_mask, [self.target_height, self.target_width, 1])
        
        return final_mask


    def _apply_affine_transform(self, image: tf.Tensor, transform_matrix: tf.Tensor, 
                           interpolation: str = 'bilinear') -> tf.Tensor:
        """Apply affine transformation using tf.raw_ops.ImageProjectiveTransformV3."""
        # Convert 3x3 matrix to 8-element projective transform
        transforms = tf.stack([
            transform_matrix[0, 0], transform_matrix[0, 1], transform_matrix[0, 2],
            transform_matrix[1, 0], transform_matrix[1, 1], transform_matrix[1, 2],
            transform_matrix[2, 0], transform_matrix[2, 1]
        ])
        
        # Expand dims for batch processing
        image_batch = tf.expand_dims(image, 0)
        transforms_batch = tf.expand_dims(transforms, 0)
        
        # Get output shape
        output_shape = tf.stack([self.target_height, self.target_width])
        
        # Apply transformation with proper parameters
        transformed = tf.raw_ops.ImageProjectiveTransformV3(
            images=image_batch,
            transforms=transforms_batch,
            output_shape=output_shape,
            interpolation=interpolation.upper(),
            fill_mode='REFLECT',
            fill_value=0.0
        )
        
        return tf.squeeze(transformed, 0)

    def _transform_bbox(self, bbox: tf.Tensor, transform_matrix: tf.Tensor) -> tf.Tensor:
        """Transform bounding box using transformation matrix."""
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        
        # Get all four corners of the bbox
        corners = tf.stack([
            [xmin, ymin, 1.0],
            [xmax, ymin, 1.0],
            [xmax, ymax, 1.0],
            [xmin, ymax, 1.0]
        ])
        
        # Apply transformation
        transformed_corners = tf.linalg.matmul(corners, transform_matrix, transpose_b=True)
        
        # Get new bounding box from transformed corners
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        
        new_xmin = tf.reduce_min(x_coords)
        new_xmax = tf.reduce_max(x_coords)
        new_ymin = tf.reduce_min(y_coords)
        new_ymax = tf.reduce_max(y_coords)
        
        return self._normalize_bbox(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax], axis=-1))

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

    def random_rotation(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                       seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random rotation using built-in TensorFlow functions."""
        angle_degrees = tf.random.stateless_uniform(
            [], seed=seed, 
            minval=-self.rotation_range,
            maxval=self.rotation_range
        )
        
        angle_radians = angle_degrees * tf.constant(3.14159265359) / 180.0
        
        # Get image center
        height = tf.cast(self.target_height, tf.float32)
        width = tf.cast(self.target_width, tf.float32)
        center_x, center_y = width / 2.0, height / 2.0
        
        # Create rotation matrix around center
        cos_a = tf.cos(angle_radians)
        sin_a = tf.sin(angle_radians)
        
        # Rotation matrix with translation to rotate around center
        transform_matrix = tf.stack([
            [cos_a, -sin_a, center_x * (1 - cos_a) + center_y * sin_a],
            [sin_a, cos_a, center_y * (1 - cos_a) - center_x * sin_a],
            [0.0, 0.0, 1.0]
        ])
        
        # Apply transformation
        rotated_image = self._apply_affine_transform(image, transform_matrix, 'bilinear')
        rotated_mask = self._apply_affine_transform(mask, transform_matrix, 'nearest')
        
        # Transform bbox coordinates (normalize coordinates first)
        normalized_transform = tf.stack([
            [cos_a, -sin_a, (center_x * (1 - cos_a) + center_y * sin_a) / width],
            [sin_a, cos_a, (center_y * (1 - cos_a) - center_x * sin_a) / height],
            [0.0, 0.0, 1.0]
        ])
        
        rotated_bbox = self._transform_bbox(bbox, normalized_transform)
        
        return rotated_image, rotated_bbox, rotated_mask

    def random_scale_and_translate(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                                 seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random scaling and translation."""
        seeds = tf.random.experimental.stateless_split(seed, 3)
        
        # Random scale
        scale = tf.random.stateless_uniform(
            [], seed=seeds[0], 
            minval=self.zoom_range[0], 
            maxval=self.zoom_range[1]
        )
        
        # Random translation
        height = tf.cast(self.target_height, tf.float32)
        width = tf.cast(self.target_width, tf.float32)
        
        tx = tf.random.stateless_uniform(
            [], seed=seeds[1], 
            minval=-self.translate_range * width, 
            maxval=self.translate_range * width
        )
        ty = tf.random.stateless_uniform(
            [], seed=seeds[2], 
            minval=-self.translate_range * height, 
            maxval=self.translate_range * height
        )
        
        # Create transformation matrix
        transform_matrix = tf.stack([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0]
        ])
        
        # Apply transformation
        transformed_image = self._apply_affine_transform(image, transform_matrix, 'bilinear')
        transformed_mask = self._apply_affine_transform(mask, transform_matrix, 'nearest')
        
        # Transform bbox (normalize translation)
        normalized_transform = tf.stack([
            [scale, 0.0, tx / width],
            [0.0, scale, ty / height],
            [0.0, 0.0, 1.0]
        ])
        
        transformed_bbox = self._transform_bbox(bbox, normalized_transform)
        
        return transformed_image, transformed_bbox, transformed_mask

    def random_shear(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                    seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random shear transformation."""
        seeds = tf.random.experimental.stateless_split(seed, 2)
        
        shear_x = tf.random.stateless_uniform(
            [], seed=seeds[0], 
            minval=-self.shear_range, 
            maxval=self.shear_range
        )
        shear_y = tf.random.stateless_uniform(
            [], seed=seeds[1], 
            minval=-self.shear_range, 
            maxval=self.shear_range
        )
        
        # Shear transformation matrix
        transform_matrix = tf.stack([
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Apply transformation
        sheared_image = self._apply_affine_transform(image, transform_matrix, 'bilinear')
        sheared_mask = self._apply_affine_transform(mask, transform_matrix, 'nearest')
        
        sheared_bbox = self._transform_bbox(bbox, transform_matrix)
        
        return sheared_image, sheared_bbox, sheared_mask

    def random_cutout(self, image: tf.Tensor, mask: tf.Tensor, seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply random cutout/erasing with improved implementation."""
        seeds = tf.random.experimental.stateless_split(seed, 4)
        
        # Get image dimensions
        height = tf.cast(self.target_height, tf.float32)
        width = tf.cast(self.target_width, tf.float32)
        
        # Random cutout parameters
        cutout_area = tf.random.stateless_uniform([], seed=seeds[0], minval=0.02, maxval=0.3)
        aspect_ratio = tf.random.stateless_uniform([], seed=seeds[1], minval=0.3, maxval=3.3)
        
        # Calculate cutout dimensions
        cutout_height = tf.sqrt(cutout_area * height * width / aspect_ratio)
        cutout_width = cutout_height * aspect_ratio
        
        # Ensure cutout fits in image
        cutout_height = tf.minimum(cutout_height, height)
        cutout_width = tf.minimum(cutout_width, width)
        
        # Random position
        cutout_x = tf.random.stateless_uniform([], seed=seeds[2], minval=0.0, maxval=width - cutout_width)
        cutout_y = tf.random.stateless_uniform([], seed=seeds[3], minval=0.0, maxval=height - cutout_height)
        
        # Convert to integers
        x1 = tf.cast(cutout_x, tf.int32)
        y1 = tf.cast(cutout_y, tf.int32)
        x2 = tf.cast(cutout_x + cutout_width, tf.int32)
        y2 = tf.cast(cutout_y + cutout_height, tf.int32)
        
        # Create cutout mask using tf.pad and slicing approach
        # This is more efficient than tensor_scatter_nd_update
        cutout_mask = tf.ones([self.target_height, self.target_width], dtype=tf.float32)
        
        # Create zero region for cutout
        y_indices = tf.range(self.target_height)
        x_indices = tf.range(self.target_width)
        
        y_mask = tf.logical_and(y_indices >= y1, y_indices < y2)
        x_mask = tf.logical_and(x_indices >= x1, x_indices < x2)
        
        # Create 2D mask
        y_mask_2d = tf.reshape(y_mask, [-1, 1])
        x_mask_2d = tf.reshape(x_mask, [1, -1])
        region_mask = tf.logical_and(y_mask_2d, x_mask_2d)
        
        # Apply cutout
        cutout_mask = tf.where(region_mask, 0.0, cutout_mask)
        cutout_mask = tf.expand_dims(cutout_mask, -1)  # Add channel dimension
        
        # Apply to image and mask
        cutout_image = image * cutout_mask
        # For segmentation mask, we might want to preserve it or zero it out
        # Here we preserve the mask structure
        cutout_seg_mask = mask  # Keep segmentation mask intact
        
        return cutout_image, cutout_seg_mask

    def enhance_photometric_augmentation(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Enhanced photometric augmentation with better control."""
        seeds = tf.random.experimental.stateless_split(seed, 8)
        
        # Brightness
        brightness_delta = tf.random.stateless_uniform(
            [], seed=seeds[0], minval=-self.brightness_range, maxval=self.brightness_range
        )
        image = tf.image.adjust_brightness(image, brightness_delta)
        
        # Contrast
        contrast_factor = tf.random.stateless_uniform(
            [], seed=seeds[1], minval=1-self.contrast_range, maxval=1+self.contrast_range
        )
        image = tf.image.adjust_contrast(image, contrast_factor)
        
        # Saturation
        saturation_factor = tf.random.stateless_uniform(
            [], seed=seeds[2], minval=1-self.saturation_range, maxval=1+self.saturation_range
        )
        image = tf.image.adjust_saturation(image, saturation_factor)
        
        # Hue
        hue_delta = tf.random.stateless_uniform(
            [], seed=seeds[3], minval=-self.hue_shift, maxval=self.hue_shift
        )
        image = tf.image.adjust_hue(image, hue_delta)
        
        # Gamma correction
        gamma = tf.random.stateless_uniform([], seed=seeds[4], minval=0.8, maxval=1.2)
        image = tf.image.adjust_gamma(image, gamma)
        
        # Gaussian noise
        if tf.random.stateless_uniform([], seed=seeds[5]) < 0.3:
            noise_stddev = tf.random.stateless_uniform([], seed=seeds[6], minval=0.0, maxval=0.1)
            noise = tf.random.stateless_normal(tf.shape(image), seed=seeds[7], stddev=noise_stddev)
            image = image + noise
        
        return tf.clip_by_value(image, 0.0, 1.0)

    def mixup_augmentation(self, batch_data: Dict[str, tf.Tensor], seed: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Apply MixUp augmentation to a batch."""
        batch_size = tf.shape(batch_data['image'])[0]
        
        # Generate random permutation
        indices = tf.random.stateless_shuffle(tf.range(batch_size), seed=seed)
        
        # Generate mixing coefficient
        alpha = tf.random.stateless_uniform([], seed=seed, minval=0.2, maxval=0.8)
        
        # Mix images
        mixed_images = alpha * batch_data['image'] + (1 - alpha) * tf.gather(batch_data['image'], indices)
        
        # Mix masks
        mixed_masks = alpha * batch_data['segmentation_mask'] + (1 - alpha) * tf.gather(batch_data['segmentation_mask'], indices)
        
        # For bboxes, we keep the original ones (more complex mixing would require label handling)
        mixed_bboxes = batch_data['head_bbox']
        
        return {
            'image': mixed_images,
            'head_bbox': mixed_bboxes,
            'segmentation_mask': mixed_masks,
            'label': batch_data['label'],
            'species': batch_data['species'],
            'file_name': batch_data['file_name']
        }

    def mosaic_augmentation(self, images: tf.Tensor, bboxes: tf.Tensor, masks: tf.Tensor, 
                           seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create mosaic augmentation from 4 images."""
        # Resize images to half target size
        half_height = self.target_height // 2
        half_width = self.target_width // 2
        
        resized_images = tf.image.resize(images, [half_height, half_width])
        resized_masks = tf.image.resize(masks, [half_height, half_width])
        
        # Create mosaic
        top_row = tf.concat([resized_images[0], resized_images[1]], axis=1)
        bottom_row = tf.concat([resized_images[2], resized_images[3]], axis=1)
        mosaic_image = tf.concat([top_row, bottom_row], axis=0)
        
        # Create mosaic mask
        top_mask_row = tf.concat([resized_masks[0], resized_masks[1]], axis=1)
        bottom_mask_row = tf.concat([resized_masks[2], resized_masks[3]], axis=1)
        mosaic_mask = tf.concat([top_mask_row, bottom_mask_row], axis=0)
        
        # Adjust bboxes for first quadrant (top-left)
        bbox = bboxes[0]
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        
        # Scale and position in top-left quadrant
        mosaic_xmin = xmin * 0.5
        mosaic_ymin = ymin * 0.5
        mosaic_xmax = xmax * 0.5
        mosaic_ymax = ymax * 0.5
        
        mosaic_bbox = tf.stack([mosaic_xmin, mosaic_ymin, mosaic_xmax, mosaic_ymax], axis=-1)
        
        return mosaic_image, mosaic_bbox, mosaic_mask

    def _preprocess_inputs(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Preprocess inputs to ensure consistent shapes and types."""
        # Ensure inputs are float32
        image = tf.cast(image, tf.float32)
        if tf.reduce_max(image) > 1.0:
            image = image / 255.0  # Normalize if uint8
        
        # Resize image to target dimensions
        image = tf.image.resize(image, [self.target_height, self.target_width])
        
        # Ensure bbox is float32 and normalized
        bbox = tf.cast(bbox, tf.float32)
        bbox = self._normalize_bbox(bbox)
        
        # Process mask to ensure consistent shape
        mask = self._ensure_mask_shape(mask)
        
        return image, bbox, mask

    def augment_sample(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                      seed: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply comprehensive augmentation pipeline."""
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            # Still preprocess to ensure consistent shapes
            return self._preprocess_inputs(image, bbox, mask)
        
        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        elif tf.rank(seed) == 0:
            # Convert scalar seed to 2-element vector
            seed = tf.stack([seed, seed + 1])
        
        # Preprocess inputs
        image, bbox, mask = self._preprocess_inputs(image, bbox, mask)
        
        # Split seeds for different augmentations
        seeds = tf.random.experimental.stateless_split(seed, 12)
        
        # Geometric augmentations
        if tf.random.stateless_uniform([], seed=seeds[0]) < self.prob_geo:
            image, bbox, mask = self.random_flip_horizontal(image, bbox, mask, seeds[1])
        
        if tf.random.stateless_uniform([], seed=seeds[2]) < self.prob_geo * 0.6:
            image, bbox, mask = self.random_rotation(image, bbox, mask, seeds[3])
        
        if tf.random.stateless_uniform([], seed=seeds[4]) < self.prob_geo * 0.8:
            image, bbox, mask = self.random_scale_and_translate(image, bbox, mask, seeds[5])
        
        if tf.random.stateless_uniform([], seed=seeds[6]) < self.prob_geo * 0.4:
            image, bbox, mask = self.random_shear(image, bbox, mask, seeds[7])
        
        # Photometric augmentations
        if tf.random.stateless_uniform([], seed=seeds[8]) < self.prob_photo:
            image = self.enhance_photometric_augmentation(image, seeds[9])
        
        # Cutout augmentation
        if tf.random.stateless_uniform([], seed=seeds[10]) < self.prob_cutout:
            image, mask = self.random_cutout(image, mask, seeds[11])
        
        # Ensure final values are in valid range and have correct shapes
        image = tf.clip_by_value(image, 0.0, 1.0)
        bbox = self._normalize_bbox(bbox)
        mask = self._ensure_mask_shape(mask)
        
        return image, bbox, mask

    def __call__(self, sample: Dict[str, tf.Tensor], seed: Optional[int] = None) -> Dict[str, tf.Tensor]:
        """Apply augmentation to a sample."""
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            # Still preprocess to ensure consistent shapes
            image, bbox, mask = self._preprocess_inputs(
                sample['image'], 
                sample['head_bbox'], 
                sample['segmentation_mask']
            )
            return {
                'image': image,
                'head_bbox': bbox,
                'segmentation_mask': mask,
                'label': sample['label'],
                'species': sample['species'],
                'file_name': sample['file_name']
            }

        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        else:
            # Handle both scalar integers and tensor seeds
            if isinstance(seed, int):
                # If seed is a Python int, convert to tensor
                seed = tf.stack([tf.constant(seed, dtype=tf.int32), tf.constant(seed + 1, dtype=tf.int32)])
            else:
                # If seed is already a tensor, use it directly
                seed = tf.cast(seed, tf.int32)
                if tf.rank(seed) == 0:
                    # Convert scalar tensor seed to 2-element vector
                    seed = tf.stack([seed, seed + 1])

        # Extract components
        image = sample['image']
        bbox = sample['head_bbox']
        mask = sample['segmentation_mask']
        
        # Apply augmentation
        aug_image, aug_bbox, aug_mask = self.augment_sample(image, bbox, mask, seed)
        
        # Return augmented sample
        return {
            'image': aug_image,
            'head_bbox': aug_bbox,
            'segmentation_mask': aug_mask,
            'label': sample['label'],
            'species': sample['species'],
            'file_name': sample['file_name']
        }

    def get_output_signature(self) -> Dict[str, tf.TensorSpec]:
        """Get the output signature for dataset mapping."""
        return {
            'image': tf.TensorSpec(shape=(self.target_height, self.target_width, 3), dtype=tf.float32),
            'head_bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'segmentation_mask': tf.TensorSpec(shape=(self.target_height, self.target_width, 1), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(), dtype=tf.int32),
            'species': tf.TensorSpec(shape=(), dtype=tf.string),
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string)
        }

    def create_augmented_dataset(self, dataset: tf.data.Dataset, 
                               augmentation_factor: int = 2) -> tf.data.Dataset:
        """Create augmented dataset with multiple versions of each sample."""
        def augment_fn(sample):
            augmented_samples = []
            base_seed = tf.random.uniform([], maxval=2**30, dtype=tf.int32)
            
            for i in range(augmentation_factor):
                seed = base_seed + i
                aug_sample = self.__call__(sample, seed)
                augmented_samples.append(aug_sample)
            
            # Stack all samples
            stacked_sample = {}
            for key in augmented_samples[0].keys():
                stacked_sample[key] = tf.stack([s[key] for s in augmented_samples])
            
            return stacked_sample
        
        # Apply augmentation with proper output signature
        output_signature = {}
        sample_signature = self.get_output_signature()
        for key, spec in sample_signature.items():
            # Add batch dimension for stacked samples
            new_shape = [augmentation_factor] + list(spec.shape)
            output_signature[key] = tf.TensorSpec(shape=new_shape, dtype=spec.dtype)
        
        augmented_dataset = dataset.map(
            augment_fn, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Flatten the dataset
        def flatten_fn(batch):
            return tf.data.Dataset.from_tensor_slices(batch)
        
        return augmented_dataset.flat_map(flatten_fn)
