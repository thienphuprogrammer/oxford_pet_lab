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
        prob_mosaic: float = 0.3,  # New: Mosaic augmentation
    ):
        self.config = config or Config()
        self.prob_geo = prob_geo
        self.prob_photo = prob_photo
        self.prob_mixup = prob_mixup
        self.prob_cutout = prob_cutout
        self.prob_mosaic = prob_mosaic
        self.strength = getattr(self.config, 'AUGMENTATION_STRENGTH', 0.2)
        
        # Enhanced augmentation parameters
        self.rotation_range = 15.0
        self.zoom_range = (0.8, 1.2)  # More flexible zoom range
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

    def rotate_image_tf(self, image, angle, interpolation='bilinear'):
        # image: [H, W, C]
        angle = -angle  # tf uses negative for counterclockwise
        # Get shape
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        # Convert radians to degrees
        angle_deg = angle * 180.0 / tf.constant(3.14159265359)
        # Center of rotation
        cx, cy = tf.cast(w, tf.float32) / 2.0, tf.cast(h, tf.float32) / 2.0

        # Make transformation matrix for rotate about center
        angle_rad = angle
        cos_a = tf.cos(angle_rad)
        sin_a = tf.sin(angle_rad)
        one = tf.constant(1, tf.float32)
        zero = tf.constant(0, tf.float32)

        # Translate so center is at origin, rotate, then translate back
        # [x', y', 1] = M * [x, y, 1]
        # Where M = T(-cx,-cy) * R(angle) * T(cx,cy)
        tx = (one - cos_a) * cx - sin_a * cy
        ty = sin_a * cx + (one - cos_a) * cy
        transform = tf.stack([
            cos_a, -sin_a, tx,
            sin_a, cos_a, ty,
            zero, zero
        ])
        # tf.raw_ops.ImageProjectiveTransformV3 expects shape [N, 8]
        transform = tf.expand_dims(transform, 0)
        image = tf.expand_dims(image, 0)
        return tf.squeeze(
            tf.raw_ops.ImageProjectiveTransformV3(
                images=image,
                fill_value=0,
                transforms=tf.cast(transform, tf.float32),
                interpolation=interpolation.upper(),
                output_shape=tf.shape(image)[1:3]
            ),
            0
        )

    def random_rotation(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                       seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply random rotation using TensorFlow Addons."""
        angle = tf.random.stateless_uniform(
            [], seed=seed, 
            minval=-self.rotation_range * tf.constant(3.14159265359) / 180.0,
            maxval=self.rotation_range * tf.constant(3.14159265359) / 180.0
        )
        
        # Rotate image and mask
        rotated_image = self.rotate_image_tf(image, angle, interpolation='bilinear')
        rotated_mask = self.rotate_image_tf(mask, angle, interpolation='nearest')
        
        # Create rotation matrix
        cos_a = tf.cos(angle)
        sin_a = tf.sin(angle)
        
        # Rotation matrix for 2D transformation (with translation to center)
        transform_matrix = tf.stack([
            [cos_a, -sin_a, 0.5 * (1 - cos_a + sin_a)],
            [sin_a, cos_a, 0.5 * (1 - cos_a - sin_a)],
            [0.0, 0.0, 1.0]
        ])
        
        rotated_bbox = self._transform_bbox(bbox, transform_matrix)
        
        return rotated_image, rotated_bbox, rotated_mask

    def affine_matrix_to_flat_tf(self, matrix):
        # matrix: [3,3] hoặc [N,3,3]
        # trả về: [8] hoặc [N,8]
        if len(matrix.shape) == 2:
            # [3,3]
            matrix = tf.reshape(matrix, [3,3])
            flat = [
                matrix[0,0], matrix[0,1], matrix[0,2],
                matrix[1,0], matrix[1,1], matrix[1,2],
                matrix[2,0], matrix[2,1]
            ]
            return tf.stack(flat)
        else:
            # [N,3,3]
            m = matrix
            flat = tf.stack([
                m[:,0,0], m[:,0,1], m[:,0,2],
                m[:,1,0], m[:,1,1], m[:,1,2],
                m[:,2,0], m[:,2,1]
            ], axis=1)
            return flat
        

    def transform_image_tf(self, image, transform_matrix, interpolation='bilinear'):
        # image: [H, W, C], transform_matrix: [3,3]
        flat = self.affine_matrix_to_flat_tf(transform_matrix)
        flat = tf.expand_dims(flat, 0)
        image = tf.expand_dims(image, 0)
        out = tf.raw_ops.ImageProjectiveTransformV3(
            images=image,
            transforms=tf.cast(flat, tf.float32),
            interpolation=interpolation.upper(),
            output_shape=tf.shape(image)[1:3],
            fill_value=0
        )
        return tf.squeeze(out, 0)


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
        tx = tf.random.stateless_uniform(
            [], seed=seeds[1], 
            minval=-self.translate_range, 
            maxval=self.translate_range
        )
        ty = tf.random.stateless_uniform(
            [], seed=seeds[2], 
            minval=-self.translate_range, 
            maxval=self.translate_range
        )
        
        # Create transformation matrix
        transform_matrix = tf.stack([
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0]
        ])
        
        # Apply transformation
        transformed_image = self.transform_image_tf(image, transform_matrix, interpolation='bilinear')
        
        transformed_mask = self.transform_image_tf(mask, transform_matrix, interpolation='nearest')
        
        transformed_bbox = self._transform_bbox(bbox, transform_matrix)
        
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
        sheared_image = self.transform_image_tf(image, transform_matrix, interpolation='bilinear')
        
        sheared_mask = self.transform_image_tf(mask, transform_matrix, interpolation='nearest')
        
        sheared_bbox = self._transform_bbox(bbox, transform_matrix)
        
        return sheared_image, sheared_bbox, sheared_mask

    def random_cutout(self, image: tf.Tensor, mask: tf.Tensor, seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply random cutout/erasing with improved implementation."""
        seeds = tf.random.experimental.stateless_split(seed, 4)
        
        # Get image dimensions
        height_int = tf.shape(image)[0]
        width_int = tf.shape(image)[1]
        
        # Cast to float32 for subsequent arithmetic operations to avoid dtype mismatch
        height = tf.cast(height_int, tf.float32)
        width = tf.cast(width_int, tf.float32)
        
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
        
        # Create cutout mask
        cutout_mask = tf.ones([height_int, width_int], dtype=tf.float32)
        
        # Create indices for the cutout region
        y_indices, x_indices = tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing='ij')
        indices = tf.stack([tf.reshape(y_indices, [-1]), tf.reshape(x_indices, [-1])], axis=1)
        
        # Apply cutout
        cutout_mask = tf.tensor_scatter_nd_update(
            cutout_mask, indices, tf.zeros([tf.shape(indices)[0]], dtype=tf.float32)
        )
        
        # Expand mask for all channels
        cutout_mask = tf.expand_dims(cutout_mask, -1)
        
        # Apply to image
        cutout_image = image * cutout_mask
        
        return cutout_image, mask

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
        mixed_masks = alpha * batch_data['mask'] + (1 - alpha) * tf.gather(batch_data['mask'], indices)
        
        # For bboxes, we keep the original ones (more complex mixing would require label handling)
        mixed_bboxes = batch_data['bbox']
        
        return {
            'image': mixed_images,
            'bbox': mixed_bboxes,
            'mask': mixed_masks,
            'label': batch_data['label']  # Keep original labels
        }

    def mosaic_augmentation(self, images: tf.Tensor, bboxes: tf.Tensor, masks: tf.Tensor, 
                           seed: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create mosaic augmentation from 4 images."""
        # This is a simplified version - full implementation would be more complex
        seeds = tf.random.experimental.stateless_split(seed, 4)
        
        # Resize images to half size
        half_size = tf.shape(images[0])[:2] // 2
        resized_images = tf.image.resize(images, half_size * 2)  # Ensure even dimensions
        resized_masks = tf.image.resize(masks, half_size * 2)
        
        # Create mosaic
        top_row = tf.concat([resized_images[0], resized_images[1]], axis=1)
        bottom_row = tf.concat([resized_images[2], resized_images[3]], axis=1)
        mosaic_image = tf.concat([top_row, bottom_row], axis=0)
        
        # Create mosaic mask
        top_mask_row = tf.concat([resized_masks[0], resized_masks[1]], axis=1)
        bottom_mask_row = tf.concat([resized_masks[2], resized_masks[3]], axis=1)
        mosaic_mask = tf.concat([top_mask_row, bottom_mask_row], axis=0)
        
        # Adjust bboxes (simplified - would need proper coordinate transformation)
        mosaic_bbox = bboxes[0] * 0.5  # Scale down to fit in quadrant
        
        return mosaic_image, mosaic_bbox, mosaic_mask

    def augment_sample(self, image: tf.Tensor, bbox: tf.Tensor, mask: tf.Tensor, 
                      seed: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Apply comprehensive augmentation pipeline."""
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            return image, bbox, mask
        
        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        
        # Ensure inputs are float32
        image = tf.cast(image, tf.float32)
        if len(tf.shape(image)) == 3 and tf.shape(image)[-1] == 3:
            image = image / 255.0  # Normalize if uint8
        
        mask = tf.cast(mask, tf.float32)
        if len(tf.shape(mask)) == 3 and tf.shape(mask)[-1] == 1:
            mask = tf.squeeze(mask, -1)
        if len(tf.shape(mask)) == 2:
            mask = tf.expand_dims(mask, -1)
        
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
        
        # Ensure final values are in valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        bbox = self._normalize_bbox(bbox)
        
        # Ensure mask is properly shaped
        if len(tf.shape(mask)) == 2:
            mask = tf.expand_dims(mask, -1)
        
        return image, bbox, mask

    def __call__(self, sample: Dict[str, tf.Tensor], seed: Optional[int] = None) -> Dict[str, tf.Tensor]:
        """Apply augmentation to a sample."""
        if not getattr(self.config, 'ENABLE_AUGMENTATION', True):
            return sample

        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)

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

    def create_augmented_dataset(self, dataset: tf.data.Dataset, 
                               augmentation_factor: int = 2) -> tf.data.Dataset:
        """Create augmented dataset with multiple versions of each sample."""
        # First replicate each sample `augmentation_factor` times
        def replicate_fn(sample):
            return tf.data.Dataset.from_tensors(sample).repeat(augmentation_factor)

        replicated_dataset = dataset.flat_map(replicate_fn)

        # Apply augmentation on each replicated sample independently
        def augment_map(sample):
            seed = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
            return self.__call__(sample, seed)

        return replicated_dataset.map(augment_map, num_parallel_calls=tf.data.AUTOTUNE)
