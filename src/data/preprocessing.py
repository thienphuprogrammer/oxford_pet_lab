import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.config.config import Config

class DataPreprocessor:
    """Enhanced data preprocessing utilities for Oxford Pet dataset with performance optimizations."""
    
    def __init__(self, config: Config = None, shuffle_buffer: int = 1000):
        self.config = config or Config()
        self.shuffle_buffer = shuffle_buffer
        
        # Precompute constants for better performance
        self.img_size_tensor = tf.constant(self.config.IMG_SIZE, dtype=tf.int32)
        self.img_size_float = tf.constant(self.config.IMG_SIZE, dtype=tf.float32)
        
        # Normalization constants (ImageNet stats for transfer learning)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
        # Advanced preprocessing options
        self.use_imagenet_normalization = getattr(config, 'USE_IMAGENET_NORM', True)
        self.preserve_aspect_ratio = getattr(config, 'PRESERVE_ASPECT_RATIO', True)
        self.pad_to_square = getattr(config, 'PAD_TO_SQUARE', False)

    @tf.function
    def normalize_image(self, image: tf.Tensor, method: str = 'standard') -> tf.Tensor:
        """Enhanced image normalization with multiple methods."""
        # Convert to float32 if needed
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        if method == 'imagenet' and self.use_imagenet_normalization:
            # ImageNet normalization for transfer learning
            image = (image - self.mean) / self.std
        elif method == 'zero_one':
            # Standard [0, 1] normalization
            image = tf.clip_by_value(image, 0.0, 1.0)
        elif method == 'minus_one_one':
            # [-1, 1] normalization
            image = 2.0 * image - 1.0
        else:
            # Default: [0, 1] with saturation
            image = tf.clip_by_value(image, 0.0, 1.0)
            
        return image

    @tf.function
    def smart_resize(self, image: tf.Tensor, size: Optional[Tuple[int, int]] = None) -> tf.Tensor:
        """Smart resize with aspect ratio preservation and padding options."""
        target_size = size or self.config.IMG_SIZE
        target_h, target_w = target_size
        
        if self.preserve_aspect_ratio:
            # Get original dimensions
            original_shape = tf.shape(image)
            orig_h = tf.cast(original_shape[0], tf.float32)
            orig_w = tf.cast(original_shape[1], tf.float32)
            
            # Calculate scale factor
            scale_h = tf.cast(target_h, tf.float32) / orig_h
            scale_w = tf.cast(target_w, tf.float32) / orig_w
            scale = tf.minimum(scale_h, scale_w)
            
            # Calculate new dimensions
            new_h = tf.cast(orig_h * scale, tf.int32)
            new_w = tf.cast(orig_w * scale, tf.int32)
            
            # Resize maintaining aspect ratio
            image = tf.image.resize(image, [new_h, new_w], antialias=True)
            
            if self.pad_to_square:
                # Pad to target size
                image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)
        else:
            # Standard resize (may distort aspect ratio)
            image = tf.image.resize(target_size, antialias=True)
            
        return image

    @tf.function 
    def process_bbox_advanced(self, bbox: tf.Tensor, original_shape: tf.Tensor, 
                            target_shape: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Advanced bbox processing with proper coordinate transformation.
        
        Args:
            bbox: BBoxFeature tensor with shape (4,) in format [ymin, xmin, ymax, xmax]
            original_shape: Original image shape [H, W, C]
            target_shape: Target image shape [H, W]
        
        Returns:
            Tuple of (normalized_bbox, absolute_bbox)
        """
        # Input bbox format from Oxford Pet: [ymin, xmin, ymax, xmax] (normalized 0-1)
        ymin, xmin, ymax, xmax = tf.unstack(bbox)
        
        # Convert to [xmin, ymin, xmax, ymax] format for consistency
        bbox_reordered = tf.stack([xmin, ymin, xmax, ymax])
        
        # Get dimensions
        orig_h = tf.cast(original_shape[0], tf.float32)
        orig_w = tf.cast(original_shape[1], tf.float32)
        target_h = tf.cast(target_shape[0], tf.float32)
        target_w = tf.cast(target_shape[1], tf.float32)
        
        # Convert normalized coordinates to absolute in original image
        bbox_abs_orig = bbox_reordered * tf.stack([orig_w, orig_h, orig_w, orig_h])
        
        # Calculate scaling factors for resize transformation
        if self.preserve_aspect_ratio:
            # Calculate scale factor (same as in smart_resize)
            scale_h = target_h / orig_h
            scale_w = target_w / orig_w
            scale = tf.minimum(scale_h, scale_w)
            
            # New dimensions after resize
            new_h = orig_h * scale
            new_w = orig_w * scale
            
            # Calculate padding offsets if pad_to_square is enabled
            if self.pad_to_square:
                pad_top = (target_h - new_h) / 2.0
                pad_left = (target_w - new_w) / 2.0
            else:
                pad_top = 0.0
                pad_left = 0.0
                target_h = new_h
                target_w = new_w
            
            # Transform bbox coordinates
            xmin_scaled = bbox_abs_orig[0] * scale + pad_left
            ymin_scaled = bbox_abs_orig[1] * scale + pad_top
            xmax_scaled = bbox_abs_orig[2] * scale + pad_left
            ymax_scaled = bbox_abs_orig[3] * scale + pad_top
        else:
            # Simple scaling without aspect ratio preservation
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            
            xmin_scaled = bbox_abs_orig[0] * scale_w
            ymin_scaled = bbox_abs_orig[1] * scale_h
            xmax_scaled = bbox_abs_orig[2] * scale_w
            ymax_scaled = bbox_abs_orig[3] * scale_h
        
        bbox_transformed = tf.stack([xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled])
        
        # Normalize to [0, 1] for target image
        bbox_normalized = bbox_transformed / tf.stack([target_w, target_h, target_w, target_h])
        bbox_normalized = tf.clip_by_value(bbox_normalized, 0.0, 1.0)
        
        return bbox_normalized, bbox_transformed

    @tf.function
    def process_segmentation_mask_advanced(self, mask: tf.Tensor, target_size: Optional[Tuple[int, int]] = None) -> tf.Tensor:
        """Enhanced segmentation mask processing for Oxford Pet dataset.
        
        Args:
            mask: Segmentation mask with shape (H, W, 1) and dtype uint8
            target_size: Target size for resizing
            
        Returns:
            Processed mask with shape (target_H, target_W)
        """
        target_size = target_size or self.config.IMG_SIZE
        
        # Remove channel dimension if present
        if len(mask.shape) == 3 and mask.shape[-1] == 1:
            mask = tf.squeeze(mask, axis=-1)
        
        # Convert to float32 for processing
        mask = tf.cast(mask, tf.float32)
        
        # Oxford Pet dataset segmentation masks:
        # - 1: Foreground (pet)
        # - 2: Background  
        # - 3: Border/boundary (between foreground and background)
        
        # Resize with nearest neighbor to preserve class boundaries
        mask_resized = tf.image.resize(
            tf.expand_dims(mask, -1), 
            target_size, 
            method='nearest'
        )
        mask_resized = tf.squeeze(mask_resized, -1)
        
        # Normalize values to expected range
        # Check if mask values are in [0, 255] range and normalize if needed
        max_val = tf.reduce_max(mask_resized)
        mask_normalized = tf.cond(
            max_val > 10.0,  # Assume values > 10 are in [0, 255] range
            lambda: mask_resized / 255.0 * 3.0,  # Scale to [0, 3]
            lambda: mask_resized
        )
        
        # Ensure values are in [1, 2, 3] range
        mask_normalized = tf.clip_by_value(mask_normalized, 1.0, 3.0)
        
        # Convert to [0, 1, 2] for easier processing
        # 0: Background, 1: Foreground, 2: Boundary
        mask_processed = mask_normalized - 1.0
        
        # Apply slight smoothing to boundary regions for better training
        boundary_mask = tf.equal(tf.cast(mask_processed, tf.int32), 2)
        
        # Smooth only boundary regions
        mask_smooth = tf.nn.avg_pool2d(
            tf.expand_dims(tf.expand_dims(mask_processed, 0), -1),
            ksize=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        mask_smooth = tf.squeeze(mask_smooth)
        
        # Use smoothed values only for boundary pixels
        final_mask = tf.where(boundary_mask, mask_smooth, mask_processed)
        
        return tf.cast(final_mask, tf.float32)

    @tf.function
    def apply_quality_enhancement(self, image: tf.Tensor) -> tf.Tensor:
        """Apply image quality enhancement techniques."""
        # Histogram equalization approximation
        image_eq = tf.image.adjust_contrast(image, contrast_factor=1.2)
        
        # Sharpening filter
        sharpen_kernel = tf.constant([
            [0, -1, 0],
            [-1, 5, -1], 
            [0, -1, 0]
        ], dtype=tf.float32)
        sharpen_kernel = tf.reshape(sharpen_kernel, [3, 3, 1, 1])
        
        # Apply sharpening to each channel
        channels = tf.unstack(image, axis=-1)
        sharpened_channels = []
        
        for channel in channels:
            channel_4d = tf.expand_dims(tf.expand_dims(channel, 0), -1)
            sharpened = tf.nn.conv2d(channel_4d, sharpen_kernel, strides=[1,1,1,1], padding='SAME')
            sharpened_channels.append(tf.squeeze(sharpened, [0, -1]))
            
        image_sharp = tf.stack(sharpened_channels, axis=-1)
        
        # Blend original and enhanced
        alpha = 0.3
        enhanced_image = alpha * image_sharp + (1 - alpha) * image_eq
        
        return tf.clip_by_value(enhanced_image, 0.0, 1.0)

    @tf.function
    def preprocess_sample_optimized(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Optimized preprocessing with better error handling and performance.
        
        Args:
            sample: Dictionary containing:
                - 'image': Image tensor with shape (H, W, 3) and dtype uint8
                - 'head_bbox': BBox tensor with shape (4,) and dtype float32
                - 'segmentation_mask': Mask tensor with shape (H, W, 1) and dtype uint8
                - 'label': ClassLabel with dtype int64 (37 classes)
                - 'species': ClassLabel with dtype int64 (2 classes: cat/dog)
                - 'file_name': Text string
        
        Returns:
            Dictionary with processed tensors
        """
        # Get original image dimensions
        original_shape = tf.shape(sample['image'])
        target_shape = tf.constant(self.config.IMG_SIZE, dtype=tf.int32)
        
        # Process image with smart resizing
        image = self.smart_resize(sample['image'])
        
        # Apply quality enhancement if enabled
        if getattr(self.config, 'ENABLE_QUALITY_ENHANCEMENT', False):
            image = self.apply_quality_enhancement(image)
            
        # Normalize image
        normalization_method = getattr(self.config, 'NORMALIZATION_METHOD', 'standard')
        image = self.normalize_image(image, method=normalization_method)
        
        # Process bounding box with advanced transformation
        bbox_normalized, bbox_absolute = self.process_bbox_advanced(
            sample['head_bbox'], original_shape, target_shape
        )
        
        # Process segmentation mask
        seg_mask = self.process_segmentation_mask_advanced(sample['segmentation_mask'])
        
        # Validate bbox coordinates
        bbox_area = (bbox_normalized[2] - bbox_normalized[0]) * (bbox_normalized[3] - bbox_normalized[1])
        valid_bbox = tf.logical_and(bbox_area > 0.001, bbox_area < 1.0)  # Reasonable area range
        
        # Process labels
        pet_class = tf.cast(sample['label'], tf.int32)  # 37 pet breed classes
        species = tf.cast(sample['species'], tf.int32)  # 2 species classes (cat/dog)
        
        # Create comprehensive output
        processed_sample = {
            'image': image,
            'pet_class': pet_class,  # 37 breed classes
            'species': species,      # 2 species classes
            'bbox': bbox_normalized,
            'segmentation_mask': seg_mask,
            'bbox_absolute': bbox_absolute,
            'valid_bbox': valid_bbox,
            'original_shape': tf.cast(original_shape, tf.float32),
            'target_shape': tf.cast(target_shape, tf.float32),
            'file_name': sample['file_name'],
        }
        
        return processed_sample

    def create_detection_target_enhanced(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Enhanced detection targets with additional metadata."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
            'valid': processed_sample['valid_bbox'],
            'area': self._compute_bbox_area(processed_sample['bbox']),
        }
        return image, targets

    def create_segmentation_target_enhanced(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Enhanced segmentation targets with class weights."""
        image = processed_sample['image']
        mask = processed_sample['segmentation_mask']
        
        # Compute class weights for balanced training
        # Classes: 0=background, 1=foreground, 2=boundary
        class_counts = tf.stack([
            tf.reduce_sum(tf.cast(tf.equal(mask, 0.0), tf.float32)),  # background
            tf.reduce_sum(tf.cast(tf.equal(mask, 1.0), tf.float32)),  # foreground
            tf.reduce_sum(tf.cast(tf.equal(mask, 2.0), tf.float32)),  # boundary
        ])
        
        total_pixels = tf.reduce_prod(tf.cast(tf.shape(mask), tf.float32))
        class_weights = total_pixels / (3.0 * class_counts + 1e-7)  # Avoid division by zero
        
        targets = {
            'mask': mask,
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
            'class_weights': class_weights,
        }
        return image, targets

    def create_multitask_target_enhanced(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Enhanced multitask targets with comprehensive metadata."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
            'segmentation_mask': processed_sample['segmentation_mask'],
            'valid_bbox': processed_sample['valid_bbox'],
            'bbox_area': self._compute_bbox_area(processed_sample['bbox']),
        }
        return image, targets

    def create_classification_target_enhanced(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Create targets for classification-only tasks."""
        image = processed_sample['image']
        targets = {
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
        }
        return image, targets

    @tf.function
    def _compute_bbox_area(self, bbox: tf.Tensor) -> tf.Tensor:
        """Compute normalized bbox area."""
        xmin, ymin, xmax, ymax = tf.unstack(bbox)
        return (xmax - xmin) * (ymax - ymin)

    def _format_for_task_enhanced(self, processed_sample: Dict[str, tf.Tensor], task: str):
        """Enhanced task formatting with better target structure."""
        if task == "detection":
            return self.create_detection_target_enhanced(processed_sample)
        elif task == "segmentation":
            return self.create_segmentation_target_enhanced(processed_sample)
        elif task == "classification":
            return self.create_classification_target_enhanced(processed_sample)
        else:  # multitask
            return self.create_multitask_target_enhanced(processed_sample)

    def prepare_dataset_optimized(
        self,
        ds: tf.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        task: str = "multitask",
        cache_filename: Optional[str] = None,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """
        Optimized dataset preparation with advanced caching and performance tuning.
        
        Args:
            ds: Input dataset with Oxford Pet structure
            batch_size: Batch size for training/validation
            shuffle: Whether to shuffle the dataset
            task: Task type - "detection", "segmentation", "classification", or "multitask"
            cache_filename: Optional filename for caching
            repeat: Whether to repeat the dataset
        """
        # Set up deterministic behavior if needed
        options = tf.data.Options()
        options.experimental_deterministic = getattr(self.config, 'DETERMINISTIC', False)
        ds = ds.with_options(options)
        
        # Early caching of raw data if specified
        if cache_filename:
            ds = ds.cache(cache_filename)
        
        # Shuffle before preprocessing for better randomization
        if shuffle:
            ds = ds.shuffle(
                self.shuffle_buffer, 
                seed=getattr(self.config, 'RANDOM_SEED', None),
                reshuffle_each_iteration=True
            )
        
        # Repeat dataset if needed (for training)
        if repeat:
            ds = ds.repeat()
        
        # Preprocessing with optimized parallelization
        ds = ds.map(
            lambda s: self._format_for_task_enhanced(
                self.preprocess_sample_optimized(s), task
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        
        # Filter out invalid samples (mainly for detection tasks)
        def is_valid_sample(image, targets):
            if task == "detection" or task == "multitask":
                return targets.get('valid_bbox', True)
            return True  # Segmentation and classification samples are always valid
        
        ds = ds.filter(is_valid_sample)
        
        # Batching with padding for variable-size elements
        padded_shapes = self._get_padded_shapes(task)
        if padded_shapes:
            ds = ds.padded_batch(
                batch_size,
                padded_shapes=padded_shapes,
                drop_remainder=True
            )
        else:
            ds = ds.batch(batch_size, drop_remainder=True)
        
        # Post-processing cache (after batching)
        if not cache_filename:  # Only cache in memory if not caching to file
            ds = ds.cache()
        
        # Prefetch for pipeline optimization
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds

    def _get_padded_shapes(self, task: str) -> Optional[Tuple]:
        """Get padded shapes for variable-size batching if needed."""
        # For Oxford Pet dataset, shapes are usually fixed after preprocessing
        return None

    def create_validation_dataset(
        self,
        ds: tf.data.Dataset,
        batch_size: int,
        task: str = "multitask",
    ) -> tf.data.Dataset:
        """Create optimized validation dataset without augmentation."""
        return self.prepare_dataset_optimized(
            ds=ds,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for validation
            task=task,
            repeat=False,   # No repetition for validation
        )

    def create_training_dataset(
        self,
        ds: tf.data.Dataset,
        batch_size: int,
        task: str = "multitask",
        cache_filename: Optional[str] = None,
    ) -> tf.data.Dataset:
        """Create optimized training dataset with full preprocessing."""
        return self.prepare_dataset_optimized(
            ds=ds,
            batch_size=batch_size,
            shuffle=True,
            task=task,
            cache_filename=cache_filename,
            repeat=True,    # Repeat for training
        )

    def get_dataset_statistics(self, ds: tf.data.Dataset, num_samples: int = 1000) -> Dict[str, Any]:
        """Compute dataset statistics for monitoring and debugging."""
        stats = {
            'mean_pixel_values': [],
            'std_pixel_values': [],
            'bbox_areas': [],
            'pet_class_distribution': {},
            'species_distribution': {},
            'segmentation_class_distribution': {'background': 0, 'foreground': 0, 'boundary': 0},
        }
        
        sample_count = 0
        for batch in ds.take(num_samples // 32):  # Assuming batch_size=32
            if isinstance(batch, tuple):
                images, targets = batch
                
                # Image statistics
                batch_mean = tf.reduce_mean(images, axis=[0, 1, 2])
                batch_std = tf.math.reduce_std(images, axis=[0, 1, 2])
                stats['mean_pixel_values'].append(batch_mean.numpy())
                stats['std_pixel_values'].append(batch_std.numpy())
                
                # Bbox statistics if available
                if 'bbox' in targets:
                    areas = self._compute_bbox_area(targets['bbox'])
                    stats['bbox_areas'].extend(areas.numpy().tolist())
                
                # Pet class distribution
                if 'pet_class' in targets:
                    labels = targets['pet_class'].numpy()
                    for label in labels:
                        stats['pet_class_distribution'][int(label)] = stats['pet_class_distribution'].get(int(label), 0) + 1
                
                # Species distribution
                if 'species' in targets:
                    species = targets['species'].numpy()
                    for s in species:
                        stats['species_distribution'][int(s)] = stats['species_distribution'].get(int(s), 0) + 1
                
                # Segmentation mask statistics
                if 'mask' in targets:
                    masks = targets['mask'].numpy()
                    for mask in masks:
                        stats['segmentation_class_distribution']['background'] += np.sum(mask == 0)
                        stats['segmentation_class_distribution']['foreground'] += np.sum(mask == 1)
                        stats['segmentation_class_distribution']['boundary'] += np.sum(mask == 2)
                        
                sample_count += len(images)
                if sample_count >= num_samples:
                    break
        
        # Aggregate statistics
        if stats['mean_pixel_values']:
            stats['mean_pixel_values'] = np.mean(stats['mean_pixel_values'], axis=0)
            stats['std_pixel_values'] = np.mean(stats['std_pixel_values'], axis=0)
        
        return stats

    def get_class_names(self) -> Dict[str, List[str]]:
        """Return class names for the Oxford Pet dataset."""
        # Oxford Pet dataset has 37 pet breeds
        pet_breeds = [
            'Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
            'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair',
            'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
            'German_Shorthaired', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
            'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland',
            'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard',
            'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx',
            'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier'
        ]
        
        species_names = ['Cat', 'Dog']
        
        return {
            'pet_breeds': pet_breeds,
            'species': species_names,
            'segmentation_classes': ['Background', 'Foreground', 'Boundary']
        }