import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.config.config import Config

_SHARP_KERNEL = tf.constant(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]], dtype=tf.float32
)
_SHARP_KERNEL = tf.reshape(
    tf.tile(_SHARP_KERNEL[..., tf.newaxis], [1, 1, 3]),  # (3,3,3)
    [3, 3, 3, 1]                                         # (kh, kw, in_channels, channel_multiplier)
)

class DataPreprocessor:
    """Enhanced data preprocessing utilities for Oxford Pet dataset with guaranteed output format."""
    
    def __init__(self, config: Config = None, shuffle_buffer: int = 1000):
        self.config = config or Config()
        self.shuffle_buffer = shuffle_buffer

        self._target_h, self._target_w = self.config.IMG_SIZE
        
        # Precompute constants for better performance
        self.img_size_tensor = tf.constant(self.config.IMG_SIZE, dtype=tf.int32)
        self.img_size_float = tf.constant(self.config.IMG_SIZE, dtype=tf.float32)
        
        # Normalization constants (ImageNet stats for transfer learning)
        self.mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
        self.std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
        
        # Advanced preprocessing options
        self.use_imagenet_normalization = getattr(config, 'USE_IMAGENET_NORM', True)
        self.preserve_aspect_ratio = getattr(config, 'PRESERVE_ASPECT_RATIO', True)
        self.pad_to_square = getattr(config, 'PAD_TO_SQUARE', True)
        self.enable_quality_enhancement = getattr(config, 'ENABLE_QUALITY_ENHANCEMENT', False)
        self.normalize_method = getattr(config, 'NORMALIZE_METHOD', 'standard')

    @tf.function
    def _ensure_input_format(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Ensure input sample has the correct format and types."""
        formatted_sample = {}
        
        # Image: ensure uint8 and 3D shape (H, W, 3)
        image = sample['image']
        if image.dtype != tf.uint8:
            if image.dtype == tf.float32 or image.dtype == tf.float64:
                # Convert from float [0,1] to uint8 [0,255]
                image = tf.cast(tf.clip_by_value(image * 255.0, 0, 255), tf.uint8)
            else:
                image = tf.cast(image, tf.uint8)
        
        # Ensure 3D shape (H, W, 3)
        if len(image.shape) == 4:  # Remove batch dimension if present
            image = tf.squeeze(image, 0)
        if len(image.shape) == 2:  # Add channel dimension for grayscale
            image = tf.stack([image, image, image], axis=-1)
        elif image.shape[-1] == 1:  # Convert single channel to 3 channels
            image = tf.tile(image, [1, 1, 3])
        elif image.shape[-1] == 4:  # Remove alpha channel
            image = image[..., :3]
        
        formatted_sample['image'] = image
        
        # Head bbox: ensure float32 and shape (4,)
        bbox = sample['head_bbox']
        bbox = tf.cast(bbox, tf.float32)
        bbox = tf.reshape(bbox, [4])  # Ensure exactly 4 elements
        formatted_sample['head_bbox'] = bbox
        
        # Segmentation mask: ensure uint8 and 3D shape (H, W, 1)
        mask = sample['segmentation_mask']
        if mask.dtype != tf.uint8:
            if mask.dtype == tf.float32 or mask.dtype == tf.float64:
                # Convert from float to uint8
                mask = tf.cast(tf.clip_by_value(mask, 0, 255), tf.uint8)
            else:
                mask = tf.cast(mask, tf.uint8)
        
        # Ensure 3D shape (H, W, 1)
        if len(mask.shape) == 4:  # Remove batch dimension if present
            mask = tf.squeeze(mask, 0)
        if len(mask.shape) == 2:  # Add channel dimension
            mask = tf.expand_dims(mask, -1)
        elif len(mask.shape) == 3 and mask.shape[-1] != 1:
            mask = tf.expand_dims(mask[..., 0], -1)  # Take first channel
        
        formatted_sample['segmentation_mask'] = mask
        
        # Label: ensure int64 and scalar
        label = sample['label']
        label = tf.cast(label, tf.int64)
        if len(label.shape) > 0:
            label = tf.squeeze(label)
        formatted_sample['label'] = label
        
        # Species: ensure int64 and scalar
        species = sample['species']
        species = tf.cast(species, tf.int64)
        if len(species.shape) > 0:
            species = tf.squeeze(species)
        formatted_sample['species'] = species
        
        # File name: ensure string
        file_name = sample['file_name']
        if isinstance(file_name, bytes):
            file_name = tf.py_function(lambda x: x.decode('utf-8'), [file_name], tf.string)
        elif not isinstance(file_name, tf.Tensor) or file_name.dtype != tf.string:
            file_name = tf.cast(file_name, tf.string)
        formatted_sample['file_name'] = file_name
        
        return formatted_sample

    @tf.function
    def resize_pad(self, image, target_size):
        th, tw = target_size
        return tf.image.resize_with_pad(image, th, tw, antialias=True)

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
        target_h, target_w = size or (self._target_h, self._target_w)
        
        if self.preserve_aspect_ratio:
            image = tf.image.resize_with_pad(image, target_h, target_w, antialias=True)
            
            if self.pad_to_square:
                image = tf.image.resize_with_crop_or_pad(image, target_h, target_w)
        else:
            image = tf.image.resize(image, [target_h, target_w], antialias=True)
            
        return image
    
    @tf.function
    def _transform_bbox(self,
                        bbox: tf.Tensor,
                        orig_shape: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Transform bbox from original to target image."""
        ymin, xmin, ymax, xmax = tf.unstack(bbox)

        orig_h = tf.cast(orig_shape[0], tf.float32)
        orig_w = tf.cast(orig_shape[1], tf.float32)
        tgt_h  = tf.cast(self._target_h, tf.float32)
        tgt_w  = tf.cast(self._target_w, tf.float32)

        # scale + pad (same as smart_resize)
        scale = tf.minimum(tgt_h / orig_h, tgt_w / orig_w)
        new_h = orig_h * scale
        new_w = orig_w * scale
        pad_top  = (tgt_h - new_h) / 2.0
        pad_left = (tgt_w - new_w) / 2.0

        # absolute coordinates (original image)
        x1 = xmin * orig_w
        y1 = ymin * orig_h
        x2 = xmax * orig_w
        y2 = ymax * orig_h

        # scale + pad (same as smart_resize)
        x1_s = x1 * scale + pad_left
        y1_s = y1 * scale + pad_top
        x2_s = x2 * scale + pad_left
        y2_s = y2 * scale + pad_top

        bbox_abs = tf.stack([x1_s, y1_s, x2_s, y2_s])
        bbox_nrm = bbox_abs / tf.stack([tgt_w, tgt_h, tgt_w, tgt_h])
        bbox_nrm = tf.clip_by_value(bbox_nrm, 0.0, 1.0)
        return bbox_nrm, bbox_abs

    @tf.function
    def process_segmentation_mask(self,
                                  mask: tf.Tensor,
                                  target_size: Optional[Tuple[int, int]] = None
                                  ) -> tf.Tensor:
        """Resize mask (nearest), normalize to 0/1/2."""
        th, tw = target_size or (self._target_h, self._target_w)

        # ensure 2-D
        if mask.shape.rank == 2:
            mask = mask[..., tf.newaxis]
        if mask.shape.rank == 3 and mask.shape[-1] == 1:
            mask = tf.squeeze(mask, -1)

        mask = tf.image.resize_with_pad(mask[..., tf.newaxis],
                                        th, tw,
                                        method='nearest')
        mask = tf.squeeze(mask, -1)

        # Ensure mask is float32 for subsequent comparison and scaling
        mask = tf.cast(mask, tf.float32)

        # scale if original data is 8-bit
        max_val = tf.reduce_max(mask)
        mask = tf.cond(max_val > 3.0,
                       lambda: mask / 255.0 * 3.0,
                       lambda: mask)
        mask = tf.clip_by_value(mask, 0.0, 3.0)  # 0-3
        mask = mask - 1.0                         # 0:BG,1:FG,2:Border
        return mask

    @tf.function
    def apply_quality_enhancement(self, image: tf.Tensor) -> tf.Tensor:
        """Apply image quality enhancement techniques."""
        # Histogram equalization approximation
        image_eq = tf.image.adjust_contrast(image, contrast_factor=1.2)
        
        # depth-wise conv
        img4d = image_eq[tf.newaxis, ...]
        sharp  = tf.nn.depthwise_conv2d(
            img4d, 
            _SHARP_KERNEL,
            strides=[1, 1, 1, 1],
            padding='SAME',
        )

        sharp = tf.squeeze(sharp, 0)
        return tf.clip_by_value(sharp * 0.3 + image_eq * 0.7, 0.0, 1.0)

    @tf.function
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Optimized preprocessing with guaranteed output format."""
        # First, ensure input format is correct
        formatted_sample = self._ensure_input_format(sample)
        
        # Get original image dimensions
        original_shape = tf.shape(formatted_sample['image'])
        
        # Process image with smart resizing
        image = self.smart_resize(formatted_sample['image'])
        
        # Apply quality enhancement if enabled
        if self.enable_quality_enhancement:
            image = self.apply_quality_enhancement(image)
            
        # Normalize image
        image = self.normalize_image(image, self.normalize_method)
        
        # Process bounding box with advanced transformation
        bbox_normalized, bbox_absolute = self._transform_bbox(
            formatted_sample['head_bbox'], original_shape
        )
        
        # Process segmentation mask
        seg_mask = self.process_segmentation_mask(formatted_sample['segmentation_mask'])
        
        # Validate bbox coordinates
        bbox_area = (bbox_normalized[2] - bbox_normalized[0]) * (bbox_normalized[3] - bbox_normalized[1])
        valid_bbox = tf.logical_and(bbox_area > 0.001, bbox_area < 1.0)  # Reasonable area range
        
        # Process labels
        pet_class = tf.cast(formatted_sample['label'], tf.int32)  # 37 pet breed classes
        species = tf.cast(formatted_sample['species'], tf.int32)  # 2 species classes (cat/dog)
        
        # Create comprehensive output
        processed_sample = {
            'image': image,
            'bbox': bbox_normalized,
            'bbox_absolute': bbox_absolute,
            'valid_bbox': valid_bbox,
            'segmentation_mask': seg_mask,
            'pet_class': pet_class,  # 37 breed classes
            'species': species,      # 2 species classes
            'original_shape': tf.cast(original_shape, tf.float32),
            'target_shape': self.img_size_float,
            'file_name': formatted_sample['file_name'],
        }
        
        return processed_sample

    def _compute_bbox_area(self, bbox: tf.Tensor) -> tf.Tensor:
        xmin, ymin, xmax, ymax = tf.unstack(bbox, axis=-1)
        return (xmax - xmin) * (ymax - ymin)

    def create_detection_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Detection targets with additional metadata."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
            'valid_bbox': processed_sample['valid_bbox'],  # Include valid_bbox
            'area': self._compute_bbox_area(processed_sample['bbox']),
        }
        return image, targets

    def create_segmentation_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Segmentation targets with additional metadata."""
        image = processed_sample['image']
        mask = processed_sample['segmentation_mask']
        
        # Compute class weights for balanced training
        # Classes: 0=background, 1=foreground, 2=boundary
        mask2d = tf.squeeze(mask, -1)
        class_counts = tf.stack([
            tf.reduce_sum(tf.cast(tf.equal(mask2d, 0.0), tf.float32)),  # background
            tf.reduce_sum(tf.cast(tf.equal(mask2d, 1.0), tf.float32)),  # foreground
            tf.reduce_sum(tf.cast(tf.equal(mask2d, 2.0), tf.float32)),  # boundary
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

    def create_multitask_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Multitask targets with comprehensive metadata."""
        image = processed_sample['image']
        targets = {
            'bbox': processed_sample['bbox'],
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
            'segmentation_mask': processed_sample['segmentation_mask'],
            'valid_bbox': processed_sample['valid_bbox'],  # Include valid_bbox
            'bbox_area': self._compute_bbox_area(processed_sample['bbox']),
        }
        return image, targets

    def create_classification_target(self, processed_sample: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Create targets for classification-only tasks."""
        image = processed_sample['image']
        targets = {
            'pet_class': processed_sample['pet_class'],
            'species': processed_sample['species'],
        }
        return image, targets

    @tf.function
    def create_passthrough_target(self, sample: Dict[str, Any]) -> Dict[str, tf.Tensor]:
        """Create passthrough output that maintains original format exactly."""
        # Ensure input format is correct
        formatted_sample = self._ensure_input_format(sample)
        
        # Return exactly the required format without any processing
        return {
            'file_name': formatted_sample['file_name'],
            'head_bbox': formatted_sample['head_bbox'],
            'image': formatted_sample['image'],
            'label': formatted_sample['label'],
            'segmentation_mask': formatted_sample['segmentation_mask'],
            'species': formatted_sample['species']
        }

    def _format_for_task(self, processed_sample: Dict[str, tf.Tensor], task: str):
        """Task formatting with better target structure."""
        if task == "passthrough":
            # For passthrough, we need to reconstruct the original format
            return self.create_passthrough_target({
                'file_name': processed_sample['file_name'],
                'head_bbox': processed_sample['bbox'],  # Use normalized bbox
                'image': processed_sample['image'],
                'label': processed_sample['pet_class'],
                'segmentation_mask': processed_sample['segmentation_mask'],
                'species': processed_sample['species']
            })
        elif task == "detection":
            return self.create_detection_target(processed_sample)
        elif task == "segmentation":
            return self.create_segmentation_target(processed_sample)
        elif task == "classification":
            return self.create_classification_target(processed_sample)
        else:  # multitask
            return self.create_multitask_target(processed_sample)

    def prepare_dataset(
        self,
        ds: tf.data.Dataset,
        batch_size: int = None,
        shuffle: bool = False,
        task: str = "passthrough",
        cache_filename: Optional[str] = None,
        repeat: bool = False,
    ) -> tf.data.Dataset:
        """
        Optimized dataset preparation with guaranteed output format.
        
        Args:
            ds: Input dataset with Oxford Pet structure
            batch_size: Batch size for training/validation (None for no batching)
            shuffle: Whether to shuffle the dataset
            task: Task type - "passthrough", "detection", "segmentation", "classification", or "multitask"
            cache_filename: Optional filename for caching
            repeat: Whether to repeat the dataset
        """
        # Set up deterministic behavior if needed
        options = tf.data.Options()
        options.experimental_deterministic = getattr(self.config, 'DETERMINISTIC', False)
        ds = ds.with_options(options)
        
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
        
        def _proc(sample):
            """Process a single sample from the dataset."""
            # Ensure sample is a dictionary with expected keys
            if not isinstance(sample, dict):
                raise TypeError(f"Expected sample to be a dict, got {type(sample)}")
            
            required_keys = ['image', 'head_bbox', 'segmentation_mask', 'label', 'species', 'file_name']
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                raise KeyError(f"Missing required keys in sample: {missing_keys}")
            
            if task == "passthrough":
                # For passthrough, directly format the input without heavy processing
                return self.create_passthrough_target(sample)
            else:
                processed = self.preprocess_sample(sample)
                return self._format_for_task(processed, task)
        
        # Preprocessing with optimized parallelization
        ds = ds.map(
            _proc,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        
        # Only filter for valid_bbox if the task includes bbox information (and not passthrough)
        if task in ('detection', 'multitask'):
            ds = ds.filter(lambda img, targets: targets['valid_bbox'])
        
        # Post-processing cache (after filtering but before batching)
        if cache_filename:
            ds = ds.cache(cache_filename)
        else:
            ds = ds.cache()
        
        # Batch only if batch_size is specified
        if batch_size is not None:
            ds = ds.batch(batch_size, drop_remainder=True)
        
        # Prefetch for pipeline optimization
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_validation_dataset(
        self,
        ds: tf.data.Dataset,
        batch_size: int = None,
        task: str = "passthrough",
    ) -> tf.data.Dataset:
        """Create optimized validation dataset without augmentation."""
        return self.prepare_dataset(
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
        task: str = "detection",
        cache_filename: Optional[str] = None,
    ) -> tf.data.Dataset:
        """Create optimized training dataset with full preprocessing."""
        return self.prepare_dataset(
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
            'segmentation_class_distribution': {
                'background': 0, 
                'foreground': 0, 
                'boundary': 0,
            },
        }
        
        sample_count = 0
        for batch in ds.take(num_samples // 32):  # Assuming batch_size=32              
            if isinstance(batch, dict):
                # For passthrough format
                images = batch['image']
                pet_classes = batch['label']
                species = batch['species']
                bboxes = batch['head_bbox']
                masks = batch['segmentation_mask']
            else:
                # For other formats (images, targets)
                images, targets = batch
                pet_classes = targets.get('pet_class', targets.get('label'))
                species = targets.get('species')
                bboxes = targets.get('bbox', targets.get('head_bbox'))
                masks = targets.get('mask', targets.get('segmentation_mask'))
            
            # Handle both batched and unbatched data
            if len(images.shape) == 3:  # Single image
                images = tf.expand_dims(images, 0)
                if pet_classes is not None:
                    pet_classes = tf.expand_dims(pet_classes, 0)
                if species is not None:
                    species = tf.expand_dims(species, 0)
                if bboxes is not None:
                    bboxes = tf.expand_dims(bboxes, 0)
                if masks is not None:
                    masks = tf.expand_dims(masks, 0)
            
            # Image statistics (convert to float for statistics)
            images_float = tf.cast(images, tf.float32) / 255.0
            batch_mean = tf.reduce_mean(images_float, axis=[0, 1, 2])
            batch_std = tf.math.reduce_std(images_float, axis=[0, 1, 2])
            stats['mean_pixel_values'].append(batch_mean.numpy())
            stats['std_pixel_values'].append(batch_std.numpy())
                
            # Bbox statistics if available
            if bboxes is not None:
                areas = self._compute_bbox_area(bboxes)
                stats['bbox_areas'].extend(areas.numpy().tolist())
                
            # Pet class distribution
            if pet_classes is not None:
                for lbl in pet_classes.numpy():
                    stats['pet_class_distribution'][int(lbl)] = \
                        stats['pet_class_distribution'].get(int(lbl), 0) + 1
                    
            # Species distribution
            if species is not None:
                for s in species.numpy():
                    stats['species_distribution'][int(s)] = \
                        stats['species_distribution'].get(int(s), 0) + 1
                
            # Segmentation mask statistics
            if masks is not None:
                for mask in masks.numpy():
                    unique_vals, counts = np.unique(mask, return_counts=True)
                    for val, count in zip(unique_vals, counts):
                        if val == 0:
                            stats['segmentation_class_distribution']['background'] += count
                        elif val == 1:
                            stats['segmentation_class_distribution']['foreground'] += count
                        elif val == 2:
                            stats['segmentation_class_distribution']['boundary'] += count
                        
            sample_count += images.shape[0]
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

    def verify_output_format(self, ds: tf.data.Dataset) -> bool:
        """Verify that the dataset output matches the expected TensorSpec format."""
        expected_spec = {
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string),
            'head_bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'image': tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64),
            'segmentation_mask': tf.TensorSpec(shape=(None, None, 1), dtype=tf.uint8),
            'species': tf.TensorSpec(shape=(), dtype=tf.int64)
        }
        
        try:
            sample = next(iter(ds))
            
            for key, expected_tensor_spec in expected_spec.items():
                if key not in sample:
                    print(f"Missing key: {key}")
                    return False
                
                actual_tensor = sample[key]
                
                # Check dtype
                if actual_tensor.dtype != expected_tensor_spec.dtype:
                    print(f"Dtype mismatch for {key}: expected {expected_tensor_spec.dtype}, got {actual_tensor.dtype}")
                    return False
                
                # Check shape compatibility
                expected_shape = expected_tensor_spec.shape
                actual_shape = actual_tensor.shape
                
                if len(expected_shape) != len(actual_shape):
                    print(f"Shape rank mismatch for {key}: expected {len(expected_shape)}, got {len(actual_shape)}")
                    return False
                
                for i, (exp_dim, act_dim) in enumerate(zip(expected_shape, actual_shape)):
                    if exp_dim is not None and exp_dim != act_dim:
                        print(f"Shape dimension mismatch for {key} at index {i}: expected {exp_dim}, got {act_dim}")
                        return False
            
            print("✅ Output format verification passed!")
            print("Actual output spec:")
            for key, tensor in sample.items():
                print(f"  '{key}': TensorSpec(shape={tuple(tensor.shape)}, dtype={tensor.dtype})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False

    