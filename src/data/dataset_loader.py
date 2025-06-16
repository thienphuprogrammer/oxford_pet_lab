import os
import logging
from typing import Tuple, Dict, Any, Optional, List, Union
from pathlib import Path
import warnings

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class OxfordPetDatasetLoader:
    """
    Dataset loader for Oxford-IIIT Pet Dataset using TensorFlow Datasets.
    
    Features:
    - Automatic dataset download if not available
    - Flexible data splitting with stratification
    - Support for classification, detection, segmentation tasks
    - Basic data format conversion without preprocessing
    - Comprehensive dataset statistics
    """
    
    def __init__(
        self, 
        data_dir: Optional[str] = None,
        download: bool = True,
        log_level: str = 'INFO'
        ):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory to store the dataset. If None, uses default TFDS directory
            download: Whether to download the dataset if not found
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        # Setup logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.dataset_name = f'oxford_iiit_pet:4.*.*'  # Use latest version
        self.data_dir = data_dir
        self.download = download
        
        # Dataset info (lazy loaded)
        self._dataset_info = None
        self._class_names = None
        self._num_classes = None
        
        # Dataset splits
        self._raw_datasets = {}
        
        self.logger.info(f"Initialized Oxford Pet Dataset Loader")
        self.logger.info(f"Data directory: {self.data_dir or 'default TFDS directory'}")

    @property
    def dataset_info(self):
        """Lazy load dataset info"""
        if self._dataset_info is None:
            self._load_dataset_info()
        return self._dataset_info
    
    @property
    def class_names(self) -> List[str]:
        """Get class names"""
        if self._class_names is None:
            self._class_names = self.dataset_info.features['label'].names
        return self._class_names
    
    @property
    def num_classes(self) -> int:
        """Get number of classes"""
        if self._num_classes is None:
            self._num_classes = len(self.class_names)
        return self._num_classes
    
    def _load_dataset_info(self):
        """Load dataset information"""
        try:
            self.logger.info("Loading dataset information...")
            builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
            self._dataset_info = builder.info
            
            self.logger.info(f"Dataset: {self._dataset_info.name}")
            self.logger.info(f"Version: {self._dataset_info.version}")
            self.logger.info(f"Description: {self._dataset_info.description[:100]}...")
            self.logger.info(f"Number of classes: {len(self._dataset_info.features['label'].names)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset info: {e}")
            raise
    
    def download_and_prepare(self, force_download: bool = False):
        """
        Download and prepare the dataset if needed.
        
        Args:
            force_download: Force re-download even if dataset exists
        """
        try:
            self.logger.info("Checking dataset availability...")
            builder = tfds.builder(self.dataset_name, data_dir=self.data_dir)
            
            if not builder.data_path.exists() or force_download:
                self.logger.info("Dataset not found or force download requested. Downloading...")
                builder.download_and_prepare()
                self.logger.info("Dataset downloaded and prepared successfully!")
            else:
                self.logger.info("Dataset already available.")
                
        except Exception as e:
            self.logger.error(f"Failed to download/prepare dataset: {e}")
            if not self.download:
                self.logger.error("Set download=True to automatically download the dataset")
            raise
    
    def load_raw_dataset(self, 
                        splits: Union[str, List[str]] = None,
                        force_reload: bool = False) -> Dict[str, tf.data.Dataset]:
        """
        Load raw dataset splits.
        
        Args:
            splits: Which splits to load. If None, loads all available splits
            force_reload: Force reload even if already cached
            
        Returns:
            Dictionary mapping split names to tf.data.Dataset objects
        """
        if not force_reload and self._raw_datasets:
            return self._raw_datasets
        
        # Ensure dataset is available
        self.download_and_prepare()
        
        if splits is None:
            splits = ['train', 'test']
        elif isinstance(splits, str):
            splits = [splits]
        
        try:
            self.logger.info(f"Loading raw dataset splits: {splits}")
            
            datasets, info = tfds.load(
                self.dataset_name,
                split=splits,
                data_dir=self.data_dir,
                with_info=True,
                as_supervised=False,
                shuffle_files=True
            )
            
            self._dataset_info = info
            
            # Handle single split case
            if len(splits) == 1:
                datasets = {splits[0]: datasets}
            else:
                datasets = dict(zip(splits, datasets))
            
            self._raw_datasets.update(datasets)
            
            # Log dataset sizes
            for split_name, dataset in datasets.items():
                size = info.splits[split_name].num_examples
                self.logger.info(f"{split_name.capitalize()} set: {size:,} examples")
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_train_val_test_splits(
        self,
        val_split: float = 0.2,
        test_split: float = None,
        seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train/validation/test splits.
        
        Args:
            val_split: Fraction of training data to use for validation
            test_split: If None, uses existing test split. If float, creates new test split from train
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        self.logger.info("Creating train/validation/test splits...")
        
        # Load raw datasets
        raw_datasets = self.load_raw_dataset(['train', 'test'])
        
        if test_split is None:
            # Use existing train/test split, create validation from train
            train_ds = raw_datasets['train']
            test_ds = raw_datasets['test']
            
            # Calculate validation size
            train_size = self.dataset_info.splits['train'].num_examples
            val_size = int(train_size * val_split)
            final_train_size = train_size - val_size
            
            # Create validation split
            train_ds = train_ds.shuffle(1000, seed=seed)
            val_ds = train_ds.take(val_size)
            train_ds = train_ds.skip(val_size)
            
            self.logger.info(f"Split sizes - Train: {final_train_size:,}, Val: {val_size:,}, Test: {self.dataset_info.splits['test'].num_examples:,}")
            
        else:
            # Create custom splits from all data
            all_data = raw_datasets['train'].concatenate(raw_datasets['test'])
            
            # Calculate split sizes
            total_size = (self.dataset_info.splits['train'].num_examples + 
                         self.dataset_info.splits['test'].num_examples)
            test_size = int(total_size * test_split)
            val_size = int((total_size - test_size) * val_split)
            train_size = total_size - val_size - test_size
            
            # Create splits
            all_data = all_src.data.shuffle(10000, seed=seed)
            test_ds = all_src.data.take(test_size)
            remaining = all_src.data.skip(test_size)
            val_ds = remaining.take(val_size)
            train_ds = remaining.skip(val_size)
            
            self.logger.info(f"Custom split sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        
        return train_ds, val_ds, test_ds
    
    def preprocess_for_classification(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Basic preprocessing for classification task.
        Returns (image, label) pairs without any augmentation or normalization.
        
        Args:
            dataset: Input tf.data.Dataset
            
        Returns:
            tf.data.Dataset with (image, label) pairs
        """
        def extract_classification_data(example):
            return example['image'], example['label']
        
        return dataset.map(extract_classification_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    def preprocess_for_segmentation(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Basic preprocessing for segmentation task.
        Returns (image, segmentation_mask) pairs without any augmentation or normalization.
        
        Args:
            dataset: Input tf.data.Dataset
            
        Returns:
            tf.data.Dataset with (image, segmentation_mask) pairs
        """
        def extract_segmentation_data(example):
            return example['image'], example['segmentation_mask']
        
        return dataset.map(extract_segmentation_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    def preprocess_for_detection(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Basic preprocessing for object detection task.
        Returns (image, bbox_info) pairs where bbox_info contains bounding box coordinates and labels.
        
        Note: Oxford Pet dataset doesn't have explicit bounding box annotations in TFDS.
        This function prepares the data structure for detection tasks where bounding boxes
        might be derived from segmentation masks or added separately.
        
        Args:
            dataset: Input tf.data.Dataset
            
        Returns:
            tf.data.Dataset with (image, detection_info) pairs
        """
        def extract_detection_data(example):
            return example['image'], example['head_bbox'], example['species'], example['label']
        
        return dataset.map(extract_detection_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    def get_dataset_statistics(self, 
                             dataset: tf.data.Dataset = None,
                             max_samples: int = 1000) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics.
        
        Args:
            dataset: Dataset to analyze. If None, uses raw train dataset
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        if dataset is None:
            raw_datasets = self.load_raw_dataset(['train'])
            dataset = raw_datasets['train']
        
        self.logger.info(f"Computing dataset statistics (max {max_samples} samples)...")
        
        # Take subset for analysis
        dataset_subset = dataset.take(max_samples)
        
        # Initialize statistics
        stats = {
            'basic_info': {
                'dataset_name': self.dataset_name,
                'num_classes': self.num_classes,
                'class_names': self.class_names
            },
            'split_info': dict(self.dataset_info.splits) if self.dataset_info else {},
            'image_stats': {
                'heights': [],
                'widths': [],
                'channels': []
            },
            'class_distribution': {}
        }
        
        # Analyze samples
        sample_count = 0
        for example in dataset_subset:
            image = example['image']
            label = example['label'].numpy()
            
            # Image statistics
            height, width, channels = image.shape
            stats['image_stats']['heights'].append(height)
            stats['image_stats']['widths'].append(width)
            stats['image_stats']['channels'].append(channels)
            
            # Class distribution
            class_name = self.class_names[label]
            stats['class_distribution'][class_name] = stats['class_distribution'].get(class_name, 0) + 1
            
            sample_count += 1
        
        # Compute summary statistics
        if stats['image_stats']['heights']:
            stats['image_summary'] = {
                'height': {
                    'mean': np.mean(stats['image_stats']['heights']),
                    'std': np.std(stats['image_stats']['heights']),
                    'min': np.min(stats['image_stats']['heights']),
                    'max': np.max(stats['image_stats']['heights'])
                },
                'width': {
                    'mean': np.mean(stats['image_stats']['widths']),
                    'std': np.std(stats['image_stats']['widths']),
                    'min': np.min(stats['image_stats']['widths']),
                    'max': np.max(stats['image_stats']['widths'])
                },
                'channels': {
                    'mode': max(set(stats['image_stats']['channels']), 
                              key=stats['image_stats']['channels'].count)
                }
            }
        
        # Remove raw lists to save memory
        del stats['image_stats']
        
        stats['analysis_info'] = {
            'samples_analyzed': sample_count,
            'max_samples_requested': max_samples
        }
        
        self.logger.info(f"Analyzed {sample_count} samples")
        return stats
    
    def visualize_samples(self, 
                         dataset: tf.data.Dataset = None,
                         num_samples: int = 9,
                         figsize: Tuple[int, int] = (12, 12),
                         show_segmentation: bool = False):
        """
        Visualize sample images from the dataset.
        
        Args:
            dataset: Dataset to visualize. If None, uses raw train dataset
            num_samples: Number of samples to show
            figsize: Figure size for matplotlib
            show_segmentation: Whether to show segmentation masks
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib is required for visualization")
            return
        
        if dataset is None:
            raw_datasets = self.load_raw_dataset(['train'])
            dataset = raw_datasets['train']
        
        # Take samples
        samples = list(dataset.take(num_samples))
        
        # Create subplot grid
        cols = int(np.ceil(np.sqrt(num_samples)))
        rows = int(np.ceil(num_samples / cols))
        
        if show_segmentation:
            cols *= 2  # Show image and mask side by side
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, example in enumerate(samples):
            row = i // (cols // (2 if show_segmentation else 1))
            col = (i % (cols // (2 if show_segmentation else 1))) * (2 if show_segmentation else 1)
            
            # Show image
            image = example['image'].numpy()
            label = example['label'].numpy()
            class_name = self.class_names[label]
            
            axes[row][col].imshow(image)
            axes[row][col].set_title(f'{class_name}')
            axes[row][col].axis('off')
            
            # Show segmentation mask if requested
            if show_segmentation and 'segmentation_mask' in example:
                mask = example['segmentation_mask'].numpy()
                axes[row][col + 1].imshow(mask, cmap='tab20')
                axes[row][col + 1].set_title(f'{class_name} - Mask')
                axes[row][col + 1].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * (cols // (2 if show_segmentation else 1))):
            row = i // (cols // (2 if show_segmentation else 1))
            col = (i % (cols // (2 if show_segmentation else 1))) * (2 if show_segmentation else 1)
            axes[row][col].axis('off')
            if show_segmentation:
                axes[row][col + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_sample_data(self, split: str = 'train', num_samples: int = 5) -> List[Dict]:
        """
        Get sample data for inspection.
        
        Args:
            split: Which split to sample from
            num_samples: Number of samples to return
            
        Returns:
            List of sample dictionaries
        """
        raw_datasets = self.load_raw_dataset([split])
        dataset = raw_datasets[split]
        
        samples = []
        for i, example in enumerate(dataset.take(num_samples)):
            sample = {
                'index': i,
                'image_shape': example['image'].shape.as_list(),
                'label': example['label'].numpy(),
                'class_name': self.class_names[example['label'].numpy()],
                'has_segmentation': 'segmentation_mask' in example
            }
            
            if 'segmentation_mask' in example:
                sample['segmentation_shape'] = example['segmentation_mask'].shape.as_list()
            
            samples.append(sample)
        
        return samples