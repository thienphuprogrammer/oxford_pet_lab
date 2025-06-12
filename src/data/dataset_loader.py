# data/dataset_loader.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Any
import logging
from config.config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class OxfordPetDatasetLoader:
    """Dataset loader for Oxford-IIIT Pet Dataset."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dataset_info = None
        self.class_names = None
        
    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and split the Oxford-IIIT Pet dataset.
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        logger.info(f"Loading dataset: {self.config.DATASET_NAME}")
        
        # Load the complete dataset
        (train_ds, test_ds), dataset_info = tfds.load(
            self.config.DATASET_NAME,
            split=['train', 'test'],
            with_info=True,
            as_supervised=False,
        )
        
        self.dataset_info = dataset_info
        self.class_names = dataset_info.features['label'].names
        
        logger.info(f"Dataset info: {dataset_info}")
        logger.info(f"Number of classes: {len(self.class_names)}")
        logger.info(f"Train samples: {dataset_info.splits['train'].num_examples}")
        logger.info(f"Test samples: {dataset_info.splits['test'].num_examples}")
        
        # Split training data into train and validation
        train_size = dataset_info.splits['train'].num_examples
        val_size = int(train_size * self.config.VALIDATION_SPLIT)
        train_size = train_size - val_size
        
        # Shuffle and split
        train_ds = train_ds.shuffle(buffer_size=1000, seed=self.config.RANDOM_SEED)
        val_ds = train_ds.take(val_size)
        train_ds = train_ds.skip(val_size)
        
        logger.info(f"Final split - Train: {train_size}, Val: {val_size}, Test: {dataset_info.splits['test'].num_examples}")
        
        return train_ds, val_ds, test_ds
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.dataset_info is None:
            raise ValueError("Dataset not loaded yet. Call load_dataset() first.")
        
        return {
            'total_classes': len(self.class_names),
            'class_names': self.class_names,
            'features': self.dataset_info.features,
            'splits_info': self.dataset_info.splits,
            'description': self.dataset_info.description,
        }
