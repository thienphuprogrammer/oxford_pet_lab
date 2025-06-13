# data/dataset_loader.py
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Tuple, Dict, Any
import logging
from src.config.config import Config
import pandas as pd
import os

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class OxfordPetDatasetLoader:
    """Dataset loader for Oxford-IIIT Pet Dataset."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dataset_info = None
        self.class_names = None
        
    def load_annotations(self) -> pd.DataFrame:
        """Load all annotations from XML files"""
        annotations = []
        
        if not os.path.exists(self.config.XMLS_PATH):
            self.logger.error(f"Annotations path does not exist: {self.config.XMLS_PATH}")
            return pd.DataFrame()
        
        xml_files = [f for f in os.listdir(self.config.XMLS_PATH) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(self.config.XMLS_PATH, xml_file)
            
            try:
                annotation = self._parse_xml_annotation(xml_path)
                if annotation:
                    annotations.append(annotation)
            except Exception as e:
                self.logger.warning(f"Failed to parse {xml_file}: {e}")
                continue
        
        df = pd.DataFrame(annotations)
        self.logger.info(f"Loaded {len(df)} annotations")
        
        return df


    def _parse_xml_annotation(self, xml_path: str) -> Optional[Dict]:
        """Parse XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image info
            filename = root.find('filename').text
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            # Get object info
            obj = root.find('object')
            class_name = obj.find('name').text
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Normalize coordinates
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height
            
            return {
                'filename': filename,
                'class_name': class_name,
                'class_id': self.class_to_idx.get(class_name, -1),
                'img_width': img_width,
                'img_height': img_height,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'xmin_norm': xmin_norm,
                'ymin_norm': ymin_norm,
                'xmax_norm': xmax_norm,
                'ymax_norm': ymax_norm,
                'bbox_area': (xmax - xmin) * (ymax - ymin),
                'bbox_area_norm': (xmax_norm - xmin_norm) * (ymax_norm - ymin_norm)
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing {xml_path}: {e}")
            return None
    
    def load_segmentation_masks(self) -> Dict[str, str]:
        """Load segmentation mask file paths"""
        mask_paths = {}
        
        if not os.path.exists(self.config.TRIMAPS_PATH):
            self.logger.warning(f"Trimaps path does not exist: {self.config.TRIMAPS_PATH}")
            return mask_paths
        
        mask_files = [f for f in os.listdir(self.config.TRIMAPS_PATH) if f.endswith('.png')]
        
        for mask_file in mask_files:
            base_name = mask_file.replace('.png', '.jpg')
            mask_paths[base_name] = os.path.join(self.config.TRIMAPS_PATH, mask_file)
        
        self.logger.info(f"Found {len(mask_paths)} segmentation masks")
        return mask_paths

    def split_dataset(
        self, 
        annotations_df: pd.DataFrame,
        test_size: float = None,
        val_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets"""
        
        test_size = test_size or self.config.TEST_SPLIT
        val_size = val_size or self.config.VALIDATION_SPLIT
        random_state = random_state or self.config.RANDOM_SEED
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            annotations_df,
            test_size=test_size,
            random_state=random_state,
            stratify=annotations_df['class_name']
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=train_val_df['class_name']
        )
        
        self.logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df

    def create_tf_dataset_detection(
        self, 
        annotations_df: pd.DataFrame,
    ):
        pass

    def create_tf_dataset_segmentation(
        self, 
        annotations_df: pd.DataFrame,
    ):
        pass

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
