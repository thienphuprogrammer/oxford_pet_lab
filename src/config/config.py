# config/config.py
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Global configuration for the Oxford Pet Lab project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    MODELS_DIR = RESULTS_DIR / "models"
    LOGS_DIR = RESULTS_DIR / "logs"
    PLOTS_DIR = RESULTS_DIR / "plots"
    PREDICTIONS_DIR = RESULTS_DIR / "predictions"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, PREDICTIONS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration
    DATASET_NAME = "oxford_iiit_pet:4.0.0"
    DATASET_VERSION = "4.0.0"
    
    # Data preprocessing
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    
    # Training configuration
    BATCH_SIZE = 16
    VALIDATION_SPLIT = 0.2
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    
    # Model configuration
    NUM_CLASSES = 37  # 37 different pet breeds
    LEARNING_RATE = 1e-4
    
    # Bounding box configuration
    BBOX_FORMAT = "xyxy"  # x1, y1, x2, y2 format
    
    # Augmentation configuration
    ENABLE_AUGMENTATION = True
    AUGMENTATION_STRENGTH = 0.2
    
    # Reproducibility
    RANDOM_SEED = 42
    
    # Logging
    LOG_LEVEL = "INFO"
    TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"
    
    # Model checkpointing
    CHECKPOINT_MONITOR = "val_loss"
    CHECKPOINT_MODE = "min"
    SAVE_BEST_ONLY = True
    
    # Visualization
    PLOT_STYLE = "seaborn-v0_8"
    FIGURE_SIZE = (12, 8)
    DPI = 100


    @classmethod
    def setup_gpu(cls):
        """Setup GPU configuration"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, cls.GPU_MEMORY_GROWTH)
                if cls.MIXED_PRECISION:
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.MODEL_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.PLOTS_DIR,
            "results/predictions",
            "experiments/detection/with_pretrained",
            "experiments/detection/without_pretrained",
            "experiments/segmentation/with_pretrained",
            "experiments/segmentation/without_pretrained",
            "experiments/multitask"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


    @classmethod
    def get_class_names(cls) -> List[str]:
        """Get Oxford Pet dataset class names"""
        return [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
            'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
            'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
            'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
            'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
            'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
            'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
            'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier'
        ]

    @classmethod
    def get_segmentation_class_names(cls) -> List[str]:
        """Get segmentation class names"""
        return ['background', 'pet', 'border']