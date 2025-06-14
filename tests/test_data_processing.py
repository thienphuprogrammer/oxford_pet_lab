from src.config import Config
from src.data import DataPreprocessor

config = Config()
config.USE_IMAGENET_NORM = True          # For transfer learning
config.PRESERVE_ASPECT_RATIO = True      # Better image quality
config.ENABLE_QUALITY_ENHANCEMENT = True # Image enhancement
config.NORMALIZATION_METHOD = 'imagenet' # Best for pretrained models

preprocessor = DataPreprocessor(config=config, shuffle_buffer=5000)

train_ds = preprocessor.create_training_dataset(
    train_raw_ds, 
    batch_size=32,
    task="multitask",
    cache_filename="train_cache"  # File caching for speed
)

val_ds = preprocessor.create_validation_dataset(
    val_raw_ds,
    batch_size=32, 
    task="multitask"
)

stats = preprocessor.get_dataset_statistics(train_ds)
print(f"Dataset stats: {stats}")