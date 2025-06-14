from src.data import OxfordPetDatasetLoader
from src.config import Config
from src.data import DataPreprocessor, DataAugmentor


config = Config()
config.USE_IMAGENET_NORM = True          # For transfer learning
config.PRESERVE_ASPECT_RATIO = True      # Better image quality
config.ENABLE_QUALITY_ENHANCEMENT = True # Image enhancement
config.NORMALIZATION_METHOD = 'imagenet' # Best for pretrained models


def main():
    loader = OxfordPetDatasetLoader(
        data_dir="./tensorflow_datasets",  # Tùy chọn
        download=True,
        log_level='INFO'
    )

    train_ds, val_ds, test_ds = loader.create_train_val_test_splits(
        val_split=0.2,
        seed=42
    )

    # Concate train and test datasets
    train_ds = train_ds.concatenate(test_ds)


    augmentor = DataAugmentor(
        config=config,
        prob_geo=0.7,
        prob_photo=0.8,
        prob_mixup=0.3,
        prob_cutout=0.2,
        prob_mosaic=0.3,
    )
    
    train_ds = augmentor.create_augmented_dataset(
        train_ds,
        augmentation_factor=3
    )
    
    val_ds = augmentor.create_augmented_dataset(
        val_ds,
        augmentation_factor=1
    )

    preprocessor = DataPreprocessor(config=config, shuffle_buffer=5000)

    train_ds = preprocessor.create_training_dataset(
        train_ds, 
        batch_size=32,
        task="detection",
        cache_filename="train_cache"  # File caching for speed
    )

    val_ds = preprocessor.create_validation_dataset(
        val_ds,
        batch_size=32, 
        task="detection"
    )
    
    # stats = preprocessor.get_dataset_statistics(train_ds)
    


if __name__ == "__main__":
    main()