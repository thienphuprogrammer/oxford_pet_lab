from src.data import OxfordPetDatasetLoader

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


    from src.config import Config
    from src.data import DataPreprocessor

    config = Config()
    config.USE_IMAGENET_NORM = True          # For transfer learning
    config.PRESERVE_ASPECT_RATIO = True      # Better image quality
    config.ENABLE_QUALITY_ENHANCEMENT = True # Image enhancement
    config.NORMALIZATION_METHOD = 'imagenet' # Best for pretrained models

    preprocessor = DataPreprocessor(config=config, shuffle_buffer=5000)

    train_ds = preprocessor.create_training_dataset(
        train_ds, 
        batch_size=32,
        task="multitask",
        cache_filename="train_cache"  # File caching for speed
    )

    val_ds = preprocessor.create_validation_dataset(
        val_processed,
        batch_size=32, 
        task="multitask"
    )
    
    stats = preprocessor.get_dataset_statistics(train_ds)
    print(f"Dataset stats: {stats}")


if __name__ == "__main__":
    main()