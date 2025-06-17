from src.data import dataset_loader

dataset_loader = dataset_loader.OxfordPetDatasetLoader(
    data_dir="./data",
)
train_ds, val_ds, test_ds = dataset_loader.create_train_val_test_splits()

dataset = dataset_loader.visualize_samples(train_ds)