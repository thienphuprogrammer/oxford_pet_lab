import matplotlib.pyplot as plt
import numpy as np
from src.data import dataset_loader

def plot_class_distribution(class_dist):
    classes = list(class_dist.keys())
    counts = list(class_dist.values())
    
    plt.figure(figsize=(15, 8))
    plt.barh(classes, counts, color='skyblue')
    plt.xlabel('Number of Examples')
    plt.title('Class Distribution in Oxford-IIIT Pet Dataset')
    plt.tight_layout()
    plt.savefig('results/plots/dataset/class_distribution.png')
    plt.close()

def plot_image_dimensions(dim_stats):
    plt.figure(figsize=(12, 6))
    
    # Height distribution
    plt.subplot(1, 2, 1)
    plt.hist(np.random.normal(dim_stats['height']['mean'], dim_stats['height']['std'], 1000), bins=30, color='lightcoral')
    plt.title('Image Height Distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    
    # Width distribution
    plt.subplot(1, 2, 2)
    plt.hist(np.random.normal(dim_stats['width']['mean'], dim_stats['width']['std'], 1000), bins=30, color='lightblue')
    plt.title('Image Width Distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/plots/dataset/image_dimensions.png')
    plt.close()

def plot_split_distribution(split_info):
    splits = list(split_info.keys())
    counts = [info.num_examples for info in split_info.values()]
    
    plt.figure(figsize=(8, 6))
    plt.bar(splits, counts, color='lightgreen')
    plt.title('Dataset Split Distribution')
    plt.ylabel('Number of Examples')
    plt.tight_layout()
    plt.savefig('results/plots/dataset/split_distribution.png')
    plt.close()

# Load dataset and get statistics
dataset_loader = dataset_loader.OxfordPetDatasetLoader(
    data_dir="./data",
)
train_ds, val_ds, test_ds = dataset_loader.create_train_val_test_splits()

stats = dataset_loader.get_dataset_statistics(train_ds.concatenate(val_ds).concatenate(test_ds))
print("\nDataset statistics:")
print(stats)

# Create plots
plot_class_distribution(stats['class_distribution'])
plot_image_dimensions(stats['image_summary'])
plot_split_distribution(stats['split_info'])

print("\nPlots have been saved to:")
print("- class_distribution.png")
print("- image_dimensions.png")
print("- split_distribution.png")
