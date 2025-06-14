import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
from src.config.config import Config

class DataVisualizer:
    def __init__(self):
        self.config = Config()

    def visualize_sample(self, image, bbox=None, mask=None, breed=None):
        """Visualize a single sample with optional bounding box and mask."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(array_to_img(image))
        plt.title('Original Image')
        plt.axis('off')

        # Bounding box
        if bbox is not None:
            plt.subplot(1, 3, 2)
            plt.imshow(array_to_img(image))
            plt.plot([bbox[0], bbox[2]], [bbox[1], bbox[1]], 'r-')
            plt.plot([bbox[0], bbox[2]], [bbox[3], bbox[3]], 'r-')
            plt.plot([bbox[0], bbox[0]], [bbox[1], bbox[3]], 'r-')
            plt.plot([bbox[2], bbox[2]], [bbox[1], bbox[3]], 'r-')
            plt.title('With Bounding Box')
            plt.axis('off')

        # Mask
        if mask is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(array_to_img(image))
            plt.imshow(mask, alpha=0.5, cmap='gray')
            plt.title('With Mask')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_breed_distribution(self, dataset):
        """Plot distribution of pet breeds in the dataset."""
        breed_counts = {}
        for data in dataset:
            breed = data['label'].numpy()
            breed_counts[breed] = breed_counts.get(breed, 0) + 1

        plt.figure(figsize=(15, 5))
        plt.bar(range(len(breed_counts)), list(breed_counts.values()))
        plt.title('Distribution of Pet Breeds')
        plt.xlabel('Breed Index')
        plt.ylabel('Number of Samples')
        plt.xticks(range(len(breed_counts)), rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_image_size_distribution(self, dataset):
        """Plot distribution of image sizes in the dataset."""
        widths = []
        heights = []
        
        for data in dataset:
            image = data['image'].numpy()
            heights.append(image.shape[0])
            widths.append(image.shape[1])

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(widths, bins=30)
        plt.title('Image Width Distribution')
        plt.xlabel('Width (pixels)')
        
        plt.subplot(1, 2, 2)
        sns.histplot(heights, bins=30)
        plt.title('Image Height Distribution')
        plt.xlabel('Height (pixels)')
        
        plt.tight_layout()
        plt.show()

    def visualize_augmentation(self, image, bbox=None, mask=None, num_samples=5):
        """Visualize data augmentation results."""
        plt.figure(figsize=(15, 5))
        
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            # Apply augmentation here
            augmented_image = image  # Placeholder for augmentation
            plt.imshow(array_to_img(augmented_image))
            plt.title(f'Augmented {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()
