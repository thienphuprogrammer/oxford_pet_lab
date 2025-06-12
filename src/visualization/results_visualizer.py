import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..config.config import Config

class ResultsVisualizer:
    def __init__(self):
        self.config = Config()

    def plot_training_curves(self, history, model_name):
        """Plot training and validation curves."""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy curves if available
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'{model_name} - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name}_training_curves.png')
        plt.show()

    def visualize_detections(self, image, true_bbox, pred_bbox, model_name):
        """Visualize detection results."""
        plt.figure(figsize=(10, 5))
        
        # True bounding box
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.plot([true_bbox[0], true_bbox[2]], [true_bbox[1], true_bbox[1]], 'r-')
        plt.plot([true_bbox[0], true_bbox[2]], [true_bbox[3], true_bbox[3]], 'r-')
        plt.plot([true_bbox[0], true_bbox[0]], [true_bbox[1], true_bbox[3]], 'r-')
        plt.plot([true_bbox[2], true_bbox[2]], [true_bbox[1], true_bbox[3]], 'r-')
        plt.title(f'{model_name} - True Bounding Box')
        plt.axis('off')
        
        # Predicted bounding box
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.plot([pred_bbox[0], pred_bbox[2]], [pred_bbox[1], pred_bbox[1]], 'b-')
        plt.plot([pred_bbox[0], pred_bbox[2]], [pred_bbox[3], pred_bbox[3]], 'b-')
        plt.plot([pred_bbox[0], pred_bbox[0]], [pred_bbox[1], pred_bbox[3]], 'b-')
        plt.plot([pred_bbox[2], pred_bbox[2]], [pred_bbox[1], pred_bbox[3]], 'b-')
        plt.title(f'{model_name} - Predicted Bounding Box')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/predictions/{model_name}_detection.png')
        plt.show()

    def visualize_segmentation(self, image, true_mask, pred_mask, model_name):
        """Visualize segmentation results."""
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # True mask
        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(true_mask, alpha=0.5, cmap='gray')
        plt.title('True Mask')
        plt.axis('off')
        
        # Predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(pred_mask, alpha=0.5, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/predictions/{model_name}_segmentation.png')
        plt.show()

    def plot_confusion_matrix(self, cm, classes, model_name):
        """Plot confusion matrix for classification results."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name}_confusion_matrix.png')
        plt.show()

    def plot_iou_distribution(self, iou_scores, model_name):
        """Plot distribution of IoU scores."""
        plt.figure(figsize=(10, 5))
        sns.histplot(iou_scores, bins=30)
        plt.title(f'{model_name} - IoU Score Distribution')
        plt.xlabel('IoU Score')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig(f'results/plots/{model_name}_iou_distribution.png')
        plt.show()
