import os
import json
import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ResultManager:
    """Manager for saving and organizing experiment results."""
    
    def __init__(self, base_dir: str = "results", experiment_dir: str = "experiments"):
        """Initialize result manager with base directories."""
        self.base_dir = Path(base_dir)
        self.experiment_dir = Path(experiment_dir)
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        # Results directories
        (self.base_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "reports").mkdir(parents=True, exist_ok=True)
        
        # Experiment directories
        (self.experiment_dir / "detection").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "segmentation").mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "multitask").mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: tf.keras.Model, task_type: str, name: Optional[str] = None):
        """Save model weights and architecture."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = name or f"model_{timestamp}"
        
        # Save model weights
        weights_path = self.base_dir / "models" / f"{model_name}_weights.h5"
        model.save_weights(str(weights_path))
        
        # Save model architecture
        arch_path = self.base_dir / "models" / f"{model_name}_architecture.json"
        with open(arch_path, 'w') as f:
            f.write(model.to_json())
        
        return str(weights_path), str(arch_path)
    
    def save_metrics(self, metrics: Dict[str, Any], task_type: str, name: Optional[str] = None):
        """Save evaluation metrics."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_name = name or f"metrics_{timestamp}"
        
        # Save metrics to JSON
        metrics_path = self.base_dir / "reports" / f"{metrics_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return str(metrics_path)
    
    def save_predictions(self, images: np.ndarray, predictions: Dict[str, np.ndarray], 
                        task_type: str, name: Optional[str] = None):
        """Save model predictions."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_name = name or f"predictions_{timestamp}"
        
        # Save predictions as numpy arrays
        pred_dir = self.base_dir / "predictions" / pred_name
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for key, value in predictions.items():
            np.save(str(pred_dir / f"{key}.npy"), value)
        
        return str(pred_dir)
    
    def save_plots(self, history: Dict[str, Any], task_type: str, name: Optional[str] = None):
        """Save training plots."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_name = name or f"plots_{timestamp}"
        
        # Create plots directory
        plot_dir = self.base_dir / "plots" / plot_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot training metrics
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy if available
        if 'accuracy' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(str(plot_dir / 'training_history.png'))
        plt.close()
        
        return str(plot_dir)
    
    def save_experiment_config(self, config: Dict[str, Any], task_type: str, name: Optional[str] = None):
        """Save experiment configuration."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = name or f"config_{timestamp}"
        
        # Save config to JSON
        config_path = self.experiment_dir / task_type / f"{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return str(config_path)
    
    def save_experiment_results(self, results: Dict[str, Any], task_type: str, name: Optional[str] = None):
        """Save complete experiment results."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_name = name or f"results_{timestamp}"
        
        # Save results to JSON
        results_path = self.experiment_dir / task_type / f"{results_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return str(results_path) 