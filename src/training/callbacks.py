import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime
from src.config.config import Config
from src.visualization.results_visualizer import ResultsVisualizer
import pickle

class TrainingHistory(Callback):
    """Custom callback to track and save training history."""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.history = {}
        self.config = Config()
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
                
            # Save history periodically
            if epoch % 10 == 0:
                self.save_history()
                
    def save_history(self):
        """Save training history to file."""
        history_path = os.path.join(
            self.config.RESULTS_DIR,
            'history',
            f'{self.model_name}_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)

class ModelVisualizer(Callback):
    """Custom callback to visualize model predictions during training."""
    def __init__(self, validation_data, model_name, num_samples=5):
        super().__init__()
        self.validation_data = validation_data
        self.model_name = model_name
        self.num_samples = num_samples
        self.visualizer = ResultsVisualizer()
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Visualize every 5 epochs
            self.visualize_predictions()
            
    def visualize_predictions(self):
        """Visualize predictions on validation data."""
        samples = self.validation_data.take(self.num_samples)
        for sample in samples:
            image = sample['image']
            true_bbox = sample['bbox']
            true_mask = sample['segmentation_mask']
            
            # Get predictions
            if len(self.model.output_names) == 3:  # Multi-task model
                pred_bbox, pred_class, pred_mask = self.model.predict(image)
            elif len(self.model.output_names) == 2:  # Single task models
                if 'bbox' in self.model.output_names:
                    pred_bbox, pred_class = self.model.predict(image)
                    pred_mask = None
                else:
                    pred_mask = self.model.predict(image)
                    pred_bbox = None
                    pred_class = None
            
            # Visualize results
            if pred_bbox is not None:
                self.visualizer.visualize_detections(
                    image.numpy(),
                    true_bbox.numpy(),
                    pred_bbox.numpy(),
                    self.model_name
                )
            
            if pred_mask is not None:
                self.visualizer.visualize_segmentation(
                    image.numpy(),
                    true_mask.numpy(),
                    pred_mask.numpy(),
                    self.model_name
                )

def get_callbacks(model_name, validation_data, save_dir):
    """Create and return a list of callbacks."""
    return [
        TrainingHistory(model_name),
        ModelVisualizer(validation_data, model_name),
        ModelCheckpoint(
            filepath=os.path.join(save_dir, f'{model_name}_best.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-6
        )
    ]
