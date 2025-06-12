import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..config.config import Config
from ..models.detection_model import DetectionModel
from ..models.segmentation_model import SegmentationModel
from ..models.multitask_model import MultiTaskModel
from ..data.dataset_loader import OxfordPetDatasetLoader
from ..data.preprocessor import DataPreprocessor
from .callbacks import get_callbacks
from .metrics import DetectionMetrics, SegmentationMetrics, MultiTaskMetrics

class Trainer:
    def __init__(self, model_type='detection', model_name='resnet50', pretrained=True):
        """Initialize trainer with specified model type and configuration."""
        self.config = Config()
        self.model_type = model_type
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Initialize data loader and preprocessor
        self.data_loader = OxfordPetDatasetLoader()
        self.preprocessor = DataPreprocessor()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize metrics
        self.metrics = {
            'detection': DetectionMetrics,
            'segmentation': SegmentationMetrics,
            'multitask': MultiTaskMetrics
        }
        
    def _initialize_model(self):
        """Initialize the appropriate model based on configuration."""
        if self.model_type == 'detection':
            return DetectionModel(self.model_name, self.pretrained)
        elif self.model_type == 'segmentation':
            return SegmentationModel(f'unet_{self.model_name}', self.pretrained)
        elif self.model_type == 'multitask':
            return MultiTaskModel(self.pretrained)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def prepare_data(self):
        """Prepare and preprocess the dataset."""
        # Load dataset
        train_ds, val_ds, test_ds = self.data_loader.load_dataset()
        
        # Preprocess data
        train_ds = self.preprocessor.prepare_dataset(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        val_ds = self.preprocessor.prepare_dataset(
            val_ds,
            batch_size=self.config.BATCH_SIZE
        )
        
        test_ds = self.preprocessor.prepare_dataset(
            test_ds,
            batch_size=self.config.BATCH_SIZE
        )
        
        return train_ds, val_ds, test_ds
    
    def compile_model(self):
        """Compile the model with appropriate loss functions and metrics."""
        if self.model_type == 'detection':
            self.model.compile(
                learning_rate=self.config.LEARNING_RATE
            )
        elif self.model_type == 'segmentation':
            self.model.compile(
                learning_rate=self.config.LEARNING_RATE
            )
        elif self.model_type == 'multitask':
            self.model.compile(
                learning_rate=self.config.LEARNING_RATE
            )
        
    def train(self, epochs=None):
        """Train the model."""
        if epochs is None:
            epochs = self.config.EPOCHS
            
        # Prepare data
        train_ds, val_ds, _ = self.prepare_data()
        
        # Compile model
        self.compile_model()
        
        # Get callbacks
        callbacks = get_callbacks(
            f'{self.model_type}_{self.model_name}',
            val_ds,
            os.path.join(self.config.MODELS_DIR, self.model_type)
        )
        
        # Train model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_ds=None):
        """Evaluate the model on test data."""
        if test_ds is None:
            _, _, test_ds = self.prepare_data()
            
        results = self.model.evaluate(test_ds)
        
        # Print evaluation metrics
        print(f"\nEvaluation results for {self.model_type}_{self.model_name}:")
        for metric, value in zip(self.model.metrics_names, results):
            print(f"{metric}: {value:.4f}")
            
        return results
    
    def predict(self, image):
        """Make predictions on a single image."""
        # Preprocess image
        image = self.preprocessor.preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(image)
        
        return predictions
    
    def save_model(self, filepath=None):
        """Save the trained model."""
        if filepath is None:
            filepath = os.path.join(
                self.config.MODELS_DIR,
                self.model_type,
                f'{self.model_type}_{self.model_name}.h5'
            )
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from: {filepath}")
