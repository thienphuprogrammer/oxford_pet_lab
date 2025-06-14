from typing import Dict
import tensorflow as tf

from src.training.base import BaseTrainer

class Trainer(BaseTrainer):
    """Main trainer implementation for single-task learning."""
    
    @tf.function
    def train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized training step."""
        if self.task_type == 'detection':
            return self._detection_train_step(batch)
        elif self.task_type == 'segmentation':
            return self._segmentation_train_step(batch)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
    @tf.function
    def validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized validation step."""
        if self.task_type == 'detection':
            return self._detection_validation_step(batch)
        elif self.task_type == 'segmentation':
            return self._segmentation_validation_step(batch)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
    @tf.function
    def _detection_train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized detection training step."""
        images, targets = batch
        predictions = self.model(images, training=True)
        loss = self.loss_fn(targets, predictions)
        
        # Update metrics
        self.metrics.update_state(targets, predictions)
        
        return {'loss': loss}
        
    @tf.function
    def _segmentation_train_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized segmentation training step."""
        images, masks = batch
        predictions = self.model(images, training=True)
        loss = self.loss_fn(masks, predictions)
        
        # Update metrics
        self.metrics.update_state(masks, predictions)
        
        return {'loss': loss}
        
    @tf.function
    def _detection_validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized detection validation step."""
        images, targets = batch
        predictions = self.model(images, training=False)
        loss = self.loss_fn(targets, predictions)
        
        return {'loss': loss}
        
    @tf.function
    def _segmentation_validation_step(self, batch) -> Dict[str, tf.Tensor]:
        """Optimized segmentation validation step."""
        images, masks = batch
        predictions = self.model(images, training=False)
        loss = self.loss_fn(masks, predictions)
        
        return {'loss': loss}
