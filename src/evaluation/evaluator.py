from __future__ import annotations

"""evaluator.py
General-purpose Evaluation helper used across detection, segmentation and
multitask pipelines. The evaluator is intentionally light-weight so it can be
used both inside notebooks and scripts.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import json
import os
from datetime import datetime
import logging

from src.metrics.sota_metrics import SOTAMetrics

__all__ = ["Evaluator"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluator for model evaluation across different task types.
    Supports detection, segmentation, and multitask evaluation with
    customizable metrics and evaluation modes.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        task_type: str,
        num_classes: Optional[int] = None,
        custom_metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        batch_size: Optional[int] = None,
        mixed_precision: bool = True
    ):
        """
        Initialize evaluator with model and task type.

        Args:
            model: The model to evaluate
            task_type: Type of task ('detection', 'segmentation', 'multitask')
            num_classes: Number of classes for classification
            custom_metrics: Optional list of custom metrics to use
            batch_size: Optional batch size for evaluation
            mixed_precision: Whether to use mixed precision evaluation
        """
        self.model = model
        self.task_type = task_type.lower()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision
        self.custom_metrics = custom_metrics
        
        # Validate task type
        if self.task_type not in ['detection', 'segmentation', 'multitask']:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                "Must be one of ['detection', 'segmentation', 'multitask']"
            )
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Initialize evaluation state
        self._evaluation_state = {
            'is_evaluating': False,
            'current_batch': 0,
            'total_batches': 0,
            'last_results': None
        }
        
        # Enable mixed precision if requested
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics based on task type and custom metrics."""
        try:
            if self.custom_metrics is not None:
                self.metrics = self.custom_metrics
                return
            
            if self.task_type == 'detection':
                self.metrics = SOTAMetrics.get_detection_metrics_structured(
                    self.model,
                    num_classes=self.num_classes or 37
                )
            elif self.task_type == 'segmentation':
                if self.num_classes is None:
                    raise ValueError("num_classes must be provided for segmentation task")
                self.metrics = SOTAMetrics.get_segmentation_metrics_structured(
                    self.model,
                    num_classes=self.num_classes
                )
            elif self.task_type == 'multitask':
                if self.num_classes is None:
                    raise ValueError("num_classes must be provided for multitask evaluation")
                self.metrics = SOTAMetrics.get_multitask_metrics_structured(
                    self.model,
                    num_detection_classes=self.num_classes,
                    num_segmentation_classes=self.num_classes
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize metrics: {str(e)}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics to their initial state."""
        try:
            if isinstance(self.metrics, list):
                for metric in self.metrics:
                    metric.reset_state()
            else:
                self.metrics.reset_state()
        except Exception as e:
            raise RuntimeError(f"Failed to reset metrics: {str(e)}")
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        mode: str = 'standard',
        save_results: bool = False,
        results_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset and return metrics.

        Args:
            dataset: Dataset to evaluate on
            mode: Evaluation mode ('standard', 'detailed', 'fast')
            save_results: Whether to save results to disk
            results_dir: Directory to save results (if save_results is True)

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            self._evaluation_state['is_evaluating'] = True
            self._evaluation_state['current_batch'] = 0
            
            # Reset metrics
            self.reset_metrics()
            
            # Configure dataset
            if self.batch_size is not None:
                dataset = dataset.batch(self.batch_size)
            
            # Get total number of batches
            self._evaluation_state['total_batches'] = tf.data.experimental.cardinality(dataset).numpy()
            
            # Evaluate on dataset
            for batch in tqdm(dataset, desc="Evaluating", unit="batch"):
                images, targets = batch
                
                # Get model predictions
                predictions = self.model(images, training=False)
                
                # Update metrics based on task type
                if self.task_type == 'detection':
                    # For detection, update metrics for each output
                    for metric in self.metrics:
                        if metric.name.startswith('bbox'):
                            metric.update_state(targets['bbox'], predictions['bbox'])
                        elif metric.name.startswith('label'):
                            metric.update_state(targets['label'], predictions['label'])
                
                elif self.task_type == 'segmentation':
                    # For segmentation, update metrics directly
                    if isinstance(self.metrics, list):
                        for metric in self.metrics:
                            metric.update_state(targets, predictions)
                    else:
                        self.metrics.update_state(targets, predictions)
                
                elif self.task_type == 'multitask':
                    # For multitask, update metrics for each task
                    for metric in self.metrics:
                        if metric.name.startswith('detection'):
                            if 'bbox' in metric.name:
                                metric.update_state(targets['bbox'], predictions['bbox'])
                            elif 'label' in metric.name:
                                metric.update_state(targets['label'], predictions['label'])
                        elif metric.name.startswith('segmentation'):
                            metric.update_state(targets['segmentation'], predictions['segmentation'])
                
                self._evaluation_state['current_batch'] += 1
            
            # Get metric results
            results = self._get_metric_results()
            
            # Save results if requested
            if save_results:
                self._save_results(results, results_dir)
            
            self._evaluation_state['last_results'] = results
            return results
            
        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {str(e)}")
        finally:
            self._evaluation_state['is_evaluating'] = False
    
    def _get_metric_results(self) -> Dict[str, Any]:
        """Get results from all metrics."""
        try:
            if isinstance(self.metrics, list):
                results = {}
                for metric in self.metrics:
                    results[metric.name] = metric.result().numpy()
                return results
            else:
                return {self.metrics.name: self.metrics.result().numpy()}
        except Exception as e:
            raise RuntimeError(f"Failed to get metric results: {str(e)}")
    
    def _save_results(
        self,
        results: Dict[str, Any],
        results_dir: Optional[str] = None
    ) -> str:
        """
        Save evaluation results to disk.

        Args:
            results: Results to save
            results_dir: Directory to save results

        Returns:
            Path where results were saved
        """
        try:
            if results_dir is None:
                results_dir = os.path.join('results', 'evaluation')
            
            # Create directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{self.task_type}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
            
            logger.info(f"Evaluation results saved to {filepath}")
            return filepath
            
        except Exception as e:
            raise RuntimeError(f"Failed to save results: {str(e)}")
    
    def get_evaluation_state(self) -> Dict[str, Any]:
        """Get current evaluation state."""
        return self._evaluation_state.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        try:
            if self._evaluation_state['last_results'] is None:
                raise ValueError("No evaluation results available")
            
            summary = {
                'task_type': self.task_type,
                'num_classes': self.num_classes,
                'metrics': self._evaluation_state['last_results'],
                'evaluation_state': self._evaluation_state
            }
            
            return summary
            
        except Exception as e:
            raise RuntimeError(f"Failed to get metrics summary: {str(e)}")
    
    def evaluate_batch(
        self,
        images: tf.Tensor,
        targets: Union[tf.Tensor, Dict[str, tf.Tensor]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single batch of data.

        Args:
            images: Batch of input images
            targets: Batch of target values (tensor or dictionary of tensors)

        Returns:
            Dictionary containing evaluation metrics for the batch
        """
        try:
            # Get model predictions
            predictions = self.model(images, training=False)
            
            # Update metrics based on task type
            if self.task_type == 'detection':
                # For detection, update metrics for each output
                for metric in self.metrics:
                    if metric.name.startswith('bbox'):
                        metric.update_state(targets['bbox'], predictions['bbox'])
                    elif metric.name.startswith('label'):
                        metric.update_state(targets['label'], predictions['label'])
            
            elif self.task_type == 'segmentation':
                # For segmentation, update metrics directly
                if isinstance(self.metrics, list):
                    for metric in self.metrics:
                        metric.update_state(targets, predictions)
                else:
                    self.metrics.update_state(targets, predictions)
            
            elif self.task_type == 'multitask':
                # For multitask, update metrics for each task
                for metric in self.metrics:
                    if metric.name.startswith('detection'):
                        if 'bbox' in metric.name:
                            metric.update_state(targets['bbox'], predictions['bbox'])
                        elif 'label' in metric.name:
                            metric.update_state(targets['label'], predictions['label'])
                    elif metric.name.startswith('segmentation'):
                        metric.update_state(targets['segmentation'], predictions['segmentation'])
            
            # Get current results
            return self._get_metric_results()
            
        except Exception as e:
            raise RuntimeError(f"Batch evaluation failed: {str(e)}")
    
    def evaluate_single(
        self,
        image: tf.Tensor,
        target: Union[tf.Tensor, Dict[str, tf.Tensor]]
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample.

        Args:
            image: Single input image
            target: Single target value (tensor or dictionary of tensors)

        Returns:
            Dictionary containing evaluation metrics for the sample
        """
        try:
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = tf.expand_dims(image, 0)
            
            # Add batch dimension to target if needed
            if isinstance(target, dict):
                target = {k: tf.expand_dims(v, 0) if len(v.shape) == 2 else v 
                         for k, v in target.items()}
            elif len(target.shape) == 2:
                target = tf.expand_dims(target, 0)
            
            return self.evaluate_batch(image, target)
            
        except Exception as e:
            raise RuntimeError(f"Single sample evaluation failed: {str(e)}")
