import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict, Tuple

class BaseDetectionModel(Model):
  def __init__(self, num_classes: int, **kwargs):
    super().__init__(**kwargs)
    self.num_classes = num_classes

  def call(
      self,
      inputs: Dict[str, tf.Tensor],
      training: bool = False,
  ) -> Dict[str, tf.Tensor]:
    raise NotImplementedError("Subclasses must implement the call method.")
  