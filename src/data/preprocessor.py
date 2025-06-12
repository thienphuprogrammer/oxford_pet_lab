import tensorflow as tf
from tensorflow.keras.layers import Resizing, Rescaling
from ..config.config import Config

class DataPreprocessor:
    def __init__(self, config: Config = None):
        self.config = congif or Config()
        self.resize_layer = Resizing(
            self.config.IMG_HEIGHT,
            self.config.IMG_WIDTH,
            interpolation='bilinear'
        )
        self.rescale_layer = Rescaling(1./255)

    def resize_and_rescale(self, image, mask=None):
        """Resize and rescale image and mask."""
        image = self.resize_layer(image)
        image = self.rescale_layer(image)
        
        if mask is not None:
            mask = self.resize_layer(mask)
            mask = tf.cast(mask, tf.float32) / 255.0
            return image, mask
        
        return image

    def normalize_bbox(self, bbox, img_shape):
        """Normalize bounding box coordinates."""
        height, width = img_shape[:2]
        x_min, y_min, x_max, y_max = bbox
        
        x_min = x_min / width
        y_min = y_min / height
        x_max = x_max / width
        y_max = y_max / height
        
        return [x_min, y_min, x_max, y_max]

    def denormalize_bbox(self, bbox, img_shape):
        """Denormalize bounding box coordinates."""
        height, width = img_shape[:2]
        x_min, y_min, x_max, y_max = bbox
        
        x_min = x_min * width
        y_min = y_min * height
        x_max = x_max * width
        y_max = y_max * height
        
        return [x_min, y_min, x_max, y_max]

    def convert_bbox_format(self, bbox, from_format='xywh', to_format='xyxy'):
        """Convert bounding box format."""
        if from_format == 'xywh' and to_format == 'xyxy':
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        elif from_format == 'xyxy' and to_format == 'xywh':
            x1, y1, x2, y2 = bbox
            return [x1, y1, x2 - x1, y2 - y1]
        return bbox

    def preprocess_image(self, image, mask=None):
        """Complete preprocessing pipeline for image."""
        image = tf.cast(image, tf.float32)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            return self.resize_and_rescale(image, mask)
        return self.resize_and_rescale(image)

    def preprocess_bbox(self, bbox, img_shape):
        """Complete preprocessing pipeline for bounding box."""
        bbox = self.convert_bbox_format(bbox, 'xywh', 'xyxy')
        return self.normalize_bbox(bbox, img_shape)

    def postprocess_bbox(self, bbox, img_shape):
        """Complete postprocessing pipeline for bounding box."""
        bbox = self.denormalize_bbox(bbox, img_shape)
        return self.convert_bbox_format(bbox, 'xyxy', 'xywh')

    def prepare_dataset(self, dataset, batch_size=None, shuffle=False):
        """Prepare dataset with preprocessing and batching."""
        if shuffle:
            dataset = dataset.shuffle(1000)
        
        dataset = dataset.map(
            lambda x: self.preprocess_image(x['image'], x.get('segmentation_mask', None)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if batch_size:
            dataset = dataset.batch(batch_size)
        
        return dataset.prefetch(tf.data.AUTOTUNE)
