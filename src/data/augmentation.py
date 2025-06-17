import tensorflow as tf
from typing import Dict, Any, Optional
from src.config.config import Config
from src.data.preprocessing import DataPreprocessor 


class DataAugmentor:
    def __init__(
        self, 
        config: Config = None,
        target_height: int = 224, 
        target_width: int = 224,
        prob_flip: float = 0.5,
        prob_photo: float = 0.7,
        prob_cutout: float = 0.2,
        cutout_frac: float = 0.25,
    ):
        self.config = config or Config()
        self.target_height = target_height
        self.target_width = target_width
        self.prob_flip = prob_flip
        self.prob_photo = prob_photo
        self.prob_cutout = prob_cutout
        self.cutout_frac = cutout_frac

        self.preprocessor = DataPreprocessor(config=self.config)

    def _flip_bbox(self, bbox: tf.Tensor) -> tf.Tensor:
        xmin, ymin, xmax, ymax = tf.unstack(bbox)
        return tf.stack([1.0 - xmax, ymin, 1.0 - xmin, ymax])

    def random_flip_horizontal(self, image, bbox, mask, seed):
        flip = tf.random.stateless_uniform([], seed=seed) < self.prob_flip
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
        bbox = tf.cond(flip, lambda: self._flip_bbox(bbox), lambda: bbox)
        return image, bbox, mask
    
    
    def photometric(self, image, seed):
        seeds = tf.random.experimental.stateless_split(seed, 4)
        image = tf.image.stateless_random_brightness(image, 0.2, seeds[0])
        image = tf.image.stateless_random_contrast(image, 0.8, 1.2, seeds[1])
        image = tf.image.stateless_random_saturation(image, 0.8, 1.2, seeds[2])
        image = tf.image.stateless_random_hue(image, 0.05, seeds[3])
        return tf.clip_by_value(image, 0., 1.)

    def random_cutout(self, image, mask, seed):
        h, w = self.target_height, self.target_width
        cutout_size = int(h * self.cutout_frac)
        seeds = tf.random.experimental.stateless_split(seed, 2)
        offset_h = tf.random.stateless_uniform([], seed=seeds[0], minval=0, maxval=h-cutout_size, dtype=tf.int32)
        offset_w = tf.random.stateless_uniform([], seed=seeds[1], minval=0, maxval=w-cutout_size, dtype=tf.int32)
        mask_shape = [h, w, 1]
        cutout_mask = tf.ones(mask_shape, tf.float32)
        cutout_mask = tf.tensor_scatter_nd_update(
            cutout_mask, 
            indices=tf.stack(
                [tf.repeat(tf.range(offset_h, offset_h+cutout_size), cutout_size),
                 tf.tile(tf.range(offset_w, offset_w+cutout_size), [cutout_size]),
                 tf.zeros([cutout_size*cutout_size], dtype=tf.int32)],
                axis=1),
            updates=tf.zeros([cutout_size*cutout_size], tf.float32)
        )
        image = image * cutout_mask
        return image, mask

    def augment_sample(self, sample: Dict[str, tf.Tensor], seed: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        processed = self.preprocessor.preprocess_sample(sample)
        image = processed['image']
        bbox = processed['head_bbox']
        mask = processed['segmentation_mask']

        image = tf.image.resize(image, [self.target_height, self.target_width])
        mask = tf.image.resize(mask, [self.target_height, self.target_width], method='nearest')
        bbox = tf.clip_by_value(bbox, 0., 1.) 

        if seed is None:
            seed = tf.random.uniform([2], maxval=2**31-1, dtype=tf.int32)
        seeds = tf.random.experimental.stateless_split(seed, 3)

        image, bbox, mask = self.random_flip_horizontal(image, bbox, mask, seeds[0])

        image = tf.cond(tf.random.stateless_uniform([], seed=seeds[1]) < self.prob_photo,
                        lambda: self.photometric(image, seeds[1]),
                        lambda: image)

        image, mask = tf.cond(tf.random.stateless_uniform([], seed=seeds[2]) < self.prob_cutout,
                              lambda: self.random_cutout(image, mask, seeds[2]),
                              lambda: (image, mask))

        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        bbox = tf.cast(bbox, tf.float32)

        output = {
            'image': image,
            'head_bbox': bbox,
            'segmentation_mask': mask,
            'label': tf.cast(processed['label'], tf.int64),
            'species': tf.cast(processed['species'], tf.int64),
            'file_name': processed['file_name'],
        }
        return output

    def get_output_signature(self) -> Dict[str, tf.TensorSpec]:
        return {
            'image': tf.TensorSpec(shape=(self.target_height, self.target_width, 3), dtype=tf.float32),
            'head_bbox': tf.TensorSpec(shape=(4,), dtype=tf.float32),
            'segmentation_mask': tf.TensorSpec(shape=(self.target_height, self.target_width, 1), dtype=tf.float32),
            'label': tf.TensorSpec(shape=(), dtype=tf.int64),
            'species': tf.TensorSpec(shape=(), dtype=tf.int64),
            'file_name': tf.TensorSpec(shape=(), dtype=tf.string),
        }

    def __call__(self, sample: Dict[str, tf.Tensor], seed: Optional[int] = None) -> Dict[str, tf.Tensor]:
        return self.augment_sample(sample, seed)

    def augment_dataset(self, dataset: tf.data.Dataset, seed: Optional[int] = None) -> tf.data.Dataset:
        def aug_fn(sample):
            return self.augment_sample(sample)
        return dataset.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

