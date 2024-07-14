from typing import Callable, Tuple 
import tensorflow as tf
from tensorflow import Tensor, data
from utils import rle_to_mask


# Augmentation layer of data pipeline
class Augment(tf.keras.layers.Layer):
    
  def __init__(self, seed:int=42, p:float=0.35):
    super().__init__()
    self.p = p
    self.seed = (seed, seed)
    self.augmentations = [tf.image.stateless_random_flip_up_down, 
                          tf.image.stateless_random_flip_left_right]


  def call(self, inputs:Tensor, masks:Tensor) -> Tuple[Tensor, Tensor]:
    """
        Applies augmentations to the inputs and masks tensors with a certain probability.

        Args:
            inputs (tf.Tensor): A tf.float32 tensor containing input images.
            msks (tf.Tensor): A tf.float32 tensor containing binary masks.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The possibly augmented inputs and masks tensors.
    """
    if tf.random.uniform(shape=(1,), minval=0, maxval=1) < self.p:
        for fn in self.augmentations:
            inputs = fn(inputs, seed=self.seed)
            masks = fn(masks, seed=self.seed)
    
    return (inputs, masks)



# Preprocessing layer of data pipeline
class Preprocess(tf.keras.layers.Layer):
    
    def __init__(self, preprocessing_fn:Callable):
        super().__init__()
        self.preprocessing_fn = preprocessing_fn


    def preprocess_img(self, img:Tensor) -> Tensor:
        img = tf.io.decode_jpeg(img, channels=3) # Convert the compressed string to a 3D uint8 tensor
        img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0 # Normalization
        img = self.preprocessing_fn(img)
        return img


    def get_mask(self, encoding:Tensor) -> Tensor:
        
        # Function for eager execution
        def _mask(encoding):
            encoding = encoding.numpy().decode("utf-8")
            mask = rle_to_mask(encoding, (768, 768))
            return mask
    
        mask = tf.py_function(_mask, [encoding], tf.float32) # Incorporating mask function for eager use in TensorFlow's computation graph
        mask.set_shape([768, 768])
        mask = tf.expand_dims(mask, axis=-1)
        return tf.image.resize(mask, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    def __call__(self, file_paths:Tensor, encodings:Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes file paths and encodings tensors.

        Args:
            file_paths (tf.Tensor): A tensor of dtype tf.string containing file paths.
            encodings (tf.Tensor): A tensor of dtype tf.string containing encodings.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Two tensors of dtype tf.float32.
        """
        imgs = tf.map_fn(tf.io.read_file, file_paths, fn_output_signature=tf.string)
        imgs = tf.map_fn(self.preprocess_img, imgs, fn_output_signature=tf.float32)
        masks = tf.map_fn(self.get_mask, encodings, fn_output_signature=tf.float32)
        return (imgs, masks)



# Utils for data preparation
class DataUtils:
    
    @staticmethod
    def split_data(ds:data.Dataset, train_percentage:float) -> Tuple[data.Dataset, data.Dataset]:
        num_samples = len(ds)
        train_size = int(train_percentage * num_samples)
        shuffled_ds = ds.shuffle(buffer_size=num_samples, reshuffle_each_iteration=False)
        train_ds = shuffled_ds.take(train_size)
        val_ds = shuffled_ds.skip(train_size)
        return (train_ds, val_ds)
    
    
    @staticmethod
    def prepare_ds(ds:data.Dataset, batch_size:int, preprocessing_fn:Callable) -> data.Dataset:
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(Preprocess(preprocessing_fn), num_parallel_calls=tf.data.AUTOTUNE) # Parallel preprocessing
        ds = ds.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE) # Parallel augmentation
        ds = ds.cache("cache").prefetch(tf.data.AUTOTUNE) # Reducing step time
        return ds