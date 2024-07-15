import pandas as pd
import tensorflow as tf
from utils import rle_to_mask
import segmentation_models as sm
from typing import Callable, Tuple
from tensorflow import Tensor, data


# Augmentation layer of data pipeline
class Augment(tf.keras.layers.Layer):
        
    def __init__(self, masks:bool, seed:int=42, p:float=0.35):
        super().__init__()
        self.p = p
        self.masks = masks
        self.seed = (seed, seed)
        self.augmentations = [tf.image.stateless_random_flip_up_down, 
                            tf.image.stateless_random_flip_left_right]

    
    def _augment(self, inputs:Tensor) -> Tensor:
        for fn in self.augmentations:
            inputs = fn(inputs, seed=self.seed)

        return inputs
    
    
    def __call__(self, inputs:Tensor, targets:Tensor) -> Tuple[Tensor, Tensor]:
        """
            Applies augmentations to the inputs and masks tensors with a certain probability.

            Args:
                inputs (tf.Tensor): A tf.float32 tensor containing input images.
                targets (tf.Tensor): A tf.float32 tensor containing binary masks if present else: tf.int32.

            Returns:
                Tuple[tf.Tensor, tf.Tensor]: The possibly augmented inputs and masks tensors.
        """
        if tf.random.uniform(shape=(1,), minval=0, maxval=1) < self.p:
            inputs = self._augment(inputs)
            if self.masks: targets = self._augment(targets)
        
        return (inputs, targets)



# Preprocessing layer of data pipeline
class Preprocess(tf.keras.layers.Layer):
        
    def __init__(self, preprocessing_fn:Callable, masks:bool):
        super().__init__()
        self.masks = masks
        self.preprocessing_fn = preprocessing_fn


    def preprocess_img(self, img:Tensor) -> Tensor:
        try:
            img = tf.io.decode_jpeg(img, channels=3) # Convert the compressed string to a 3D uint8 tensor
            img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
            img = tf.cast(img, tf.float32) / 255.0 # Normalization
            img = self.preprocessing_fn(img)
            return img
        except tf.errors.InvalidArgumentError:
            print(f"Skipping corrupted image: {img}")
            

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


    def __call__(self, file_paths:Tensor, targets:Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes file paths and encodings tensors.

        Args:
            file_paths (tf.Tensor): A tensor of dtype tf.string containing file paths.
            targets (tf.Tensor): A tensor of dtype tf.string when containing rle encodings, else: tf.int32.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tensors of dtype=tf.float32
        """
        imgs = tf.map_fn(tf.io.read_file, file_paths, fn_output_signature=tf.string)
        imgs = tf.map_fn(self.preprocess_img, imgs, fn_output_signature=tf.float32)
        
        if self.masks: 
            targets = tf.map_fn(self.get_mask, targets, fn_output_signature=tf.float32)
        else:
            targets = tf.cast(targets, tf.float32)
            
        return (imgs, targets)



# Utils for data preparation
class DataUtils:
    
    @staticmethod
    def load_data(imgs_dir:str, csv_dir:str) -> data.Dataset:
        df = pd.read_csv(csv_dir)
        labels = df["Label"].tolist()
        imgs_ds = tf.data.Dataset.list_files(f"{imgs_dir}/*.jpg", shuffle=False)  # Loading image directory
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)  # Loading labels
        ds = tf.data.Dataset.zip((imgs_ds, labels_ds))  # Zipping images and their labels
        return ds
    
    
    @staticmethod
    def split_data(ds:data.Dataset, train_percentage:float) -> Tuple[data.Dataset, data.Dataset]:
        num_samples = len(ds)
        train_size = int(train_percentage * num_samples)
        shuffled_ds = ds.shuffle(buffer_size=num_samples, reshuffle_each_iteration=False)
        train_ds = shuffled_ds.take(train_size)
        val_ds = shuffled_ds.skip(train_size)
        return (train_ds, val_ds)


    @staticmethod
    def prepare_ds(ds:data.Dataset, batch_size:int, preprocessing_fn:Callable, masks:bool=False) -> data.Dataset:
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(Preprocess(preprocessing_fn, masks), num_parallel_calls=data.AUTOTUNE)  # Parallel preprocessing
        ds = ds.map(Augment(masks), num_parallel_calls=data.AUTOTUNE)  # Parallel augmentation
        ds = ds.cache("cache").prefetch(data.AUTOTUNE)  # Reducing step time
        return ds
    
    
    @staticmethod
    def prepare_sample(img_path:str, preprocessing_fn:Callable) -> Tensor:
        preprocess = Preprocess(preprocessing_fn, False)
        encoded_img = tf.io.read_file(img_path)
        img = preprocess.preprocess_img(encoded_img)
        img = tf.expand_dims(img, axis=0)
        return img