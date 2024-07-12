import tensorflow as tf
from utils import rle_to_mask


class Augment(tf.keras.layers.Layer):
    
  def __init__(self, seed=42):
    super().__init__()
    self.seed = (seed, seed)
    self.augmentations = [tf.image.stateless_random_flip_up_down, 
                          tf.image.stateless_random_flip_left_right]

  def call(self, inputs, labels):
    if tf.random.uniform(shape=(1,), minval=0, maxval=1) < 0.35:
        for fn in self.augmentations:
            inputs = fn(inputs, seed=self.seed)
            labels = fn(labels, seed=self.seed)
    
    return inputs, labels



class Preprocess(tf.keras.layers.Layer):
    
    def __init__(self, preprocessing_fn):
        super().__init__()
        self.preprocessing_fn = preprocessing_fn

    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        img = tf.image.resize(img, [224, 224], method=tf.image.ResizeMethod.BILINEAR)
        # Normalize the image to the range [0, 1]
        img = tf.cast(img, tf.float32) / 255.0
        img = self.preprocessing_fn(img)
        return img

    def get_mask(self, encoding):
  
        def _mask(encoding):
            encoding = encoding.numpy().decode("utf-8")
            mask = rle_to_mask(encoding, (768, 768))
            return mask
    
        mask = tf.py_function(_mask, [encoding], tf.float32)
        mask.set_shape([768, 768])
        mask = tf.expand_dims(mask, axis=-1)
        return tf.image.resize(mask, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def __call__(self, file_paths, encodings):
        imgs = tf.map_fn(tf.io.read_file, file_paths, fn_output_signature=tf.string)
        imgs = tf.map_fn(self.decode_img, imgs, fn_output_signature=tf.float32)
        masks = tf.map_fn(self.get_mask, encodings, fn_output_signature=tf.float32)
        return imgs, masks
    


class DataUtils:
    
    @staticmethod
    def split_data(ds:tf.data.Dataset, train_percentage:float):        
        num_samples = len(ds)
        train_size = int(train_percentage * num_samples)
        shuffled_ds = ds.shuffle(buffer_size=num_samples, reshuffle_each_iteration=False)
        train_ds = shuffled_ds.take(train_size)
        val_ds = shuffled_ds.skip(train_size)
        return train_ds, val_ds
    
    @staticmethod
    def prepare_ds(ds:tf.data.Dataset, batch_size:int, preprocessing_fn):
        ds = ds.batch(batch_size)
        ds = ds.map(Preprocess(preprocessing_fn), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(Augment(), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache("cache").prefetch(tf.data.AUTOTUNE)
        return ds