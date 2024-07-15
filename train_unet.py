import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log level to remove extra warnings

import time
import pandas as pd
import tensorflow as tf
from data import DataUtils
import segmentation_models as sm

data_dir = "airbus-ships/ships"  # Image directory

# Loading ship rle encodings
df = pd.read_csv("unet_data/encodings.csv")
encodings = df["EncodedPixels"].tolist()


imgs_ds = tf.data.Dataset.list_files(f"{data_dir}/*.jpg", shuffle=False)   # Loading image directory
masks_ds = tf.data.Dataset.from_tensor_slices(encodings)   # Encodings used to create binary masks
train_ds = tf.data.Dataset.zip((imgs_ds, masks_ds))   # Zipping images and their encodings
train_ds, val_ds = DataUtils.split_data(train_ds, 0.8)   # Splitting data


### Hyperparameters ###
EPOCHS = 20
LR = 6e-5
BATCH_SIZE = 16
BACKBONE = "resnet50"
ACTIVATION = "sigmoid"
N_CLASSES = 1
preprocessing_fn = sm.get_preprocessing(BACKBONE)  # Preprocessing function for feature extraction in U-Net model

# Preparing datasets
train_ds = DataUtils.prepare_ds(train_ds, BATCH_SIZE, preprocessing_fn)
val_ds = DataUtils.prepare_ds(val_ds, BATCH_SIZE, preprocessing_fn)

# Loaing model
model = sm.Unet(BACKBONE, classes=N_CLASSES, activation=ACTIVATION)

# Model compilation
optimizer = tf.keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
metrics = sm.metrics.FScore(threshold=0.5)
model.compile(optimizer, dice_loss, metrics)

# Setting callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./weights/test.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    tf.keras.callbacks.TensorBoard(log_dir="unet_logs", histogram_freq=1)
]


start = time.perf_counter()

# Training
history = model.fit(
    train_ds,
    steps_per_epoch=len(train_ds), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_ds, 
    validation_steps=len(val_ds)
)

passed = time.perf_counter() - start
print(f"\n\nTime passed: {passed}s")

# val_loss: 0.7236 - val_f1-score: 0.3246: 1e-4
# val_loss: 0.8736 - val_f1-score: 0.3736: 6e-5