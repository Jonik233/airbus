import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log level to remove extra warnings

import tensorflow as tf
from data import DataUtils
import segmentation_models as sm

IMG_DIR = "unet_imgs"
CSV_DIR = "unet_data\encodings.csv"

### Hyperparameters ###
LR = 6e-5
EPOCHS = 20
N_CLASSES = 1
BATCH_SIZE = 16
BACKBONE = "resnet50"
ACTIVATION = "sigmoid"
preprocessing_fn = sm.get_preprocessing(BACKBONE)

ds = DataUtils.load_data(IMG_DIR, CSV_DIR) # Loading data
train_ds, val_ds = DataUtils.split_data(ds, 0.8) # Splitting data

# Preparing datasets
train_ds = DataUtils.prepare_ds(train_ds, BATCH_SIZE, preprocessing_fn, masks=True)
val_ds = DataUtils.prepare_ds(val_ds, BATCH_SIZE, preprocessing_fn, masks=True)

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

# Training
history = model.fit(
    train_ds,
    steps_per_epoch=len(train_ds), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_ds, 
    validation_steps=len(val_ds)
)