import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log level to remove extra warnings

import tensorflow as tf
from data import DataUtils
from tensorflow import keras
import segmentation_models as sm

IMG_DIR = "resnet_imgs"
CSV_DIR = "resnet50_data/resnet_data.csv"

### Hyperparameters ###
LR = 7e-5
EPOCHS = 5
N_CLASSES = 1
BATCH_SIZE = 8
ACTIVATION = "sigmoid"
preprocessing_fn = sm.get_preprocessing("resnet50")

ds = DataUtils.load_data(IMG_DIR, CSV_DIR) # Loading data
train_ds, val_ds = DataUtils.split_data(ds, 0.8)  # Splitting data

# Preparing data
train_ds = DataUtils.prepare_ds(train_ds, BATCH_SIZE, preprocessing_fn)
val_ds = DataUtils.prepare_ds(val_ds, BATCH_SIZE, preprocessing_fn)

# Loading model
resnet50 = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = keras.Sequential([resnet50, keras.layers.Flatten(), keras.layers.Dense(N_CLASSES, activation=ACTIVATION)])

# Compiling model
optmizer = keras.optimizers.Adam(LR)
model.compile(optimizer=optmizer, loss="binary_crossentropy", metrics=['accuracy'])

# Setting callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./weights/r.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
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