import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
from containers import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from preprocessing import get_preprocessing

IMG_DIR = "airbus-ship-detection/train_v2"
TRAIN_CSV_DIR = "resnet50_data/train.csv"
VAL_CSV_DIR = "resnet50_data/val.csv"

BATCH_SIZE = 38
LR = 6e-5
EPOCHS = 10

df_train = pd.read_csv(TRAIN_CSV_DIR)
df_val = pd.read_csv(VAL_CSV_DIR)

df_train["Label"] = None
df_val["Label"] = None

train_indxs = df_train["EncodedPixels"].isna()
val_indxs = df_val["EncodedPixels"].isna()

df_train.loc[train_indxs, "Label"] = 0
df_train.loc[~train_indxs, "Label"] = 1

df_val.loc[val_indxs, "Label"] = 0
df_val.loc[~val_indxs, "Label"] = 1

df_train.drop("EncodedPixels", axis=1, inplace=True)
df_val.drop("EncodedPixels", axis=1, inplace=True)

preprocessing_input = lambda image, cols, rows: preprocess_input(image)
preprocessing_fn = get_preprocessing(preprocessing_input)
train_dataset = Dataset(IMG_DIR, df_train, preprocessing_fn)
val_dataset = Dataset(IMG_DIR, df_val, preprocessing_fn)

train_loader = DataLoader(train_dataset, BATCH_SIZE)
val_loader = DataLoader(val_dataset, BATCH_SIZE)

n_classes = 1
activation = "sigmoid"
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([resnet50, Flatten(), Dense(n_classes, activation=activation)])
model.compile(optimizer=Adam(learning_rate=LR), loss="binary_crossentropy", metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./weights/resnet50.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
]

history = model.fit(
    train_loader, 
    steps_per_epoch=len(train_loader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_loader, 
    validation_steps=len(val_loader)
)