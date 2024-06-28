import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import tensorflow as tf
import segmentation_models as sm
from containers import Dataset, DataLoader
from preprocessing import get_preprocessing, get_training_augmentation, get_validation_augmentation

IMG_DIR = "airbus-ship-detection/train_v2"
TRAIN_CSV_DIR = "unet_data/train.csv"
VAL_CSV_DIR = "unet_data/val.csv"

df_train = pd.read_csv(TRAIN_CSV_DIR); df_train.rename(columns={'EncodedPixels': 'Label'}, inplace=True)
df_val = pd.read_csv(VAL_CSV_DIR); df_train.rename(columns={'EncodedPixels': 'Label'}, inplace=True)

BACKBONE = "resnet50"
BATCH_SIZE = 22
LR = 1e-4
EPOCHS = 10

preprocessing_input = sm.get_preprocessing(BACKBONE)
preprocessing_fn = get_preprocessing(preprocessing_input)

train_dataset = Dataset(IMG_DIR, df_train, preprocessing_fn=preprocessing_fn, mask_mode=True)
val_dataset = Dataset(IMG_DIR, df_val, preprocessing_fn=preprocessing_fn, mask_mode=True)

train_loader1 = DataLoader(train_dataset, BATCH_SIZE, augmentations=get_training_augmentation())
val_loader1 = DataLoader(val_dataset, BATCH_SIZE, augmentations=get_validation_augmentation())

train_loader2 = DataLoader(train_dataset, BATCH_SIZE)
val_loader2 = DataLoader(val_dataset, BATCH_SIZE)

n_classes = 1
activation = 'sigmoid'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

optim = tf.keras.optimizers.Adam(LR)
dice_loss = sm.losses.DiceLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
model.compile(optim, dice_loss, metrics)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./weights/unet50.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3)
]

history = model.fit(
    train_loader1, 
    steps_per_epoch=len(train_loader1), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_loader1, 
    validation_steps=len(val_loader1)
)

history = model.fit(
    train_loader2, 
    steps_per_epoch=len(train_loader2), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=val_loader2, 
    validation_steps=len(val_loader2)
)