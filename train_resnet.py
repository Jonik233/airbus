import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log level to remove extra warnings

from utils import Config
from data import DataUtils
from tensorflow import keras
from dotenv import load_dotenv
import segmentation_models as sm

load_dotenv()

# Data directories
IMG_DIR = os.getenv("RESNET_IMG_DIR")
CSV_DIR = os.getenv("RESNET_CSV_DIR")
CONFIG_DIR = os.getenv("RESNET_CONFIG_DIR")

#Loading configuration
config = Config(CONFIG_DIR).load()

### Hyperparameters ###
LR = config["learning_rate"]
EPOCHS = config["epochs"]
NUM_CLASSES = config["num_classes"]
BATCH_SIZE = config["batch_size"]
ACTIVATION = config["activation"]
preprocessing_fn = sm.get_preprocessing(config["preprocessing"])

ds = DataUtils.load_data(IMG_DIR, CSV_DIR) # Loading data
train_ds, val_ds = DataUtils.split_data(ds, 0.8)  # Splitting data

# Preparing data
train_ds = DataUtils.prepare_ds(train_ds, BATCH_SIZE, preprocessing_fn)
val_ds = DataUtils.prepare_ds(val_ds, BATCH_SIZE, preprocessing_fn)

# Loading model
resnet50 = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = keras.Sequential([resnet50, keras.layers.Flatten(), keras.layers.Dense(NUM_CLASSES, activation=ACTIVATION)])
model.load_weights(config["weights_path"])

# Compiling model
optmizer = keras.optimizers.Adam(LR)
model.compile(optimizer=optmizer, loss="binary_crossentropy", metrics=['accuracy'])

# Setting callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(config["weights_path"], save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(patience=3)
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