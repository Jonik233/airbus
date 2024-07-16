import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Setting log level to remove extra warnings

import argparse
import numpy as np
from data import DataUtils
from keras import Sequential
import matplotlib.pyplot as plt
import segmentation_models as sm
from typing import Tuple, Callable
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from keras.engine.functional import Functional


def get_preprocessing_fn(model_title:str) -> Callable:
    preprocessing_fn = sm.get_preprocessing(model_title)
    return preprocessing_fn
    
    
def load_models(n_classes:int=1, activation:str='sigmoid') -> Tuple:
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model1 = Sequential([resnet50, Flatten(), Dense(n_classes, activation=activation)])
    model1.load_weights("weights/resnet50.h5")

    model2 = sm.Unet("resnet50", classes=n_classes, activation=activation)
    model2.load_weights("weights/unet.h5")

    return (model1, model2)


def predict(image:np.ndarray, model1:Sequential, model2:Functional) -> np.ndarray:
    presence_prob = model1.predict(image)[0][0]
    ship_is_present = presence_prob > 0.5
    prediction = model2.predict(image).squeeze() if ship_is_present else np.zeros((image.shape[1], image.shape[2]))
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference function")
    parser.add_argument("img_path", type=str, help="Image file path")
    args = parser.parse_args()
    
    img_path = args.img_path
    model1, model2 = load_models()
    preprocessing_fn = get_preprocessing_fn("resnet50")
    
    img = DataUtils.prepare_sample(img_path, preprocessing_fn).numpy()
    
    prediction = predict(img, model1, model2)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img[0])
    axs[0].set_title("Ground truth")
    axs[0].axis("off")
    axs[1].imshow(prediction)
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    plt.show()