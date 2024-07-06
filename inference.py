import os     
import cv2
import argparse
import numpy as np
import albumentations
import tensorflow as tf
from keras import Sequential
import matplotlib.pyplot as plt
import segmentation_models as sm
from keras.layers import Dense, Flatten
from keras.applications import ResNet50
from preprocessing import get_preprocessing
from keras.engine.functional import Functional
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def dice_c(pd: np.array, gt: np.array):
    pd = tf.where(pd >= 0.5, 1.0, 0.0).numpy()
    intersection = np.logical_and(pd, gt).sum()
    union = pd.sum() + gt.sum()
    dice = (2 * intersection) / (union + 1e-8)
    return dice


def iou_c(pd: np.array, gt: np.array):
    pd = tf.where(pd >= 0.5, 1.0, 0.0).numpy()
    intersection = np.logical_and(pd, gt).sum()
    union = np.logical_or(pd, gt).sum()
    iou = intersection / (union + 1e-8)
    return iou


def load_models(n_classes:int=1, activation:str='sigmoid') -> tuple:
    BACKBONE = "resnet50"
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing_fn = get_preprocessing(preprocess_input)
    
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model1 = Sequential([resnet50, Flatten(), Dense(n_classes, activation=activation)])
    model1.load_weights("weights/resnet50.h5")

    model2 = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    model2.load_weights("weights/unet50.h5")

    return (preprocessing_fn, model1, model2)


def predict(image:np.ndarray, model1:Sequential, model2:Functional) -> np.ndarray:
    presence_prob = model1.predict(image)[0][0]
    ship_is_present = presence_prob > 0.5
    prediction = model2.predict(image).squeeze() if ship_is_present else np.zeros((image.shape[1], image.shape[2]))
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference function")
    parser.add_argument("img_path", type=str, help="Image file path")
    args = parser.parse_args()
    
    bgr_image = cv2.imread(args.img_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    
    preprocessing_fn, model1, model2 = load_models()
    prediction = predict(args.img_path, model1, model2, preprocessing_fn)
    
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(rgb_image)
    axs[0].set_title("Ground truth")
    axs[0].axis("off")
    axs[1].imshow(prediction)
    axs[1].set_title("Prediction")
    axs[1].axis("off")
    plt.show()