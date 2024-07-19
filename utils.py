import os
import cv2
import time
import json
import numpy as np
import albumentations
from tensorflow import data
from typing import Dict, Any
import matplotlib.pyplot as plt


class Utils:
    
    # Heplper function for plotting
    @staticmethod
    def read_image(img_path:str, preprocessing_fn:albumentations=None) -> np.ndarray:
        if os.path.exists(img_path):
            bgr_image = cv2.imread(img_path)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            image = preprocessing_fn(image=rgb_image)["image"] if preprocessing_fn else rgb_image
            image = np.expand_dims(image, axis=0)
            return image
        else:
            raise FileNotFoundError(f"Could not find {img_path}")


    # Function for convinient plotting images 
    @staticmethod
    def subplot(images:list, title:str="") -> None:
        fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(10, 7))
        for ax, img in zip(axes, images): ax.imshow(img); ax.axis("off")
        fig.suptitle(title, fontsize=12)
        plt.subplots_adjust(top=1.4, bottom=0.1, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
        plt.show()


    # Converts rle encoding into binary mask of a given shape
    @staticmethod
    def rle_to_mask(rle_string:str, shape:tuple) -> np.ndarray:
        s = rle_string.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends): img[lo:hi] = 1
        return img.reshape(shape).T  # Needed to align to the original image


    # Used for tracking pipeline perfomance
    @staticmethod
    def benchmark(dataset:data.Dataset, num_epochs:int=2) -> None:
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            print(epoch_num+1)
            for i, (img, mask) in enumerate(dataset):
                # Performing a training step
                time.sleep(0.01)
                if i % 100 == 0: print(f"{i+1}/{len(dataset)}")
        print("Execution time:", time.perf_counter() - start_time)



class Config:
    def __init__(self, file_path:str) -> None:
        self.file_path = file_path
    
    
    def save(self, data:Dict[str, Any]) -> None:
        with open(self.file_path, "w") as f:
            json.dump(data, f, sort_keys=True)
            
    
    def load(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                
        except FileNotFoundError as e:
            print(f"{e.strerror}: {e.filename}")
            
        else:
            return data