import os
import cv2
import time
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import albumentations
import matplotlib.pyplot as plt
       
        
def rle_to_mask(rle_string, shape):
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    return img.reshape(shape).T


def combine_masks(rle_list, shape):
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        if isinstance(rle, str):  # Check if RLE is valid (not NaN or similar)
            mask = rle_to_mask(rle, shape)
            combined_mask = np.maximum(combined_mask, mask)
            
    return combined_mask


def create_masks(df, img_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)
    df = df.set_index("ImageId")
    for file_name in os.listdir(img_dir):
        rle = df.loc[file_name, 'EncodedPixels']
        if isinstance(rle, pd.Series):
            rle = rle.tolist()
        else:
            rle = [rle]
                
        mask = combine_masks(rle, (768, 768))
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_image_path = os.path.join(masks_dir, f"{os.path.splitext(file_name)[0]}.png")
        mask_image.save(mask_image_path)


def read_image(img_path:str, preprocessing_fn:albumentations=None) -> np.ndarray:
    if os.path.exists(img_path):
        bgr_image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = preprocessing_fn(image=rgb_image)["image"] if preprocessing_fn else rgb_image
        image = np.expand_dims(image, axis=0)
        return image
    else:
        raise FileNotFoundError(f"Could not find {img_path}")


def subplot(images:list, title:str="") -> None:
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(10, 7))
    for ax, img in zip(axes, images): ax.imshow(img); ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.subplots_adjust(top=1.4, bottom=0.1, left=0.05, right=0.95, hspace=0.4, wspace=0.3)
    plt.show()


def rle_to_mask(rle_string:str, shape:tuple) -> np.ndarray:
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends): img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to the original image


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
      print(epoch_num+1)
      for i, (img, mask) in enumerate(dataset):
          # Performing a training step
          time.sleep(0.01)
          if i % 100 == 0: print(f"{i+1}/{len(dataset)}")
    print("Execution time:", time.perf_counter() - start_time)