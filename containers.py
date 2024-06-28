import os
import cv2
import random
import numpy as np
import tensorflow as tf
from data_prep import create_mask


class Dataset:
    def __init__(self, img_dir, df, preprocessing_fn=None, mask_mode=False):
        self.df = df
        self.img_dir = img_dir
        self.mask_mode = mask_mode
        self.preprocessing_fn = preprocessing_fn
    
    def __getitem__(self, i):
        item = self.df.loc[i]
        label = create_mask(self.df, item["ImageId"]) if self.mask_mode else item["Label"]
        image = cv2.imread(os.path.join(self.img_dir, item["ImageId"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.preprocessing_fn:
            preprocessed_unit = self.preprocessing_fn(image=image)
            image = preprocessed_unit["image"]
            
        return image, label
    
    def __len__(self):
        return self.df.shape[0]


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, augmentations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.augmentations = augmentations
        
    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        
        for j in range(start, stop):
            image, label = self.dataset[j]
            if self.augmentations:
                aug_unit = self.augmentations(image=image, label=label if self.dataset.mask_mode else None)
                image, label = aug_unit.get("image", image), aug_unit.get("label", label)
                
            data.append((image, label))

        random.shuffle(data)
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch
    
    def __len__(self):
        return len(self.dataset) // self.batch_size