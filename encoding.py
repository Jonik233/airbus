import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from inference import predict, load_models
from skimage.measure import label, regionprops


ROOT = "airbus-ship-detection\\test_v2"
list_dir = os.listdir(ROOT)
df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
preprocessing_fn, model1, model2 = load_models()

for i, img_title in enumerate(list_dir):

    img_path = os.path.join(ROOT, img_title)
    pd_mask = predict(img_path, model1, model2, preprocessing_fn)
    pd_mask = tf.where(pd_mask >= 0.5, 1.0, 0.0).numpy()
    print(f"Encoding image {i + 1}/{len(list_dir)}, title: {img_title}")

    #ship encoding
    labeled_mask = label(pd_mask)
    regions = regionprops(labeled_mask)

    if len(regions) == 0:
        sample = {"ImageId":img_title, "EncodedPixels": ""}
        df = df.append(sample, ignore_index=True)
        continue

    for region in regions:
        # Extract the RLE encoding for the current ship
        y0, x0, y1, x1 = region.bbox
        ship_mask = labeled_mask[y0:y1, x0:x1] == region.label
        full_mask = np.zeros_like(pd_mask)
        full_mask[y0:y1, x0:x1] = ship_mask
        pixels = np.concatenate([[0], full_mask.flatten(), [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

        # Convert RLE encoding to string
        ship_rle = ' '.join(str(x) for x in runs)

        # Store the RLE encoding for the current ship
        sample = {"ImageId":img_title, "EncodedPixels": ship_rle}
        df = df.append(sample, ignore_index=True)


df.to_csv("submission.csv", index=False)