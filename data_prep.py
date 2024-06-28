import os
import shutil
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_img_dir(df:pd.DataFrame, original_dir:str, new_dir:str) -> None:
    errors = 0
    os.makedirs(new_dir, exist_ok=True)
    
    for file_name in df['ImageId']:
        source_file = os.path.join(original_dir, file_name)
        destination_file = os.path.join(new_dir, file_name)

        # Check if the source file exists and then copy it
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        else:
            errors += 1
            print(f"File not found: {source_file}")

    if errors == 0:
        print("~"*30)
        print("FILES COPIED SUCCESSFULLY")
        print("~"*30)
    else:
        print("~"*30)
        print(f"FILES COPIED WITH {errors} errors")
        print("~"*30)


def rle_to_mask(rle_string:str, shape:tuple) -> np.array:
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to the original image


def mask_to_rle(mask:np.array) -> list:
    rle = []
    start_index = None
    current_pixel = 0
    
    for i, pixel in enumerate(mask.flatten()):
        if pixel != current_pixel:
            current_pixel = pixel
            if current_pixel == 0:
                rle.append((start_index, i - start_index))
            else:
                start_index = i
        
    if current_pixel == 1:
        rle.append((start_index, len(mask.flatten()) - start_index))
    
    return rle


def _combine_masks(rle_list:list, shape:tuple) -> np.array:
    # Combine multiple RLEs into one mask
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for rle in rle_list:
        if isinstance(rle, str):  # Check if RLE is valid (not NaN or similar)
            mask = rle_to_mask(rle, shape)
            combined_mask = np.maximum(combined_mask, mask)
    return combined_mask


def create_mask(df:pd.DataFrame, file_name:str) -> np.array:

    df = df.set_index("ImageId")
    rle = df.loc[file_name, "Label"]
    if isinstance(rle, pd.Series): rle = rle.tolist()
    else: rle = [rle]
   
    mask = _combine_masks(rle, (224, 224)).astype(np.float32)
    return mask


def create_dataset(df_data:pd.DataFrame) -> pd.DataFrame:
    indxs = df_data["EncodedPixels"].isna()

    for i, title in enumerate(df_data[~indxs]["ImageId"].unique()):
        ids = df_data["ImageId"] == title
        items = df_data[ids]
        if len(items) > 1:
            print(i)
            encoding = items["EncodedPixels"].str.cat(sep=' ')
            item = {"ImageId":title, "EncodedPixels":encoding}
            df_data = df_data[~ids]
            df_data = df_data.append(item, ignore_index=True)
        
        return df_data
        
        
def split_data(new_data_path:str, rle_path:str, blank_percentage:float) -> None:
    os.makedirs(new_data_path, exist_ok=True)
    
    df = pd.read_csv(rle_path)
    df_unique = df.drop_duplicates(subset='ImageId', keep='first')
    
    indxs_unique = df_unique["EncodedPixels"].isna()
    indxs = df["EncodedPixels"].isna()
    
    #splitting dataset into p% of blank images and (100 - p)% of representable images
    N_BLANK_SAMPLES = (df_unique[~indxs_unique].shape[0] * blank_percentage) / (100 - blank_percentage)
    N_BLANK_SAMPLES = int(N_BLANK_SAMPLES)
    
    imgs_with_ships = df[~indxs]
    imgs_without_ships = df_unique[indxs]
    imgs_without_ships = imgs_without_ships.sample(n=N_BLANK_SAMPLES, random_state=1)
    
    #creating final dataframe, combining encodings and removing duplicates
    df_final = pd.concat([imgs_without_ships, imgs_with_ships])
    df_final = df_final.groupby("ImageId")["EncodedPixels"].apply(lambda x: ' '.join(x.dropna())).reset_index()
    df_final["EncodedPixels"] = df_final["EncodedPixels"].replace('', np.nan)
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    
    #splitting data into datasets
    print("Splitting data...")
    df_train, df_val = train_test_split(df_final, test_size=0.2, random_state=42)
    df_train.to_csv(os.path.join(new_data_path, "train.csv"), index=False)
    df_val.to_csv(os.path.join(new_data_path, "val.csv"), index=False)
    
    print("--"*20)
    print("DATA SPLIT DONE")
    print("--"*20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits images into train, val and test datasets, datasets are represented in the form of directories")
    parser.add_argument("new_data_dir", type=str, help="New directory for data.")
    parser.add_argument("rle_path", type=str, help="Path to the csv file with rle encodings.")
    parser.add_argument("blank_percentage", type=float, help="Percentage of blank images.")

    args = parser.parse_args()
    split_data(args.new_data_dir, args.rle_path, args.blank_percentage)