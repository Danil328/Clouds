import argparse
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import read_config, rle2mask


def parse_args():
    parser = argparse.ArgumentParser(description="Create mask for training")
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()

mapping = {'Fish':0, 'Flower':1, 'Gravel':2, 'Sugar':3}

if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config_file, stage="MAIN")

    if not os.path.exists(os.path.join(config['path_to_data'], 'train_masks')):
        os.makedirs(os.path.join(config['path_to_data'], 'train_masks'))

    df = pd.read_csv(os.path.join(config['path_to_data'], 'train.csv'))
    df['ImageId'] = df['Image_Label'].map(lambda x: x.split("_")[0])
    df['ClassId'] = df[['Image_Label', 'EncodedPixels']].apply(
        lambda x: x[0].split("_")[1] if x[1] is not np.nan else '0', axis=1)
    df.drop(columns='Image_Label', inplace=True)
    df.drop_duplicates(inplace=True)

    # drop row with label 0 which exist label != 0
    max_class_df = df.groupby("ImageId")['ClassId'].max().reset_index()
    max_class_df = max_class_df[max_class_df['ClassId'] != '0']
    df = df[~((df['ImageId'].isin(max_class_df['ImageId'].values)) & (df['ClassId'] == '0'))]
    print(df.columns)
    images = df['ImageId'].unique()
    for image in tqdm(images):
        temp_df = df[df['ImageId'] == image]
        mask = np.zeros((config['n_classes'], config['height'], config['width']), dtype=np.uint8)
        for row in temp_df.values:
            if row[0] is not np.nan:
                mask[mapping[row[2]]] = rle2mask(row[0], shape=(config['width'], config['height']))
        mask = np.moveaxis(mask, 0, -1)
        cv2.imwrite(os.path.join(config['path_to_data'], 'train_masks', image.replace('.jpg', '.png')), mask)

