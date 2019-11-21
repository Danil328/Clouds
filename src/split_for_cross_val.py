import argparse
import glob
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm

from utils import read_config
from sklearn.model_selection import StratifiedKFold, train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Create mask for training")
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--n-folds", default=5, type=int)
    return parser.parse_args()


def get_stage(x, all_train_mask, train_masks_split):
    if x in all_train_mask:
        if x in train_masks_split:
            return 'train'
        else:
            return 'val'
    else:
        return 'holdout'


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config_file, stage="MAIN")

    masks = glob.glob(os.path.join(config['path_to_data'], "train_masks", "*.png"))
    labels = [cv2.imread(i, cv2.IMREAD_UNCHANGED).reshape(-1,4).max(0) for i in tqdm(masks)]
    labels = (np.vstack(labels)>0).astype(int)
    labels = labels * np.array((1,2,4,8))
    labels = labels.sum(1)

    masks = list(map(lambda x: x.replace('train_masks', 'train_images').replace('.png', '.jpg'), masks))
    train_mask, test_mask, train_labels, test_labels = train_test_split(masks, labels, test_size=0.1, stratify=labels)

    skf = StratifiedKFold(n_splits=args.n_folds)

    cv_df = pd.DataFrame()
    cv_df['images'] = masks
    for i, (train_index, test_index) in enumerate(skf.split(train_mask, train_labels)):
        train_mask_split = [train_mask[i] for i in train_index]
        cv_df[f'fold_{i}'] = cv_df['images'].map(lambda x: get_stage(x, train_mask, train_mask_split))

    cv_df.to_csv(os.path.join(config['path_to_data'], 'cross_val_DF.csv'), index=False)