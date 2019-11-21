import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    RandomBrightnessContrast,
    CLAHE,
    Normalize,
    ShiftScaleRotate,
    CropNonEmptyMaskIfExists, RandomResizedCrop, Resize, ImageCompression, RandomGamma)
from albumentations.pytorch import ToTensor, ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm

ORIG_SHAPE = (1400, 2100)
TRAIN_SHAPE = (352, 512)
NUM_CLASSES = 4

bad_images = [
    '046586a', '1588d4c', '1e40a05', '41f92e5', '449b792', '563fc48', '8bd81ce', 'b092cc1',
    'c0306e5', 'c26c635', 'e04fea3', 'e5f2f24', 'eda52f2', 'fa645da'
]

AUGMENTATIONS_TRAIN = Compose([
    RandomRotate90(p=0.25),
    RandomResizedCrop(height=TRAIN_SHAPE[0], width=TRAIN_SHAPE[1], scale=(0.8,1.0)),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-20, 20), border_mode=0, interpolation=1, p=0.25),
    OneOf([
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        RandomGamma(),
        CLAHE()
    ], p=0.4),
    OneOf([
        ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(border_mode=0, distort_limit=0.05, interpolation=1, shift_limit=0.05, p=1.0),
    ], p=0.25),
    ImageCompression(quality_lower=50, p=0.5),
    Normalize(),
    ToTensor(num_classes=NUM_CLASSES, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST = Compose([
    Resize(height=TRAIN_SHAPE[0], width=TRAIN_SHAPE[1], always_apply=True),
    Normalize(),
    ToTensor(num_classes=NUM_CLASSES, sigmoid=True)
], p=1)


AUGMENTATIONS_TRAIN_CROP = Compose([
    CropNonEmptyMaskIfExists(height=256, width=448, always_apply=True),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.1),
    ShiftScaleRotate(shift_limit=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-10, 10), border_mode=0, interpolation=1, p=0.25),
    OneOf([
        ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.03),
        GridDistortion(p=0.5),
        OpticalDistortion(border_mode=0, distort_limit=0.05, interpolation=1, shift_limit=0.05, p=1.0),
    ], p=0.1),
    OneOf([
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    ], p=0.25),
    Normalize(),
    ToTensor(num_classes=NUM_CLASSES, sigmoid=True)
], p=1)

AUGMENTATIONS_TEST_CROP = Compose([
    CropNonEmptyMaskIfExists(height=256, width=448, always_apply=True),
    Normalize(),
    ToTensor(num_classes=NUM_CLASSES, sigmoid=True)
], p=1)

class SegmentationDataset(Dataset):

    def __init__(self, data_folder, transforms, phase, fold=-1, empty_mask_params: dict = None, activation="sigmoid"):
        assert phase in ['train', 'val', 'test', 'holdout'], "Fuck you!"

        self.root = data_folder
        self.transforms = transforms
        self.phase = phase
        self.fold = fold
        self.activation = activation
        if phase != 'test':
            self.images = np.asarray(self.split_train_val(glob.glob(os.path.join(self.root, "train_images", "*.jpg"))))#[:200]

            # Get labels for classification
            self.labels = np.zeros((self.images.shape[0], 4), dtype=np.float32)
            for idx, image_name in enumerate(tqdm(self.images)):
                image = cv2.imread(image_name.replace('train_images', 'train_masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
                self.labels[idx] = (np.amax(image, axis=(0, 1)) > 0).astype(float)

            self.empty_images = self.images[self.labels.max(axis=1) == 0]
            self.non_empty_images = self.images[self.labels.max(axis=1) == 1]
        else:
            self.images = glob.glob(os.path.join(self.root, "test_images", "*.jpg"))#[:200]

        if empty_mask_params is not None and empty_mask_params['state'] == 'true':
            self.start_value = empty_mask_params['start_value']
            self.delta = (empty_mask_params['end_value'] - empty_mask_params['start_value']) / empty_mask_params['n_epochs']
            self.positive_ratio = self.start_value
        else:
            self.positive_ratio = 1.0


    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        if self.phase != 'test':
            mask = cv2.imread(self.images[idx].replace('train_images', 'train_masks').replace('.jpg', '.png'), cv2.IMREAD_UNCHANGED)
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            if self.activation == 'softmax':
                #mask = torch.cat([mask, (1.0-mask.max(0).values).unsqueeze(0)], 0).type(torch.LongTensor)
                mask = torch.argmax(torch.cat([(1.0-mask.max(0).values).unsqueeze(0), mask], 0).type(torch.LongTensor), 0, True)
            return {"image": img, "mask": mask, "label": mask.max().detach(), "filename": self.images[idx].split("/")[-1]}
        else:
            augmented = self.transforms(image=img)
            img = augmented['image']
            return {"image": img, "filename": self.images[idx].split("/")[-1]}

    def __len__(self):
        return len(self.images)

    def split_train_val(self, images: list):
        if self.fold < 0:
            train, val = train_test_split(images, test_size=0.1, random_state=17)
            if self.phase == 'train':
                return train
            elif self.phase == 'val':
                return val
        else:
            cv_df = pd.read_csv(os.path.join(self.root, 'cross_val_DF.csv'))
            return cv_df[cv_df[f'fold_{self.fold}'] == self.phase]['images'].values

    def update_empty_mask_ratio(self, epoch: int):
        self.positive_ratio = self.start_value + self.delta * epoch
        self.images = np.hstack((self.non_empty_images, self.empty_images[:int(self.positive_ratio * self.empty_images.shape[0])]))


class FourBalanceClassSampler(Sampler):
    def __init__(self, labels):
        label = labels.reshape(-1,4)
        label = np.hstack([label.sum(1,keepdims=True)==0,label]).T

        self.neg_index  = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        #assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4*num_neg


    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg,pos1,pos2,pos3,pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    rnd = np.random.randint(1, 100)
    dataset = SegmentationDataset(data_folder='../data', transforms=AUGMENTATIONS_TRAIN, phase='train', activation='sigmoid',
                           empty_mask_params={"state": "true",
                                              "start_value": 0.0,
                                              "end_value": 1.0,
                                              "n_epochs": 50})

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True)


    data = dataset[rnd]
    image = data['image'].numpy()
    mask = data['mask'].numpy()
    print(image.shape)
    print(mask.shape)
    print(image.min(), image.max())
    print(mask.min(), mask.max())
    print(dataset.labels.sum(0))

    print(f"Len before update {dataset.__len__()}")
    dataset.update_empty_mask_ratio(10)
    print(f"Len after update {dataset.__len__()}")

    for i in train_loader:
        break

