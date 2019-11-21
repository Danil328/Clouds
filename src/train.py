# import time
# time.sleep(200*60)
import argparse
import os
import shutil
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset, AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST, AUGMENTATIONS_TRAIN_CROP, AUGMENTATIONS_TEST_CROP, FourBalanceClassSampler
from callbacks import Callbacks, CheckpointSaver, Logger, TensorBoard, FreezerCallback
from factory import Factory
from runner import Runner
from utils import read_config, set_global_seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def create_callbacks(name, dumps):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    save_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks(
        [
            Logger(log_dir),
            CheckpointSaver(
                metric_name=dumps['metric_name'],
                save_dir=save_dir,
                save_name='epoch_{epoch}.pth',
                num_checkpoints=4,
                mode='max'
            ),
            TensorBoard(str(log_dir)),
            FreezerCallback()
        ]
    )
    return callbacks


def main():
    args = parse_args()
    set_global_seeds(666)
    config = read_config(args.config, "TRAIN")
    config_main = read_config(args.config, "MAIN")
    pprint(config)
    factory = Factory(config['train_params'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    callbacks = create_callbacks(config['train_params']['name'], config['dumps'])
    trainer = Runner(stages=config['stages'], factory=factory, callbacks=callbacks, device=device)

    aug_train = AUGMENTATIONS_TRAIN_CROP if config['train_params']['type'] == 'crop' else AUGMENTATIONS_TRAIN
    aug_test = AUGMENTATIONS_TEST_CROP if config['train_params']['type'] == 'crop' else AUGMENTATIONS_TEST

    train_dataset = SegmentationDataset(data_folder=config_main['path_to_data'], transforms=aug_train, phase='train', activation=config_main['activation'],
                                 fold=config['fold'], empty_mask_params=config['data_params']['empty_mask_increase'])

    val_dataset = SegmentationDataset(data_folder=config_main['path_to_data'], transforms=aug_test, phase='val',
                               fold=config['fold'], activation=config_main['activation'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)

    os.makedirs(os.path.join(config['dumps']['path'], config['dumps']['weights'], config['train_params']['name']), exist_ok=True)
    shutil.copy(args.config, os.path.join(config['dumps']['path'], config['dumps']['weights'], config['train_params']['name'], args.config.split('/')[-1]))
    trainer.fit(train_loader, val_loader)


if __name__ == '__main__':
    main()