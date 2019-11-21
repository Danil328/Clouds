import argparse
import glob
import os
import pydoc

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SegmentationDataset, AUGMENTATIONS_TEST, TRAIN_SHAPE
from make_masks import mapping
from metrics import dice_coef_numpy
from utils import read_config, mask2rle, CRF, optimize_trapezoid
import ttach as tta

TEST_SHAPE = (350, 525)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="../config.yaml", metavar="FILE", help="path to config file", type=str)
    return parser.parse_args()


def main():
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    transforms = tta.Compose([ tta.HorizontalFlip() ])

    #best_threshold, best_min_size_threshold = search_threshold(device, transforms)
    best_threshold = [0.8, 0.7, 0.8, 0.7]
    best_min_size_threshold = 0

    predict(best_threshold, best_min_size_threshold, device, transforms)


def search_threshold(device, transforms):
    val_dataset = SegmentationDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='val',
                               fold=config['fold'], activation=config_main['activation'])
    if len(config['cls_predict_val']) > 0:
        val_dataset.start_value = 0.1
        val_dataset.delta = 0.0
        val_dataset.update_empty_mask_ratio(0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    if len(config['cls_predict_val']) > 0:
        print("Use classification model results")
        cls_df = pd.read_csv(config['cls_predict_val'])
        cls_df['is_mask_empty'] = cls_df['label'].map(lambda x: 1 if x==0 else 0)
        cls_df.index = cls_df.Image_Label.values
        cls_df.drop_duplicates(inplace=True)
    else:
        cls_df = None

    models = []

    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'cosine/') + "*.pth"):
        model = pydoc.locate(config['model'])(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        models.append(model)
    print(f"Use {len(models)} models.")
    assert len(models) > 0, "Models not loaded"

    masks, predicts = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            fnames = batch["filename"]
            images = batch["image"].to(device)
            mask = batch['mask'].cpu().numpy()
            mask_pred_shape = np.zeros((images.size(0), 4, TEST_SHAPE[0], TEST_SHAPE[1]), dtype=np.float32)
            batch_preds = np.zeros((images.size(0), 4, TRAIN_SHAPE[0], TRAIN_SHAPE[1]), dtype=np.float32)
            batch_preds_test_shape = np.zeros((images.size(0), 4, TEST_SHAPE[0], TEST_SHAPE[1]), dtype=np.float32)
            if config['type'] == 'crop':
                for model in models:
                    if config['TTA'] == 'true':
                        model = tta.SegmentationTTAWrapper(model, transforms)
                    tmp_batch_preds = np.zeros((images.size(0), 4, TRAIN_SHAPE[0], TRAIN_SHAPE[1]), dtype=np.float32)
                    for step in np.arange(0, TRAIN_SHAPE[1], 384)[:-1]:
                        tmp_pred = torch.sigmoid(model(images[:,:,:,step:step+448])).cpu().numpy()
                        tmp_batch_preds[:,:,:,step:step+448] += tmp_pred
                    tmp_batch_preds[:,:,:,384:384+64] /= 2
                    tmp_batch_preds[:,:,:,2*384:2*384+64] /= 2
                    tmp_batch_preds[:,:,:,3*384:3*384+64] /= 2
                    batch_preds += tmp_batch_preds
            else:
                for model in models:
                    if config['TTA'] == 'true':
                        model = tta.SegmentationTTAWrapper(model, transforms)
                    batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)

            for i in range(batch_preds.shape[0]):
                tmp = cv2.resize(np.moveaxis(batch_preds[i], 0, -1), (TEST_SHAPE[1], TEST_SHAPE[0]))
                batch_preds_test_shape[i] = np.moveaxis(tmp, -1, 0)

                tmp = cv2.resize(np.moveaxis(mask[i], 0, -1), (TEST_SHAPE[1], TEST_SHAPE[0]))
                mask_pred_shape[i] = np.moveaxis(tmp, -1, 0)

            for num_file in range(batch_preds_test_shape.shape[0]):
                for cls in range(config_main['n_classes']):
                    if cls_df is not None:
                        if cls_df.loc[fnames[num_file] + f"_{inv_map[cls]}"]['is_mask_empty'] == 1:
                            batch_preds_test_shape[num_file, cls] = np.zeros((TEST_SHAPE[0], TEST_SHAPE[1]))

            predicts.append(batch_preds_test_shape)
            masks.append(mask_pred_shape)

    predicts = np.vstack(predicts)
    masks = np.vstack(masks)

    print("Search threshold ...")
    thresholds = np.arange(0.1, 1.0, 0.1)
    if config['channel_threshold'] == 'true':
        best_threshold = []
        for channel in range(4):
            scores = []
            for threshold in tqdm(thresholds):
                score = dice_coef_numpy(preds=(predicts>threshold).astype(int), trues=masks, channel=channel)
                print(f"{threshold} - {score}")
                scores.append(score)
            best_score = np.max(scores)
            print(f"Best threshold - {thresholds[np.argmax(scores)]}, best score - {best_score}")
            print(f"Scores: {scores}")
            best_threshold.append(thresholds[np.argmax(scores)])
        print(f"Best thresholds - {best_threshold}")
    else:
        scores = []
        for threshold in tqdm(thresholds):
            score = dice_coef_numpy(preds=(predicts > threshold).astype(int), trues=masks)
            print(f"{threshold} - {score}")
            scores.append(score)
        best_score = np.max(scores)
        best_threshold = thresholds[np.argmax(scores)]
        print(f"Best threshold - {best_threshold}, best score - {best_score}")
        print(f"Scores: {scores}")

    #best_threshold = [0.8, 0.8, 0.9, 0.7]
    print("Search min_size threshold ...")
    thresholds = np.arange(0, 20000, 1000)
    scores = []
    for threshold in tqdm(thresholds):
        tmp = predicts.copy()
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i,j] = post_process(tmp[i,j], best_threshold, threshold, j,
                                        use_dense_crf=config['use_dense_crf'], image=cv2.imread(val_dataset.images[i])  if config['use_dense_crf']=='true' else None,
                                        use_dilations=config['use_dilations'], use_poligonization=config['use_poligonization'])
        score = dice_coef_numpy(preds=tmp, trues=masks)
        print(f"{threshold} - {score}")
        scores.append(score)
    best_score = np.max(scores)
    best_min_size_threshold = thresholds[np.argmax(scores)]
    print(f"Best min_size threshold - {best_min_size_threshold}, best score - {best_score}")
    print(f"Scores: {scores}")

    return best_threshold, best_min_size_threshold


def predict(best_threshold, min_size, device, transforms):
    test_dataset = SegmentationDataset(data_folder=config_main['path_to_data'], transforms=AUGMENTATIONS_TEST, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16, drop_last=False)

    models = []
    for weight in glob.glob(os.path.join(config['weights'], config['name'], 'cosine/') + "*.pth"):
        model = pydoc.locate(config['model'])(**config['model_params'])
        model.load_state_dict(torch.load(weight))
        model = model.to(device)
        model.eval()
        models.append(model)

    if len(config['cls_predict_test']) > 0:
        print("Use classification model results")
        cls_df = pd.read_csv(config['cls_predict_test'])
        cls_df['is_mask_empty'] = cls_df['EncodedPixels'].map(lambda x: 1 if x==0 else 0)
        cls_df.index = cls_df.Image_Label.values
        cls_df.drop_duplicates(inplace=True)
    else:
        cls_df = None

    predictions = []
    image_names = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            fnames = batch["filename"]
            images = batch["image"].to(device)
            batch_preds = np.zeros((images.size(0), 4, TRAIN_SHAPE[0], TRAIN_SHAPE[1]), dtype=np.float32)
            batch_preds_test_shape = np.zeros((images.size(0), 4, TEST_SHAPE[0], TEST_SHAPE[1]), dtype=np.float32)
            if config['type'] == 'crop':
                for model in models:
                    if config['TTA'] == 'true':
                        model = tta.SegmentationTTAWrapper(model, transforms)
                    tmp_batch_preds = np.zeros((images.size(0), 4, TRAIN_SHAPE[0], TRAIN_SHAPE[1]), dtype=np.float32)
                    for step in np.arange(0, TRAIN_SHAPE[1], 384)[:-1]:
                        tmp_pred = torch.sigmoid(model(images[:, :, :, step:step + 448])).cpu().numpy()
                        tmp_batch_preds[:, :, :, step:step + 448] += tmp_pred
                    tmp_batch_preds[:, :, :, 384:384 + 64] /= 2
                    tmp_batch_preds[:, :, :, 2 * 384:2 * 384 + 64] /= 2
                    tmp_batch_preds[:, :, :, 3 * 384:3 * 384 + 64] /= 2
                    batch_preds += tmp_batch_preds
            else:
                for model in models:
                    if config['TTA'] == 'true':
                        model = tta.SegmentationTTAWrapper(model, transforms)
                    batch_preds += torch.sigmoid(model(images)).cpu().numpy()
            batch_preds = batch_preds / len(models)

            for i in range(batch_preds.shape[0]):
                tmp = cv2.resize(np.moveaxis(batch_preds[i], 0, -1), (TEST_SHAPE[1], TEST_SHAPE[0]))
                batch_preds_test_shape[i] = np.moveaxis(tmp, -1, 0)

            for fname, preds in zip(fnames, batch_preds_test_shape):
                for cls, pred in enumerate(preds):
                    if cls_df is not None:
                        if cls_df.loc[fname + f"_{inv_map[cls]}"]['is_mask_empty'] == 1:
                            pred = np.zeros((TEST_SHAPE[0], TEST_SHAPE[1]))
                        else:
                            pred = post_process(pred, best_threshold, min_size, cls,
                                                use_dense_crf=config['use_dense_crf'],
                                                image=cv2.imread(test_dataset.images[i]) if config['use_dense_crf']=='true' else None,
                                                use_dilations=config['use_dilations'],
                                                use_poligonization=config['use_poligonization'])

                    else:
                        pred = post_process(pred, best_threshold, min_size, cls,
                                            use_dense_crf=config['use_dense_crf'],
                                            image=cv2.imread(test_dataset.images[i]) if config['use_dense_crf']=='true' else None,
                                            use_dilations=config['use_dilations'],
                                            use_poligonization=config['use_poligonization'])
                    rle = mask2rle(pred)
                    name = fname + f"_{inv_map[cls]}"
                    image_names.append(name)
                    predictions.append(rle)

    df = pd.DataFrame()
    df["Image_Label"] = image_names
    df["EncodedPixels"] = predictions
    df.to_csv(os.path.join(config['weights'], config['name'], "submission.csv"), index=False)


def post_process(mask, threshold, min_size, cls, use_dense_crf=False, use_dilations=False, use_poligonization=False, image=None):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    if use_dense_crf  == 'true' and image is not None:
        mask = crf.dense_crf(np.array(cv2.resize(image, (TEST_SHAPE[1], TEST_SHAPE[0]))).astype(np.uint8), mask)
    else:
        if not isinstance(threshold, list):
            mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
        elif isinstance(threshold, list):
            mask = (mask > threshold[cls]).astype(np.uint8)

    if use_dilations == 'true':
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

    if min_size > 0:
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8).copy())
        mask = np.zeros((TEST_SHAPE[0], TEST_SHAPE[1]), np.float32)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                mask[p] = 1

    if use_poligonization == 'true':
        cnts, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # get largest five contour area
        poligon_mask = np.zeros_like(mask, dtype=np.float32)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h >= 15:
                poligon_mask[y : y + h, x : x + w] = 1.0

            # x = optimize_trapezoid(Polygon(c.squeeze()))
            # x0 = int(x[:4].min())
            # y0 = int(x[4:].min())
            # w = int(x[:4].max() - x[:4].min())
            # h = int(x[4:].max() - x[4:].min())
            # if h >= 15:
            #     poligon_mask[y0:y0 + h, x0:x0 + w] = 1.0
        return poligon_mask

    return mask


if __name__ == '__main__':
    args = parse_args()
    config_main = read_config(args.config_file, "MAIN")
    config = read_config(args.config_file, "TEST")
    inv_map = {v: k for k, v in mapping.items()}
    crf = CRF(h=TEST_SHAPE[0], w=TEST_SHAPE[1])
    main()