import random

import numpy as np
import pydensecrf.densecrf as dcrf
import torch
from torch import nn
import torch.functional as F
import yaml
from shapely.geometry import Polygon
from scipy.optimize import minimize


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(2100, 1400)):
    """
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape).T


def read_config(path, stage:str):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)[stage]
        except yaml.YAMLError as exc:
            print(exc)
            return None


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def batch2device(data, device):
    return {k: v if not hasattr(v, 'to') else v.to(device) for k, v in data.items()}


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class CRF:
    def __init__(self, w, h):
        self.w = w
        self.h = h


    def dense_crf(self, img, output_probs):

        output_probs = np.expand_dims(output_probs, 0)
        output_probs = np.append(1 - output_probs, output_probs, axis=0)

        d = dcrf.DenseCRF2D(self.w, self.h, 2)
        U = -np.log(output_probs)
        U = U.reshape((2, -1))
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=20, compat=3)
        d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

        Q = d.inference(5)
        Q = np.argmax(np.array(Q), axis=0).reshape((self.h, self.w))

        return Q


def optimize_trapezoid(poly):
    poly = poly.simplify(tolerance=2).convex_hull
    initial_box = poly.minimum_rotated_rectangle

    x0 = initial_box.exterior.xy
    x0 = np.concatenate([x0[0][:-1], x0[1][:-1]])

    def loss(x):
        opt_poly = Polygon([(x[i], x[i + 4]) for i in range(0, 4)])
        if opt_poly.is_valid:
            intersection = poly.intersection(opt_poly).area
            union = poly.union(opt_poly).area
            return -intersection / union
        else:
            return 0.0

    res = minimize(loss, x0, method='SLSQP', options={"maxiter": 10})
    return res['x']