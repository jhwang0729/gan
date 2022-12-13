import os
from glob import glob
from pathlib import Path

import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as T
from torch.utils.data import Dataset

from utils.general import *


def load_img(img_path: str, img_size: int):
    img = cv2.imread(img_path)
    assert img is not None, f'Image path is wrong'
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    return img / 255.0
    # return np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255.0


class MyDataset(Dataset):
    def __init__(self, img_dir: str, img_size: int, m: float = 0.0, std: float = 1.0):
        self.img_files = glob(img_dir + os.sep + '*.*')
        self.length = len(self.img_files)
        assert self.length, 'No images found'
        print(f'{self.length} number of images found')
        self.img_size = img_size
        self.m = m
        self.std = std

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        img = load_img(self.img_files[item], self.img_size)

        return img, torch.randn(100)