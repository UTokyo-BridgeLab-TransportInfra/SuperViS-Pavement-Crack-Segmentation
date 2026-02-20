import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import albumentations as A

class ImgDataSet(Dataset):
    def __init__(self, img_dir, img_size, img_fnames, img_transform,
                 mask_dir, mask_fnames, mask_transform,
                 augment=True, hyp=None, rect=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.img_fnames = img_fnames
        self.img_transform = img_transform
        self.mask_dir = mask_dir
        self.mask_fnames = mask_fnames
        self.mask_transform = mask_transform
        self.augment = augment     
        self.hyp = hyp
        self.rect = rect
        self.n = len(self.img_fnames)
        self.count = 0

    def __getitem__(self, i):
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath)

        mpath = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mpath[:-4] + '.jpg')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask[mask < 20] = 0
        mask[mask >= 20] = 255
        if self.augment:
            if random.random() < 0.5:  
                img = np.rot90(img)
                mask = np.rot90(mask)
            if random.random() < 0.25: 
                img = np.flipud(img)
                mask = np.flipud(mask)
            if random.random() < 0.15:  
                img = np.fliplr(img)
                mask = np.fliplr(mask)

        img = self.img_transform(Image.fromarray(img))
        mask = mask / 255.0
        return img, torch.from_numpy(mask)
    def __len__(self):
        return self.n

