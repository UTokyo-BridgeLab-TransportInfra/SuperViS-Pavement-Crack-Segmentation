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
                 augment, hyp, rect=False, 
                 mask_dir1=None, mask_dir2=None):

        # Image directory
        self.img_dir = img_dir

        # Target image size
        self.img_size = img_size

        # List of image file names
        self.img_fnames = img_fnames

        # Image preprocessing / augmentation transform
        self.img_transform = img_transform

        # Main ground-truth mask directory
        self.mask_dir = mask_dir

        # Auxiliary mask directory 1 (optional)
        self.mask_dir1 = mask_dir1

        # Auxiliary mask directory 2 (optional)
        self.mask_dir2 = mask_dir2

        # Mask file names
        self.mask_fnames = mask_fnames

        
        self.mask_transform = mask_transform

       
        self.augment = augment

        # Hyperparameters
        self.hyp = hyp

      
        self.rect = rect

        # Number of samples
        self.n = len(self.img_fnames)

       
        self.mosaic = self.augment and not self.rect

        self.mosaic_border = [-img_size // 2, -img_size // 2]

  
        self.count = 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):

        # Load image
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dir, fname)
        img = cv2.imread(fpath)

        # Load main mask
        mask_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mask_path[:-4] + '.jpg')

        # Load auxiliary mask 1 (if provided)
        if self.mask_dir1 is not None:
            mask1_path = os.path.join(self.mask_dir1, fname)
            mask1 = cv2.imread(mask1_path[:-4] + '.jpg')
        else:
            # Create empty mask if not provided
            mask1 = np.zeros_like(mask)

        # Load auxiliary mask 2 (if provided)
        if self.mask_dir2 is not None:
            mask2_path = os.path.join(self.mask_dir2, fname)
            mask2 = cv2.imread(mask2_path[:-4] + '.jpg')
        else:
            # Create empty mask if not provided
            mask2 = np.zeros_like(mask)

        # Convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert masks to grayscale
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
        mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

        # Binarize masks (threshold = 50)
        for m in [mask, mask1, mask2]:
            m[m < 50] = 0
            m[m >= 50] = 255

        # Save first 10 samples for debugging/visual inspection
        if self.count < 10:
            cv2.imwrite(f'test_results/train_img_batch{self.count}.jpg',
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'test_results/train_mask_batch{self.count}.jpg', mask)
            cv2.imwrite(f'test_results/train_mask1_batch{self.count}.jpg', mask1)
            cv2.imwrite(f'test_results/train_mask2_batch{self.count}.jpg', mask2)
            self.count += 1

        # Convert image to PyTorch tensor using defined transform
        img = self.img_transform(Image.fromarray(img))

        # Convert masks to float tensors in range [0, 1]
        mask = torch.from_numpy(mask / 255.0).float()
        mask1 = torch.from_numpy(mask1 / 255.0).float()
        mask2 = torch.from_numpy(mask2 / 255.0).float()

        return img, mask, mask1, mask2, fname
