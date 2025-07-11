import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from computerVisionBach.models.Unet_SS import utils

# ================================
# Dataset Class
# ================================
class SatelliteDataset(Dataset):
    def __init__(self, image_patches, mask_patches):
        self.images = image_patches
        self.masks = mask_patches

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.int64)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).squeeze()
        return image, mask



