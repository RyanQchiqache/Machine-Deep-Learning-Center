import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import numpy as np
import PIL.Image as I
import rasterio
import cv2

class SatelliteDataset(Dataset):
    def __init__(self, images, masks, rgb_to_class=None, patchify_enabled=False, patch_size=512, transform=None, relabel_fn=None):
        self.images = images
        self.masks = masks
        self.rgb_to_class = rgb_to_class
        self.patchify_enabled = patchify_enabled
        self.patch_size = patch_size
        self.from_paths = isinstance(images[0], str)
        self.transform = transform
        self.relabel_fn = relabel_fn

    def __len__(self):
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        if path.endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                img = src.read([1, 2, 3])
                img = np.transpose(img, (1, 2, 0))
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = img.astype(np.float32) / 255.0
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        # Use rasterio only for GeoTIFFs
        if path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                mask = src.read(1)
        else:
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                raise FileNotFoundError(f"Mask not found or unreadable: {path}")

            # for normal case RGB
            if mask.ndim == 3 and mask.shape[-1] == 3:
                if self.rgb_to_class:
                    mask = self.rgb_to_class(mask)
                else:
                    # Convert to grayscale if no mapping provided
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            elif mask.ndim == 2:
                # Already grayscale
                pass
            else:
                raise ValueError(f"Unsupported mask format: {mask.shape} at {path}")

        return mask.astype(np.int64)

        return mask.astype(np.int64)

    def __getitem__(self, idx):
        if self.from_paths:
            image = self._load_image(self.images[idx])
            mask = self._load_mask(self.masks[idx])
        else:
            image = self.images[idx]
            mask = self.masks[idx].astype(np.int64)

        if self.relabel_fn is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a NumPy array before relabeling"
            mask = self.relabel_fn(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        if mask is not None:
            mask = torch.from_numpy(mask).long()
            return image, mask
        else:
            return image
