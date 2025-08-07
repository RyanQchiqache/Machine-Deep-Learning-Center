import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import numpy as np
import PIL.Image as I
import rasterio
import cv2

class SatelliteDataset(Dataset):
    """
        A custom PyTorch Dataset for satellite image segmentation.

        Supports loading from image/mask paths or in-memory arrays.
        Applies optional RGB-to-class mapping, cropping, relabeling, and transformations.
    """
    def __init__(self, images, masks, rgb_to_class=None, patchify_enabled=False, patch_size=512, transform=None, relabel_fn=None, is_test=False):
        """
            Initializes the SatelliteDataset.

            Args:
                images (list): List of image file paths or NumPy arrays.
                masks (list): List of mask file paths or NumPy arrays.
                rgb_to_class (callable, optional): Function to convert RGB masks to class indices.
                patchify_enabled (bool): If True, apply patchification (not yet implemented here).
                patch_size (int): Size of the patches if patchify_enabled is True.
                transform (callable, optional): Albumentations transform to apply on image-mask pairs.
                relabel_fn (callable, optional): Function to relabel mask values.
                is_test (bool): If True, skip mask loading and return image only.
        """
        self.images = images
        self.masks = masks
        self.rgb_to_class = rgb_to_class
        self.patchify_enabled = patchify_enabled
        self.patch_size = patch_size
        self.from_paths = isinstance(images[0], str)
        self.transform = transform
        self.relabel_fn = relabel_fn
        self.is_test = is_test

    def __len__(self):
        """
            Returns the total number of samples in the dataset.

            Returns:
                int: Number of images/masks.
        """
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        """
            Loads an image from a given file path.

            Supports RGB GeoTIFF and common image formats. Normalizes pixel values to [0, 1].

            Args:
                path (str): File path to the image.

            Returns:
                np.ndarray: Normalized RGB image as a NumPy array.
        """
        if path.endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                img = src.read([1, 2, 3])
                img = np.transpose(img, (1, 2, 0))
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = img.astype(np.float32) / 255.0
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        """
            Loads a mask from a file path and processes it into class labels.

            Applies RGB-to-class mapping if specified. Supports GeoTIFF or PNG/JPEG masks.

            Args:
                path (str): File path to the mask.

            Returns:
                np.ndarray: Mask as a 2D array of class indices (dtype int64)
        """
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

    def __getitem__(self, idx):
        """
            Retrieves and processes a single sample (image and mask) from the dataset.

            Applies transformations, normalization, and relabeling if configured.

            Args:
                idx (int): Index of the sample.

            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                    If `is_test` is True or mask is None → returns only image tensor.
                    Otherwise → returns (image, mask) tensor pair.
                """
        if self.from_paths:
            image = self._load_image(self.images[idx])
            mask = self._load_mask(self.masks[idx]) if self.masks is not None else None
        else:
            image = self.images[idx]
            mask = self.masks[idx].astype(np.int64) if self.masks is not None else None

        # If test mode and no mask, just return the image
        if self.is_test or mask is None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
            return image

        if self.relabel_fn is not None:
            assert isinstance(mask, np.ndarray), "Mask must be a NumPy array before relabeling"
            mask = self.relabel_fn(mask)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
            mask = torch.from_numpy(mask).long()

        return image, mask
