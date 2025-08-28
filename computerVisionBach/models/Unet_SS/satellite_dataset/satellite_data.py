import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
import numpy as np
import PIL.Image as I
import rasterio
import cv2

class SatelliteDataset(Dataset):
    def __init__(self, images, masks, rgb_to_class=None, patchify_enabled=False, patch_size=512, transform=None, relabel_fn=None, is_test=False, allowed_labels: Optional[Tuple[int]] = None, use_processor: bool = False, is_hf_model:bool = True):
        self.images = images
        self.masks = masks
        self.rgb_to_class = rgb_to_class
        self.patchify_enabled = patchify_enabled
        self.patch_size = patch_size
        self.from_paths = isinstance(images[0], str)
        self.transform = transform
        self.relabel_fn = relabel_fn
        self.is_test = is_test
        self.allowed_labels = allowed_labels
        self.use_processor = use_processor
        self.is_hf_model = is_hf_model

    def __len__(self):
        return len(self.images)

    def _load_image(self, path: str) -> np.ndarray:
        if path.endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                img = src.read([1, 2, 3])
                img = np.transpose(img, (1, 2, 0))
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = img.astype(np.float32)
        return img

    def _load_mask(self, path: str) -> np.ndarray:
        if path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(path) as src:
                if src.count == 1:
                    mask = src.read(1)
                    if src.nodata is not None:
                        mask = np.where(mask == src.nodata, 255, mask)
                elif src.count >= 3:
                    rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                    if self.rgb_to_class:
                        mask = self.rgb_to_class(rgb)
                    else:
                        raise ValueError("RGB GeoTIFF mask requires rgb_to_class mapping.")
                else:
                    raise ValueError(f"Unsupported GeoTIFF mask with {src.count} bands")
        else:
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise FileNotFoundError(f"Mask not found or unreadable: {path}")

            if mask.ndim == 3 and mask.shape[-1] == 3:
                if self.rgb_to_class:
                    mask = self.rgb_to_class(mask)
                else:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            elif mask.ndim != 2:
                raise ValueError(f"Unsupported mask format: {mask.shape} at {path}")

        return mask.astype(np.int64)

    def __getitem__(self, idx):
        # 1) load
        if self.from_paths:
            image = self._load_image(self.images[idx])
            mask = self._load_mask(self.masks[idx]) if self.masks is not None else None
        else:
            image = self.images[idx]
            mask = self.masks[idx].astype(np.int64) if self.masks is not None else None

        # 2) early return for test mode
        if self.is_test or mask is None:
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
            return image

        # 3) relabeling (e.g., keep only 1..13 -> 0..12, others -> 255)
        if self.relabel_fn is not None:
            mask = self.relabel_fn(mask)

        # 4) inspecting labels after relabeling
        """m = mask  # numpy at this point
        vals = np.unique(m[m != 255])
        print("Unique labels (no 255):", vals[:50])"""

        # 5) enforce allowed labels (optional safety)
        if self.allowed_labels is not None:
            used = np.unique(mask[mask != 255])
            unexpected = set(used.tolist()) - set(self.allowed_labels)
            assert len(unexpected) == 0, (
                f"Mask contains unexpected labels: {sorted(unexpected)}. "
                f"Allowed: {self.allowed_labels} or 255"
            )

        # 6) transforming / tensorize
        if self.transform:
            if self.is_hf_model:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            else:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"].long()

        else:# TODO: check both woth transformation and without how things will work
            if self.use_processor:
                return image, mask
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
                mask = torch.from_numpy(mask).long()
        #print(f"[Dataset] idx={idx} | image={type(image)}, {getattr(image, 'shape', None)} | mask={type(mask)}, {getattr(mask, 'shape', None)}")
        return image, mask

