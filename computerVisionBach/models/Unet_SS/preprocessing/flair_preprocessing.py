import os
import sys
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from typing import Tuple, List, Optional
import csv
from computerVisionBach.models.Unet_SS.satellite_dataset.flair_dataset import FlairDataset
from transformers import SegformerImageProcessor
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

tf_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

tf_val = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


# =====================================
# patchify and load data FLAIR
# =====================================
def prepare_datasets_from_csvs(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: Optional[str] = None,
    base_dir: str = None
) -> Tuple[FlairDataset, FlairDataset, FlairDataset]:
    def load_csv(csv_path: str) -> List[Tuple[str, str]]:
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            return [(row[0], row[1]) for row in reader if len(row) == 2]

    def resolve_path(p: str) -> str:
        return os.path.normpath(os.path.join(base_dir, p)) if base_dir and not os.path.isabs(p) else p

    # Load CSVs
    train_pairs = load_csv(train_csv_path)
    val_pairs = load_csv(val_csv_path)


    train_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in train_pairs]
    val_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in val_pairs]

    train_imgs, train_masks = zip(*train_pairs)
    val_imgs, val_masks = zip(*val_pairs)

    def relabel_fn(mask):
        relabeled = np.full_like(mask, 255, dtype=np.int64)
        valid = (mask >= 1) & (mask <= 19)
        relabeled[valid] = mask[valid] - 1
        return relabeled

    def relabel_fn_12(mask):
        """
        Keeps only class labels 1 to 12 from the original FLAIR dataset,
        and remaps them to 0 to 11. All other pixels are mapped to 255 (ignore index).
        """
        relabeled = np.full_like(mask, 255, dtype=np.uint8)  # 255 = ignore index
        for i in range(1, 13):  # FLAIR class IDs 1–12
            relabeled[mask == i] = i - 1  # Remap to 0–12
        return relabeled

    train_dataset = FlairDataset(train_imgs, train_masks, transform=tf_train, relabel_fn=relabel_fn_12, allowed_labels=tuple(range(12)))
    val_dataset = FlairDataset(val_imgs, val_masks,transform=tf_val, relabel_fn=relabel_fn_12, allowed_labels=tuple(range(12)))

    if test_csv_path is not None:
        test_pairs = load_csv(test_csv_path)
        test_pairs = [(resolve_path(img), resolve_path(mask)) for img, mask in test_pairs]
        test_imgs, test_masks = zip(*test_pairs)
        test_dataset = FlairDataset(test_imgs, test_masks, transform=tf_val)

    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset

def relabel_mask(mask: np.ndarray, original_labels: list) -> np.ndarray:
    """
    Remaps original sparse label values (e.g., [1, 2, 6, 18]) to contiguous [0, 1, 2, ..., N-1]
    so CrossEntropyLoss works without index errors.
    """
    label_map = {orig: new for new, orig in enumerate(sorted(original_labels))}
    remapped = np.zeros_like(mask)
    for orig_label, new_label in label_map.items():
        remapped[mask == orig_label] = new_label
    return remapped