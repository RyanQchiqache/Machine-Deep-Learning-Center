import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from simple_unet_plus_resnet34 import UNet
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

# ================================
# Configuration
# ================================
PATCH_SIZE = 256
ROOT_DIR = '/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/SS_data'
N_CLASSES = 6
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

CLASS_COLOR_MAP = {
    0: np.array([60, 16, 152]),
    1: np.array([132, 41, 246]),
    2: np.array([110, 193, 228]),
    3: np.array([254, 221, 58]),
    4: np.array([226, 169, 41]),
    5: np.array([155, 155, 155])
}


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


# ================================
# Utility Functions
# ================================
def extract_patches_from_directory(directory, kind='images'):
    dataset = []
    for path, subdirs, files in os.walk(directory):
        if path.endswith(kind):
            for file in sorted(os.listdir(path)):
                if (kind == 'images' and file.endswith('.jpg')) or (kind == 'masks' and file.endswith('.png')):
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path, 1)
                    if kind == 'masks':
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE, (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
                    img = Image.fromarray(img).crop((0, 0, w, h))
                    img = np.array(img)
                    patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
                    for i in range(patches.shape[0]):
                        for j in range(patches.shape[1]):
                            dataset.append(patches[i, j, 0])
    return np.array(dataset)


def rgb_to_class_label(mask):
    label = np.zeros(mask.shape[:2], dtype=np.uint8)
    for class_id, rgb in CLASS_COLOR_MAP.items():
        label[np.all(mask == rgb, axis=-1)] = class_id
    return label


def convert_masks_to_class_labels(masks):
    return np.array([rgb_to_class_label(mask) for mask in masks])


def class_to_rgb(mask):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLOR_MAP.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask


def visualize_sample(images, masks, labels):
    idx = random.randint(0, len(images) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(images[idx])
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(masks[idx])
    plt.title("RGB Mask")
    plt.subplot(1, 3, 3)
    plt.imshow(labels[idx])
    plt.title("Label Mask")
    plt.tight_layout()
    plt.show()


# ================================
# Evaluation
# ================================
def evaluate(model, dataloader, device):
    model.eval()
    iou_metric = MulticlassJaccardIndex(num_classes=N_CLASSES, average='macro').to(device)
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(masks)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    miou = iou_metric(preds, targets)
    print(f"\n\u2713 Mean IoU: {miou:.4f}")


# ================================
# Prediction Visualization
# ================================
def visualize_prediction(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1);
            plt.imshow(image_np);
            plt.title("Image")
            plt.subplot(1, 3, 2);
            plt.imshow(class_to_rgb(mask_np));
            plt.title("Ground Truth")
            plt.subplot(1, 3, 3);
            plt.imshow(class_to_rgb(pred_np));
            plt.title("Prediction")
            plt.tight_layout();
            plt.show()
            break


# ================================
# Training Script
# ================================
if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = extract_patches_from_directory(ROOT_DIR, kind='images')
    masks_rgb = extract_patches_from_directory(ROOT_DIR, kind='masks')
    masks_label = convert_masks_to_class_labels(masks_rgb)

    visualize_sample(images, masks_rgb, masks_label)

    X_train, X_test, y_train, y_test = train_test_split(images, masks_label, test_size=0.2, random_state=RANDOM_SEED)
    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=N_CLASSES,
    ).to(device)

    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss()
    criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n\u26A1 Starting training on:", device)
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, masks in  tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            evaluate(model, test_loader, device)

    print("\n\u2705 Training completed.")
    evaluate(model, test_loader, device)
    visualize_prediction(model, test_loader, device)
