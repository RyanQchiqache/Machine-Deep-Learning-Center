import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from simple_unet import UNet
from patchify import patchify
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from computerVisionBach.models.Unet_SS import utils

# Constants
PATCH_SIZE = 256
ROOT_DIR = '/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/SS_data'
N_CLASSES = 6
N_EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SatelliteDataset(Dataset):
    def __init__(self, image_patches, mask_patches):
        self.images = image_patches
        self.masks = mask_patches
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        mask = self.masks[idx].astype(np.int64)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).squeeze()
        return image, mask

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
    for class_id, rgb in utils.CLASS_COLOR_MAP.items():
        label[np.all(mask == rgb, axis=-1)] = class_id
    return label

def convert_masks_to_class_labels(masks):
    return np.array([rgb_to_class_label(mask) for mask in masks])

def visualize_sample(images, masks, labels):
    idx = random.randint(0, len(images) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(images[idx]); plt.title("Image")
    plt.subplot(1, 3, 2); plt.imshow(masks[idx]); plt.title("RGB Mask")
    plt.subplot(1, 3, 3); plt.imshow(labels[idx]); plt.title("Label Mask")
    plt.tight_layout(); plt.show()

def train(model, dataloader, epochs, criterion, optimizer):
    """
     function for training the model using model, dataloader, epochs, criterion, optimizer
    """
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {avg_loss:.4f}")

def evaluate(model, dataloader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for cls in range(N_CLASSES):
                pred_inds = (preds == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().float()
                union = (pred_inds | target_inds).sum().float()
                if union == 0:
                    print(f"union is 0")
                    continue

                else:
                    iou = (intersection + 1e-5) / (union + 1e-5)
                ious.append(iou)
    mean_iou = torch.stack(ious).mean()
    print(f"Mean IoU: {mean_iou:.4f}")

def class_to_rgb(mask):
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in utils.CLASS_COLOR_MAP.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask


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
            plt.subplot(1, 3, 1); plt.imshow(image_np); plt.title("Image")
            plt.subplot(1, 3, 2); plt.imshow(mask_np); plt.title("Ground Truth")
            plt.subplot(1, 3, 3); plt.imshow(class_to_rgb(pred_np)); plt.title("Prediction")
            plt.tight_layout(); plt.show()
            break

def main():
    images = extract_patches_from_directory(ROOT_DIR, kind='images')
    masks_rgb = extract_patches_from_directory(ROOT_DIR, kind='masks')
    masks_label = convert_masks_to_class_labels(masks_rgb)

    visualize_sample(images, masks_rgb, masks_label)

    X_train, X_test, y_train, y_test = train_test_split(images, masks_label, test_size=0.2, random_state=42,
                                                        shuffle=True)

    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    train(model, train_loader, N_EPOCHS, criterion, optimizer)
    print("Training completed.")

    # evaluation
    evaluate(model, test_loader, device)
    visualize_prediction(model, test_loader, device)


if __name__ == "__main__":
    main()