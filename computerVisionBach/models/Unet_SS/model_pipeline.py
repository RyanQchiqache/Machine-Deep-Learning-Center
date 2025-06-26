import os
import cv2
import numpy as np
from PIL import Image
from patchify import patchify
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from computerVisionBach.models.Unet_SS.satellite_data import SatelliteDataset
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from computerVisionBach.models.Unet_SS.Unet import UNet
from computerVisionBach.models.Unet_SS import visualisation

# ================================
# Configuration
# ================================
PATCH_SIZE = 256
ROOT_DIR = '/computerVisionBach/SS_data'
N_CLASSES = 6
BATCH_SIZE = 16
NUM_EPOCHS = 15
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


def load_data(root_dir, test_size, seed):
    images = extract_patches_from_directory(root_dir, kind='images')
    masks_rgb = extract_patches_from_directory(root_dir, kind='masks')
    masks_label = convert_masks_to_class_labels(masks_rgb)

    visualisation.visualize_sample(images, masks_rgb, masks_label)

    X_train, X_test, y_train, y_test = train_test_split(images, masks_label, train_size=1 - test_size,
                                                        random_state=seed)
    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_test, y_test)

    return train_dataset, test_dataset


def get_loss_and_optimizer(model, lr):
    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss()
    criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return criterion, optimizer


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs, _ = model(images, return_features=True)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# ================================
# Training Loop
# ================================
def train(model, train_loader, test_loader, criterion, optimizer, test_dataset, device, num_epochs=15):
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f}")

        if (epoch + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                val_image = test_dataset[0][0].unsqueeze(0).to(device)
                _, features = model(val_image, return_features=True)
                for name in ["bottle_neck", "enc1", "enc2", "dec3", "dec4"]:
                    visualisation.visualise_feture_map(features[name], f"{name} (Epoch {epoch + 1})")
            evaluate(model, test_loader, device)


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


#=======================================
# main function
#=======================================
def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"file not found: {ROOT_DIR}")

    train_dataset, test_dataset = load_data(ROOT_DIR, test_size=0.2, seed=RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # SMP: ENCODER NAME, ENCODER WEIGHTS, IN_CHANNEL, NUM_CLASSES
    model = UNet(3, N_CLASSES).to(device)
    criterion, optimizer = get_loss_and_optimizer(model, lr=1e-4)

    print(f"\n⚡ Starting training on: {device}")
    train(model, train_loader, test_loader, criterion, optimizer, test_dataset, device, NUM_EPOCHS)

    print("\n\u2705 Training completed.")
    evaluate(model, test_loader, device)
    visualisation.visualize_prediction(model, test_loader, device)


# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
