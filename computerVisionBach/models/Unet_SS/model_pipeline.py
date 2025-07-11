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
import segmentation_models_pytorch as smp
from computerVisionBach.models.Unet_SS.Unet import UNet
from computerVisionBach.models.Unet_SS import visualisation
from computerVisionBach.models.Unet_SS import utils
from torch.utils.tensorboard import SummaryWriter

# ================================
# Configuration
# ================================
PATCH_SIZE = 512
ROOT_DIR = '/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/DLR_dataset'
N_CLASSES = 20
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
RANDOM_SEED = 42
MODELS = {}
writer = SummaryWriter(log_dir=os.path.join(ROOT_DIR, 'logs'))
patchify_enabled = True
NUM_RECONSTRUCTIONS = 4
# ================================
# patchify and load data kaggel
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
    masks_label = utils.convert_masks_to_class_labels(masks_rgb)

    visualisation.visualize_sample(images, masks_rgb, masks_label)

    X_train, X_test, y_train, y_test = train_test_split(images, masks_label, train_size=1 - test_size,
                                                        random_state=seed)
    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_test, y_test)

    return train_dataset, test_dataset

# =====================================
# patchify and load data DLR skyscapes
# =====================================
def load_folder(image_dir, mask_dir):
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    X, y = [], []
    for img_path, mask_path in zip(images, masks):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = utils.remap_mask(mask)

        h = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
        w = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
        img = img[:h, :w]
        mask = mask[:h, :w]

        if patchify_enabled:
            img_patches = patchify(img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
            mask_patches = patchify(mask, (PATCH_SIZE, PATCH_SIZE), step=PATCH_SIZE)

            for i in range(img_patches.shape[0]):
                for j in range(img_patches.shape[1]):
                    X.append(img_patches[i, j, 0])
                    y.append(mask_patches[i, j])
        else:
            X.append(img)
            y.append(mask)

    return np.array(X), np.array(y)


def load_data_dlr(base_dir, dataset_type="SS_Dense"):
    base = os.path.join(base_dir, dataset_type)

    X_train, y_train = load_folder(
        os.path.join(base, "train/images"),
        os.path.join(base, "train/labels/grayscale")
    )
    X_val, y_val = load_folder(
        os.path.join(base, "val/images"),
        os.path.join(base, "val/labels/grayscale")
    )

    color_map_rgb = {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}

    visualisation.visualize_sample(
        X_train,
        [utils.class_to_rgb(mask, color_map_rgb) for mask in y_train],
        y_train
    )

    train_dataset = SatelliteDataset(X_train, y_train)
    test_dataset = SatelliteDataset(X_val, y_val)

    return train_dataset, test_dataset




def get_loss_and_optimizer(model):
    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss()
    #criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return ce_loss, optimizer


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# ================================
# Training Loop
# ================================
def train(model, train_loader, test_loader, criterion, optimizer, test_dataset, device, num_epochs=15, writer=None):
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f}")

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)

        model.eval()
        with torch.no_grad():
            model.eval()
            evaluate(model, test_loader, device)
            """if (epoch + 1) % 10 == 0 or epoch == 1:
                val_image = test_dataset[0][0].unsqueeze(0).to(device)
                _, features = model(val_image, return_features=True)
                for name in ["bottleneck", "enc1", "enc2", "dec3", "dec4"]:
                    visualisation.visualise_feature_map(features[name], f"{name} (Epoch {epoch + 1})")"""

def train_and_evaluate(model_name, train_dataset, test_dataset, device, writer=None):
    print(f"\nTraining model: {model_name}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    if model_name.lower() == "unet":
        model = UNet(3, N_CLASSES).to(device)
    elif model_name.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",       # or "efficientnet-b0", "mobilenet_v2", etc.
            encoder_weights="imagenet",
            in_channels=3,
            classes=N_CLASSES,
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion, optimizer = get_loss_and_optimizer(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        test_dataset=test_dataset,
        device=device,
        num_epochs=NUM_EPOCHS,
        writer=writer
    )
    # Save model after training
    model_save_path = f"checkpoints/{model_name}_model.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    evaluate(model, test_loader, device)
    visualisation.visualize_prediction(model, test_loader, device)

    return model

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
# Reconstruction of images
# ================================
def reconstruct_two_examples(model, test_dataset, color_map, num_reconstructions):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    patch_size = 512
    patch_shape = (4, 4)
    patches_per_image = patch_shape[0] * patch_shape[1]

    all_images, all_masks, all_preds = [], [], []
    count = 0

    with torch.no_grad():
        for img_tensor, mask_tensor in test_loader:
            if count >= num_reconstructions:
                break

            img_tensor = img_tensor.cuda()
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            all_images.append(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
            all_masks.append(mask_tensor.squeeze(0).numpy())
            all_preds.append(pred)

            # Once 16 patches collected, reconstruct a full image
            if len(all_images) == patches_per_image:
                save_path = f"reconstructed_outputs/reconstruction_{count}.png"
                visualisation.reconstruct_and_visualize_patches(
                    np.array(all_images), np.array(all_masks), np.array(all_preds),
                    patch_size=patch_size,
                    grid_shape=patch_shape,
                    color_map=color_map,
                    save_path=save_path
                )
                count += 1
                all_images, all_masks, all_preds = [], [], []



#=======================================
# main function
#=======================================
def main():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(f"file not found: {ROOT_DIR}")

    #train_dataset, test_dataset = load_data(ROOT_DIR, test_size=0.2, seed=RANDOM_SEED)
    train_dataset, test_dataset = load_data_dlr(ROOT_DIR, dataset_type="SS_Dense")

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("Train label range:", np.min(train_dataset.masks), "to", np.max(train_dataset.masks))
    print("Test label range:", np.min(test_dataset.masks), "to", np.max(test_dataset.masks))
    assert np.max(train_dataset.masks) < N_CLASSES

    writer = SummaryWriter(log_dir="runs/Unet_vs_DeepLab")

    try:
        # train_and_evaluate("unet", train_dataset, test_dataset, device, writer)
        model = train_and_evaluate("deeplabv3+", train_dataset, test_dataset, device, writer)

        color_map = {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}
        reconstruct_two_examples(model, test_dataset, color_map, num_reconstructions=4)
    except KeyboardInterrupt :
        raise KeyboardInterrupt("training interrupted due to keyboard interrupt")
    finally:
        writer.close()


# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
