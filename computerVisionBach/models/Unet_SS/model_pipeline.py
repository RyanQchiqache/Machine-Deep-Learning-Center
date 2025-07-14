import os
import cv2
import sys
import numpy as np
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from patchify import patchify
from typing import Callable, Tuple, List
import csv
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datetime import datetime
from computerVisionBach.models.Unet_SS.satellite_dataset.flair_dataset import FlairDataset
from computerVisionBach.models.Unet_SS.satellite_dataset.satellite_data import SatelliteDataset
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import segmentation_models_pytorch as smp
from computerVisionBach.models.Unet_SS.Unet import UNet
from computerVisionBach.models.Unet_SS import visualisation
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, UperNetForSemanticSegmentation, SegformerImageProcessor
from transformers.modeling_utils import PreTrainedModel
from torchvision import transforms as T
from computerVisionBach.models.Unet_SS import utils
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")


# ================================
# Configuration
# ================================
PATCH_SIZE = 512
OVERLAP = 0.5
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
TRAIN_CSV_PATH =  "/home/ryqc/data/flair_dataset/cleaned-train01.csv"
TEST_CSV_PATH = "/home/ryqc/data/flair_dataset/cleaned-test01.csv"
BASE_DIR = "/home/ryqc/data/flair_dataset"
transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
FLAIR_USED_LABELS = [1, 2, 3, 6, 7, 8, 10, 11, 13, 18]


# =====================================
# patchify and load data FLAIR
# =====================================
def prepare_datasets_from_csvs(
    train_csv_path: str,
    val_csv_path: str,
    base_dir: str = None
) -> Tuple[FlairDataset, FlairDataset]:
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

    relabel_fn = lambda mask: relabel_mask(mask, original_labels=FLAIR_USED_LABELS)
    train_dataset = FlairDataset(train_imgs, train_masks, transform=transforms)
    val_dataset = FlairDataset(val_imgs, val_masks,transform=transforms)

    return train_dataset, val_dataset

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

# =====================================
# patchify and load data DLR skyscapes
# =====================================

def patchify_image_or_masks(image_mask:np.ndarray):
    patch_size = PATCH_SIZE
    step = patch_size - OVERLAP
    H, W = image_mask.shape[:2]

    pad_bottom = (patch_size - H // patch_size) % patch_size
    pad_right = (patch_size - W // patch_size) % patch_size

    if pad_bottom or pad_right:
        image = np.pad(image_mask, ((0, pad_bottom), (0, pad_right), (0, 0)) if image_mask.ndim == 3 else ((0, pad_bottom), (0, pad_right)), mode= "constant", constant_values=0)
        H,W = image.shape[:2]

    patches = []

    for top in range(0, H -patch_size + 1, step):
        for left in range(0, W - patch_size + 1, step):
            patch = image[top:top + patch_size, left:left + patch_size]
            patches.append(patch)

    return patches , (H, W)

def load_folder(image_dir, mask_dir):
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    X, y = [], []
    for img_path, mask_path in zip(images, masks):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.COLOR_RGB2BGR)
        mask = utils.remap_mask(mask)

        if patchify_enabled:
            image_p = patchify_image_or_masks(img)
            mask_p = patchify_image_or_masks(mask)
            X.append(image_p)
            y.append(mask_p)

        else:
            X.append(img)
            y.append(mask)

    return X, y

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

        if isinstance(model, PreTrainedModel):
            # Convert to numpy and preprocess
            images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
            inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs).logits
        else:
            outputs = model(images)

        # Resize if necessary
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


# ================================
# Training Loop
# ================================
def train(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=15, writer=None):
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    if model_name.lower() == "unet":
        model = UNet(3, N_CLASSES).to(device)

    elif model_name.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",  # or "timm-efficientnet-b4" (requires `timm`)
            encoder_weights="imagenet",
            in_channels=3,
            classes=N_CLASSES,
            activation=None  # set to "softmax" if you want it inside the model
        ).to(device)
    elif model_name.lower() == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=N_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(device)

    elif model_name.lower() == "upernet":
        model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small",
            num_labels=N_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(device)
    elif model_name.lower() == "unet_resnet50":
        model = smp.Unet(
            encoder_name="resnet50",
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
        device=device,
        num_epochs=NUM_EPOCHS,
        writer=writer
    )
    # Save model after training
    model_save_path = f"checkpoints_flair/{model_name}_model_flair.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    evaluate(model, test_loader, device)
    visualisation.visualize_prediction(model, test_loader, device)

    return model

# ================================
# Evaluation
# ================================
def evaluate(model, dataloader, device, epoch=None, writer=None):
    model.eval()
    iou_macro = MulticlassJaccardIndex(num_classes=N_CLASSES, average='macro').to(device)
    iou_per_class = MulticlassJaccardIndex(num_classes=N_CLASSES, average=None).to(device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            # HuggingFace transformer models
            if isinstance(model, PreTrainedModel):
                images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
                output = model(**inputs)
                logits = output.logits
            else:
                logits = model(images)

            # Resize logits if needed
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )

            preds = torch.argmax(logits, dim=1)

            # Incremental metric updates (no memory explosion)
            iou_macro.update(preds, masks)
            iou_per_class.update(preds, masks)

    # Compute final metrics
    miou = iou_macro.compute()
    per_class_ious = iou_per_class.compute()

    print(f"\n✓ Mean IoU: {miou:.4f}")
    for i, class_iou in enumerate(per_class_ious):
        print(f"  └─ Class {i:02d} IoU: {class_iou:.4f}")

    # Write to TensorBoard if available
    if writer and epoch is not None:
        writer.add_scalar("IoU/Mean", miou, epoch)
        for i, val in enumerate(per_class_ious):
            writer.add_scalar(f"IoU/Class_{i}", val, epoch)

    # Reset metrics after evaluation (optional if reused)
    iou_macro.reset()
    iou_per_class.reset()




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
    """parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["flair", "dlr"], default="dlr")
    args = parser.parse_args()"""
    dataset_name = "flair"
    """dataset_choice = args.dataset or dataset_name"""

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "flair":
        train_dataset, val_dataset = prepare_datasets_from_csvs(
            train_csv_path=TRAIN_CSV_PATH,
            val_csv_path= TEST_CSV_PATH,
            base_dir=BASE_DIR
        )
    else:  # fallback to DLR dataset
         train_dataset,val_dataset = load_data_dlr(ROOT_DIR, dataset_type="SS_Dense")

    """# Add this after loading datasets
    print("Checking FLAIR label range...")
    all_labels = torch.cat([train_dataset[i][1].flatten() for i in range(20)])  # Sample 20 masks
    print("Unique labels in train set:", torch.unique(all_labels))
    print("N_CLASSES =", N_CLASSES)"""

    print(f"Dataset chosen is : {dataset_name}")
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    """sample_image, sample_mask = train_dataset[0]
    print("Label range (first mask):", sample_mask.min().item(), "to", sample_mask.max().item())
    print("Unique labels (first mask):", torch.unique(sample_mask).tolist())"""

    log_dir = f"runs/{dataset_name}_experiment_FLAIR{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    try:
        model = train_and_evaluate("unet_resnet50", train_dataset, val_dataset, device, writer)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        writer.close()

# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
