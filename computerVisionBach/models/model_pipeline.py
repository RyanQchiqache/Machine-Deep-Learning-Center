import copy
import os
import cv2
import sys
import numpy as np
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from patchify import patchify
from typing import Callable, Tuple, List
import csv
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from datetime import datetime
from computerVisionBach.models.Unet_SS.satellite_dataset.flair_dataset import FlairDataset
from computerVisionBach.models.Unet_SS.satellite_dataset.satellite_data import SatelliteDataset
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm
import segmentation_models_pytorch as smp
from computerVisionBach.models.Unet_SS.SS_models.Unet_resnet34 import UNetResNet34
from computerVisionBach.models.Unet_SS.SS_models.Unet import UNet
from computerVisionBach.models.Unet_SS import visualisation
from torch.utils.tensorboard import SummaryWriter
from transformers import SegformerForSemanticSegmentation, UperNetForSemanticSegmentation, SegformerImageProcessor
from transformers.modeling_utils import PreTrainedModel
from torchvision import transforms as T
from computerVisionBach.models.Unet_SS import utils
from computerVisionBach.models.Unet_SS.preprocessing.flair_preprocessing import prepare_datasets_from_csvs
from computerVisionBach.models.Unet_SS.preprocessing import dlr_preprocessing
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")


# ================================
# Configuration
# ================================
PATCH_SIZE = 2000
OVERLAP = 0.5
ROOT_DIR = '/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/DLR_dataset'
N_CLASSES = 20
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
MODELS = {}
patchify_enabled = True
NUM_RECONSTRUCTIONS = 4
TRAIN_CSV_PATH =  "/home/ryqc/data/flair_dataset/cleaned-train01.csv"
TEST_CSV_PATH = "/home/ryqc/data/flair_dataset/cleaned-test01.csv"
VAL_CSV_PATH = ""
BASE_DIR = "/home/ryqc/data/flair_dataset"
FLAIR_USED_LABELS = [1, 2, 3, 6, 7, 8, 10, 11, 13, 18]


def get_loss_and_optimizer(model):
    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    #criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    return ce_loss, scheduler, optimizer


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
def train(model, train_loader,val_loader, criterion, optimizer, device, num_epochs, writer=None):
    best_miou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f}")

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)

        model.eval()
        with torch.no_grad():
            miou = evaluate(model, val_loader, device ,epoch=None, writer=writer)

        visualize_val_predictions(model, val_loader, device, epoch, processor=processor, writer=writer)

        if miou > best_miou:
            best_miou = miou
            best_model_wts = copy.deepcopy(model.state_dict())
            """if (epoch + 1) % 10 == 0 or epoch == 1:
                val_image = test_dataset[0][0].unsqueeze(0).to(device)
                _, features = model(val_image, return_features=True)
                for name in ["bottleneck", "enc1", "enc2", "dec3", "dec4"]:
                    visualisation.visualise_feature_map(features[name], f"{name} (Epoch {epoch + 1})")"""
    model.load_state_dict(best_model_wts)

def train_and_evaluate(model_name, train_dataset, val_dataset, test_dataset, device, writer=None):
    print(f"\nTraining model: {model_name}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


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

    criterion, scheduler, optimizer = get_loss_and_optimizer(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        writer=writer
    )
    # Save model after training
    # Base checkpoint directory (relative to this script)
    checkpoint_base = os.path.join(os.path.dirname(__file__), "Unet_SS/checkpoints")
    os.makedirs(checkpoint_base, exist_ok=True)
    custom_name = f"{model_name}_model_dlr_newUNETREST.pth"
    # Final model save path
    model_save_path = os.path.join(checkpoint_base, custom_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    evaluate(model, test_loader, device, writer=writer)
    visualisation.visualize_prediction(model, test_loader, device)

# ================================
# Evaluation
# ================================
def evaluate(model, dataloader, device, epoch=None, writer=None):
    model.eval()
    iou_macro = MulticlassJaccardIndex(num_classes=N_CLASSES, average='macro').to(device)
    iou_per_class = MulticlassJaccardIndex(num_classes=N_CLASSES, average=None).to(device)
    accuracy = MulticlassAccuracy(num_classes=N_CLASSES, average='macro').to(device)
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
            preds = preds.view(-1)
            masks = masks.view(-1)

            valid = masks != 255
            preds = preds[valid]
            masks = masks[valid]

            assert preds.min() >= 0 and preds.max() < 20, f"Bad preds: {preds.unique()}"
            assert masks.min() >= 0 and masks.max() < 20, f"Bad masks: {masks.unique()}"

            # Incremental metric updates (no memory explosion)
            iou_macro.update(preds, masks)
            iou_per_class.update(preds, masks)
            accuracy.update(preds, masks)

    # Compute final metrics
    miou = iou_macro.compute()
    per_class_ious = iou_per_class.compute()
    acc = accuracy.compute()

    print(f"\n✓ Mean IoU: {miou:.4f}")
    for i, class_iou in enumerate(per_class_ious):
        print(f"  └─ Class {i:02d} IoU: {class_iou:.4f}")

    # Write to TensorBoard if available
    if writer and epoch is not None:
        writer.add_scalar("IoU/Mean", miou, epoch)
        writer.add_scalar("Accuracy/Mean", acc, epoch)
        for i, val in enumerate(per_class_ious):
            writer.add_scalar(f"IoU/Class_{i}", val, epoch)

    # Reset metrics after evaluation (optional if reused)
    iou_macro.reset()
    iou_per_class.reset()

    return miou


def visualize_val_predictions(model, val_loader, device, epoch, processor=None, writer=None, num_samples=3):
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)

            # Handle HuggingFace models
            if isinstance(model, PreTrainedModel):
                images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
                outputs = model(**inputs).logits
            else:
                outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = torch.argmax(outputs, dim=1).cpu()

            # Visualize only the first batch and limited number of samples
            samples = [(images[j].cpu(), masks[j].cpu(), preds[j]) for j in range(min(num_samples, len(images)))]
            visualisation.visualize_prediction(model, samples, epoch=epoch)

            # Optional: add to TensorBoard
            if writer:
                writer.add_images("Samples/Input", images[:num_samples], epoch)
                writer.add_images("Samples/GT", masks[:num_samples].unsqueeze(1).float() / N_CLASSES, epoch)
                writer.add_images("Samples/Prediction", preds[:num_samples].unsqueeze(1).float() / N_CLASSES, epoch)

            break

#=======================================
# main function
#=======================================

def main():
    """parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["flair", "dlr"], default="dlr")
    args = parser.parse_args()"""
    dataset_name = "ResUnet50"
    """dataset_choice = args.dataset or dataset_name"""

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "flair":
        train_dataset, val_dataset, test_dataset= prepare_datasets_from_csvs(
            train_csv_path=TRAIN_CSV_PATH,
            val_csv_path= VAL_CSV_PATH,
            test_csv_path= TEST_CSV_PATH,
            base_dir=BASE_DIR)
    else:  # fallback to DLR dataset
         train_dataset,val_dataset, test_dataset = dlr_preprocessing.load_data_dlr(ROOT_DIR, dataset_type="SS_Dense")

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
        train_and_evaluate("unet_resnet50", train_dataset, val_dataset, test_dataset, device, writer)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        writer.close()

# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
