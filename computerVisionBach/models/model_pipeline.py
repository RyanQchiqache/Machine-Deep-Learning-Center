import copy
import os
import sys
from typing import List, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from torch.utils.data import DataLoader
import torch
from loguru import logger
import torch.nn as nn
from datetime import datetime
from segmentation_models_pytorch.losses import DiceLoss
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from tqdm import tqdm
import segmentation_models_pytorch as smp
from computerVisionBach.models.Unet_SS.SS_models.Unet import UNet
from computerVisionBach.models.Unet_SS import visualisation
from torch.utils.tensorboard import SummaryWriter
import transformers
print("Transformers version:", transformers.__version__)
from transformers import SegformerForSemanticSegmentation, UperNetForSemanticSegmentation, SegformerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers.modeling_utils import PreTrainedModel
from computerVisionBach.models.Unet_SS.preprocessing.flair_preprocessing import prepare_datasets_from_csvs
from computerVisionBach.models.Unet_SS.preprocessing import dlr_preprocessing
from computerVisionBach.models.Unet_SS.SS_models.Mask2former import Mask2FormerModel
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")


# ================================
# Configuration
# ================================
MODEL_NAME =""
PATCH_SIZE = 512
OVERLAP = 0.5
ROOT_DIR = '/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/DLR_dataset'
N_CLASSES = 19
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
PATIENCE = 10
patchify_enabled = True
NUM_RECONSTRUCTIONS = 4
# FLAIR paths
BASE_DIR = "/home/ryqc/data/flair_dataset"
TRAIN_CSV_PATH =  "/home/ryqc/data/flair_dataset/cleaned-train01.csv"
TEST_CSV_PATH = "/home/ryqc/data/flair_dataset/cleaned-test01.csv"
VAL_CSV_PATH = "/home/ryqc/data/flair_dataset/cleaned-test01.csv"
FLAIR_USED_LABELS = [1, 2, 3, 6, 7, 8, 10, 11, 13, 18]
class_names = [
    "Low vegetation",           # class 1
    "Paved road",               # class 2
    "Non paved road",           # class 3
    "Paved parking place",      # class 4
    "Non paved parking place",  # class 5
    "Bikeways",                 # class 6
    "Sidewalks",                # class 7
    "Entrance exit",            # class 8
    "Danger area",              # class 9
    "Lane-markings",            # class 10
    "Building",                 # class 11
    "Car",                      # class 12
    "Trailer",                  # class 13
    "Van",                      # class 14
    "Truck",                    # class 15
    "Long truck",               # class 16
    "Bus",                      # class 17
    "Clutter",                  # class 18
    "Impervious surface",       # class 19
    "Tree",                     # class 20
]


#=====================================
# Mask2former initialisation
#====================================
def load_mask2former_model(model_name: str, num_classes: int, class_names=None):
    processor = Mask2FormerImageProcessor.from_pretrained(
        model_name,
        reduce_labels=False,
        do_rescale=False
    )

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.config.ignore_index = 255

    return model, processor


def get_loss_and_optimizer(model):
    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    #criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max", patience=5, factor=0.5, verbose=True)

    return ce_loss, scheduler, optimizer


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    is_mask2former = isinstance(model, Mask2FormerForUniversalSegmentation)

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        """if isinstance(model, PreTrainedModel):
            # Convert to numpy and preprocess
            images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
            inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs).logits
        else:"""
        optimizer.zero_grad()

        if is_mask2former:
            batch_inputs = processor(images=images, segmentation_maps=masks, return_tensors="pt")
            batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else [i.to(device) for i in v]) for k, v in
                            batch_inputs.items()}
            print(**batch_inputs)
            outputs = model(**batch_inputs)
            loss = outputs.loss
            loss = getattr(outputs, "loss", None)
            if loss is None:
                print("Warning: outputs.loss is None. Skipping this batch.")
                continue
        else:
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = torch.nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# ================================
# Training Loop
# ================================
def train(model, train_loader,val_loader, criterion, optimizer, scheduler, device, num_epochs, writer=None):
    best_miou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        loss = train_one_epoch(model,train_loader, criterion, optimizer, device)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f}")

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)

        model.eval()
        with torch.no_grad():
            miou = evaluate(model, val_loader, device ,epoch=epoch, writer=writer)

        scheduler.step(miou.item())

        #visualize_val_predictions(model, val_loader, device, epoch, processor=processor, writer=writer)
        if miou > best_miou:
            best_miou = miou
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            logger.info((f"mIoU improved to {best_miou:.4f}, saving model weights."))
            """if (epoch + 1) % 10 == 0 or epoch == 1:
                val_image = test_dataset[0][0].unsqueeze(0).to(device)
                _, features = model(val_image, return_features=True)
                for name in ["bottleneck", "enc1", "enc2", "dec3", "dec4"]:
                    visualisation.visualise_feature_map(features[name], f"{name} (Epoch {epoch + 1})")"""
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement in mIoU for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= PATIENCE:
            logger.info(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    model.load_state_dict(best_model_wts)
    return model

def train_and_evaluate(model_name, train_dataset, val_dataset, test_dataset, device, writer=None):
    logger.info(f"\nTraining model: {model_name}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    # Just after fetching a batch in your DataLoader loop:
    images, masks = next(iter(train_loader))
    print("Image range:", images.min().item(), images.max().item())  # Should be around 0–1
    print("Mask unique:", masks.unique())

    if model_name.lower() == "mask2former":
        logger.info("Running Mask2Former with direct Hugging Face model...")
        model, processor = load_mask2former_model(
            model_name="facebook/mask2former-swin-small-ade-semantic",
            num_classes=N_CLASSES,
            class_names=class_names
        )
        print("Model class:", type(model))
        print("Ignore index used in model:", model.config.ignore_index)
        model = model.to(device)

    elif model_name.lower() == "unet":
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
            "openmmlab/upernet-swin-small",#"openmmlab/upernet-convnext-small"
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

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model = train(
        model=model,
        #processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        writer=writer
    )
    checkpoint_base = os.path.join(os.path.dirname(__file__), "Unet_SS/checkpoints")
    os.makedirs(checkpoint_base, exist_ok=True)
    custom_name = f"{model_name}_model_flair_deeplabv3+.pth"
    # Final model save path
    model_save_path = os.path.join(checkpoint_base, custom_name)
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"✅ Model saved to {model_save_path}")

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
    is_mask2former = isinstance(model, Mask2FormerForUniversalSegmentation)
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            # HuggingFace transformer models
            """if isinstance(model, PreTrainedModel):
                images_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
                inputs = processor(images=images_np, return_tensors="pt", do_rescale=False).to(device)
                output = model(**inputs)
                logits = output.logits
            else:"""
            if is_mask2former:
                batch_inputs = processor(images=images, return_tensors="pt")
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                logits = model(**batch_inputs)
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

    logger.info(f"\n✓ Mean IoU: {miou:.4f}")
    for i, class_iou in enumerate(per_class_ious):
        logger.info(f"  └─ Class {i:02d} IoU: {class_iou:.4f}")

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
    dataset_name = "flair"
    """dataset_choice = args.dataset or dataset_name"""

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name == "flair":
        train_dataset, val_dataset, test_dataset= prepare_datasets_from_csvs(
            train_csv_path=TRAIN_CSV_PATH,
            val_csv_path= VAL_CSV_PATH,
            base_dir=BASE_DIR)
    else:  # fallback to DLR dataset
         train_dataset,val_dataset, test_dataset = dlr_preprocessing.load_data_dlr(ROOT_DIR, dataset_type="SS_Dense", model_name="Mask2former")

    """
    logger.info("Checking FLAIR label range...")
    all_labels = torch.cat([train_dataset[i][1].flatten() for i in range(20)])  # Sample 20 masks
    logger.info("Unique labels in train set:", torch.unique(all_labels))
    logger.info("N_CLASSES =", N_CLASSES)"""

    logger.info(f"Dataset chosen is : {dataset_name}")
    logger.info(f"\nTrain samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    """sample_image, sample_mask = train_dataset[0]
    logger.info("Label range (first mask):", sample_mask.min().item(), "to", sample_mask.max().item())
    logger.info("Unique labels (first mask):", torch.unique(sample_mask).tolist())"""

    log_dir = f"runs/{dataset_name}_experiment_flair_deeplabv3+{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)

    try:
        train_and_evaluate("deeplabv3+", train_dataset, val_dataset, test_dataset, device, writer)
    except KeyboardInterrupt:
        logger.info("Training interrupted manually.")
    finally:
        writer.close()

# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
