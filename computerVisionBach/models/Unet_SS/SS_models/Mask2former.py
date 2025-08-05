from typing import Optional
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
from loguru import logger
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
)
device = torch.device or ("cuda" if torch.cuda.is_available() else "cpu")
class Mask2FormerModel:
    """Wrapper for Mask2Former model to handle initialization, training, evaluation, and inference."""

    def __init__(self,
                 model_name: str = "facebook/mask2former-swin-small-ade-semantic",
                 num_classes: int = 20,
                 class_names: Optional[list] = None,
                 ):
        logger.info("Loading image processor and model...")

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            reduce_labels=False,
            do_rescale=False
        )

        if class_names is None:
            logger.warning("No class_names provided. Defaulting to Class_0 to Class_{n-1}")
            class_names = [f"Class_{i}" for i in range(num_classes)]

        self.id2label = {i: name for i, name in enumerate(class_names)}
        self.label2id = {name: i for i, name in self.id2label.items()}

        logger.debug(f"id2label mapping: {self.id2label}")

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        # Freeze embedding + Swin stages 0 and 1
        """self.freeze()
        logger.info("✅ Frozen patch embedding and Swin stages 0 and 1.")

        # Reinitialize decoder
        self.reinit_decoder_official(self.model)
        logger.info("Decoder parameters reinitialized.")

        self.backbone_frozen = False"""
        logger.info("Model and processor initialized successfully.")

    def freeze(self):
        logger.info("Freezing patch embedding and first two Swin stages (stage 0 and 1)...")

        for name, param in self.model.named_parameters():
            if (
                    name.startswith("model.backbone.patch_embed") or
                    name.startswith("model.backbone.stages.0") or
                    name.startswith("model.backbone.stages.1")
            ):
                param.requires_grad = False
                logger.debug(f"Froze parameter: {name}")

        self.backbone_frozen = True
        logger.info("Freezing completed.")

    def reinit_decoder_official(self, model):
        xavier_std = model.config.init_xavier_std
        std = model.config.init_std

        decoder = model.model.pixel_level_module.decoder

        logger.info("Reinitializing decoder via official Mask2Former _init_weights...")

        for module in decoder.modules():
            if isinstance(module, Mask2FormerPixelDecoder):
                for p in module.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p, gain=xavier_std)
                nn.init.normal_(module.level_embed, mean=0.0, std=std)
                logger.info(" • PixelDecoder ⟶ Xavier on params + normal on level_embed")

            elif isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
                logger.info(f" • {type(module).__name__} ⟶ normal(mean=0,std={std}) + zero bias")

            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
                logger.info(" • Embedding ⟶ normal(mean=0,std={std}), zero padding_idx")

            if hasattr(module, "reference_points"):
                nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
                nn.init.constant_(module.reference_points.bias.data, 0.0)
                logger.info(" • reference_points ⟶ Xavier + zero bias")

    def train_model(self, train_loader, val_loader, epochs, lr, device=None, tensorboard_writer=None):
        self.model.to(device)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        best_miou = 0.0
        best_model_state = None
        best_epoch = -1

        try:
            for epoch in range(1, epochs + 1):
                self._set_model_mode(train=True)
                total_loss = 0.0

                logger.info(f"Starting epoch {epoch}/{epochs}...")

                for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
                    try:
                        loss = self._process_batch(batch, device, optimizer, batch_idx, epoch)
                        total_loss += loss
                        logger.info(f"[Epoch {epoch}] Batch {batch_idx + 1}/{len(train_loader)} processed.")
                    except Exception as e:
                        logger.warning(f"Error processing batch {batch_idx + 1}: {e}")
                        torch.cuda.empty_cache()

                val_miou = self._log_epoch_results(epoch, total_loss, len(train_loader), val_loader, tensorboard_writer)

                if val_miou > best_miou:
                    best_miou = val_miou
                    best_model_state = self.model.state_dict(self.model)
                    best_epoch = epoch
                    logger.info(f"[Epoch {epoch}] New best mIoU: {val_miou:.4f} — model snapshot saved in memory.")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (KeyboardInterrupt). Cleaning up...")

        if best_model_state is not None:
            save_path = f"best_model_epoch{best_epoch}_miou{best_miou:.4f}.pth"
            torch.save(best_model_state, save_path)
            logger.info(f"Best model saved to: {save_path} with mIoU = {best_miou:.4f}")

        return self.model

    def _set_model_mode(self, train=True):
        self.model.train() if train else self.model.eval()
        if train and self.backbone_frozen and hasattr(self.model, "model") and hasattr(self.model.model, "backbone"):
            self.model.model.backbone.eval()

    def _process_batch(self, batch, device, optimizer, batch_idx, epoch):
        images, masks = batch
        images = images.to(device)
        masks= masks.to(device)

        if batch_idx == 0 and epoch == 1:
            logger.info(f"[Sanity] Image range: {images.min().item()} - {images.max().item()}, dtype: {images.dtype}")
            logger.info(f"[Sanity] Mask unique: {torch.unique(masks)}")

        print("Model device:", next(self.model.parameters()).device)
        outputs = self.model(pixel_values=images, mask_labels=masks)
        loss = outputs.loss

        if torch.isnan(loss):
            logger.error(f"[FATAL] Loss is NaN at Epoch {epoch} Batch {batch_idx}. Exiting this batch.")
            raise ValueError("Loss became NaN")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    def _log_epoch_results(self, epoch, total_loss, num_batches, val_loader, writer):
        avg_loss = total_loss / num_batches
        val_miou, per_class_iou = self.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val mIoU = {val_miou:.4f}, Per-class IoU: {per_class_iou}")

        if writer:
            writer.add_scalar("Loss/Total", avg_loss, epoch)
            writer.add_scalar("IoU/val", val_miou, epoch)
            for idx, iou in enumerate(per_class_iou):
                class_name = self.id2label.get(idx, f"Class_{idx}")
                writer.add_scalar(f"IoU/{class_name}", iou, epoch)

        return val_miou

    @torch.no_grad()
    def evaluate(self, data_loader, device, epoch, writer=None):
        logger.info("Evaluating model...")
        self.model.eval()
        num_classes = self.model.config.num_labels if hasattr(self.model, "config") else 6

        # Instantiate metric
        iou_macro = MulticlassJaccardIndex(num_classes=num_classes, average="macro", ignore_index=255).to(device)
        iou_per_class = MulticlassJaccardIndex(num_classes=num_classes, average=None, ignore_index=255).to(device)
        accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)

        for batch_idx, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            logger.info(f"Evaluating batch {batch_idx + 1}/{len(data_loader)}")
            try:
                images, masks = images.to(device), masks.to(device)
                outputs = self.model(pixel_values=images)
                preds = outputs.logits.argmax(dim=1)  # shape: (B, H, W)

                iou_macro.update(preds, masks)
                iou_per_class.update(preds, masks)
                accuracy.update(preds, masks)

            except Exception as e:
                logger.warning(f"Error during evaluation: {e}")
                torch.cuda.empty_cache()

        miou = iou_macro.compute()
        per_class_ious = iou_per_class.compute()
        acc = accuracy.compute()
        # Compute mean IoU (returns a scalar, unless you set average=None)
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

    def _process_eval_batch(self, images, masks, num_classes, device):
        # Move images/masks to device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = self.model(pixel_values=images)
        seg_maps = outputs.logits.argmax(dim=1)

        total_intersection = np.zeros(num_classes, dtype=np.int64)
        total_union = np.zeros(num_classes, dtype=np.int64)

        # Detach and move to cpu for numpy ops
        seg_maps = seg_maps.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for i, (pred_mask, true_mask) in enumerate(zip(seg_maps, masks_np)):
            pred_arr = pred_mask.astype(np.uint8)
            true_arr = true_mask.astype(np.uint8)
            intersection, union = Mask2FormerModel.compute_iou(pred_arr, true_arr, num_classes)
            total_intersection += intersection
            total_union += union

        return total_intersection, total_union

    @torch.no_grad()
    def predict(self, image, device=None):
        logger.info("Generating prediction...")
        self.model.eval()

        # If image is numpy array, convert to torch tensor (assume it's already normalized)
        if isinstance(image, np.ndarray):
            if image.ndim == 3:  # (H, W, C)
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = image.unsqueeze(0) if image.ndim == 3 else image  # add batch dim if missing
        elif isinstance(image, torch.Tensor) and image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        outputs = self.model(pixel_values=image)
        pred_mask = outputs.logits.argmax(dim=1)  # (B, H, W)

        # Convert to numpy array, squeeze batch dimension
        return pred_mask[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def calculate_mean_iou(intersection, union):
        ious = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        return mean_iou, ious

    @staticmethod
    def compute_iou(pred, true, num_classes):
        intersection = np.zeros(num_classes, dtype=np.int64)
        union = np.zeros(num_classes, dtype=np.int64)

        for cls in range(num_classes):
            pred_mask = (pred == cls)
            true_mask = (true == cls)

            intersection[cls] = np.logical_and(pred_mask, true_mask).sum()
            union[cls] = np.logical_or(pred_mask, true_mask).sum()

        return intersection, union
    @staticmethod
    def unnormalize_img(img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
        return (img * std + mean).clamp(0, 1)

