import copy
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LinearLR
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

import transformers
print("Transformers version:", transformers.__version__)
from transformers import Mask2FormerForUniversalSegmentation, SegformerForSemanticSegmentation, UperNetForSemanticSegmentation


from computerVisionBach.models.Unet_SS import visualisation, utils
from computerVisionBach.models.Unet_SS.preprocessing.flair_preprocessing import prepare_datasets_from_csvs
from computerVisionBach.models.Unet_SS.preprocessing import dlr_preprocessing
from computerVisionBach.models.evaluation import evaluate
from computerVisionBach.models.models_factory import build_model
print(f"Encoders available in smp: {smp.encoders.get_encoder_names()}")

cfg = OmegaConf.load("/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/config/config.yaml")
OmegaConf.resolve(cfg)


scaler = GradScaler()

def get_loss_and_optimizer(model):
    dice_loss = DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_class_index)

    # criterion = lambda pred, target: 0.5 * ce_loss(pred, target) + 0.5 * dice_loss(pred, target)
    criterion = ce_loss  # or use the combined loss above

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=tuple(cfg.training.betas)
    )

    # SCHEDULER: warmup (LinearLR) → cosine (CosineAnnealingLR)
    if cfg.training.scheduler.type.lower() == "cosine":
        warmup_epochs = int(cfg.training.scheduler.warmup_epochs)
        max_epochs = int(cfg.training.scheduler.max_epochs)
        eta_min = float(cfg.training.scheduler.eta_min)
        start_factor = float(cfg.training.scheduler.get("warmup_start_factor", 0.1))

        # Warmup: linearly scale LR from start_factor*base_lr -> base_lr over warmup_epochs
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        # Cosine: anneal from base_lr -> eta_min over the remaining epochs
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=eta_min
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
    else:
        # (fallback) your old plateau scheduler, if you want to keep it configurable
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.scheduler.mode,
            patience=cfg.training.scheduler.patience,
            factor=cfg.training.scheduler.factor,
        )

    return criterion, scheduler, optimizer


def train_one_epoch(model, dataloader, criterion, optimizer, device, processor=None):
    model.train()
    running_loss = 0.0
    model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
    is_mask2former = isinstance(model, Mask2FormerForUniversalSegmentation) or model_type == "mask2former"
    is_hf_semseg = isinstance(model, (SegformerForSemanticSegmentation, UperNetForSemanticSegmentation)) \
                   or model_type in {"segformer", "upernet"}

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        with autocast():
            if is_hf_semseg:
                batch = processor(
                    images=images,
                    segmentation_maps=masks,
                    return_tensors="pt"
                )
                pixel_values = batch["pixel_values"].to(device)  # (B,3,H,W)
                labels_t = batch["labels"].to(device).long()  # (B,H,W)

                # utils.info_images_before_training(images, batch)

                outputs = model(pixel_values=pixel_values, labels=labels_t)
                loss = outputs.loss

            elif is_mask2former:
                batch_inputs = processor(images=list(images),
                                         segmentation_maps=list(masks),
                                         return_tensors="pt")

                # utils.info_images_before_training(images, processor)

                batch_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor)
                                    else [t.to(device) for t in v])
                                for k, v in batch_inputs.items()}
                outputs = model(**batch_inputs)
                loss = outputs.loss

                # Get per-pixel predictions for debugging using postprocessing
                with torch.no_grad():
                    H, W = masks.shape[-2], masks.shape[-1]
                    pp = processor.post_process_semantic_segmentation(
                        outputs=outputs,
                        target_sizes=[(H, W)] * masks.shape[0]
                    )  # list of (H, W) tensors
                    preds = torch.stack(pp, dim=0)  # (B, H, W)
                    # print("Unique preds:", torch.unique(preds))  # optional

                """loss = getattr(outputs, "loss", None)"""
                if loss is None:
                    logger.warning("outputs.loss is None. Skipping this batch.")
                    continue
            else:
                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(dataloader)

# ================================
# Training Loop
# ================================
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer=None, rare_class_ids=None, processor=None):
    best_miou = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    try:
        for epoch in range(num_epochs):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, device, processor=processor)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss:.4f}")
            if writer:
                writer.add_scalar("Loss/train", loss, epoch)

            # choose the right processor for eval
            eval_processor = processor

            # run thesis-grade evaluation (returns dict)
            val_metrics = evaluate(
                model, val_loader, device,
                epoch=epoch, writer=writer,
                num_classes=cfg.model.num_classes, ignore_index=cfg.model.ignore_class_index,
                rare_class_ids=rare_class_ids, processor=eval_processor,
                boundary_tolerance_px=3, log_prefix="Val",
            )

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["mIoU_macro"])
            else:
                scheduler.step()

            # early stopping on mIoU
            if val_metrics["mIoU_macro"] > best_miou:
                best_miou = val_metrics["mIoU_macro"]
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                logger.info(f"mIoU improved to {best_miou:.4f}, saving model weights.")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement in mIoU for {epochs_without_improvement} epochs.")

            if epochs_without_improvement >= cfg.training.patience:
                logger.info(f"Early stopping triggered after {cfg.training.patience} epochs without improvement.")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted, saving best weights so far...")
        ckpt_path = utils.save_checkpoint(model, processor, cfg, best_miou)
        logger.info(f" Model saved to {ckpt_path}")
        raise

    model.load_state_dict(best_model_wts)
    return model, best_miou


def train_and_evaluate(model_name, train_dataset, val_dataset, test_dataset, device, writer=None, rare_class_ids=None, processor=None):
    logger.info(
        f"\nTraining model: {model_name} "
        f"(encoder={cfg.model.smp.encoder_name}, weights={cfg.model.smp.encoder_weights}, "
        f"num_classes={cfg.model.num_classes}, dataset={cfg.project.dataset})"
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    images, masks = next(iter(train_loader))
    # for looking at some label mismatch in 10 iterations for debugging
    """print("Image range:", images.min().item(), images.max().item())
    for i in range(10):
        _, masks = next(iter(train_loader))
        print("Mask unique:", masks.unique())"""
    name = model_name.lower()
    hf_ckpt = {
        "mask2former": cfg.model.hf.checkpoint.mask2former,
        "segformer": cfg.model.hf.checkpoint.segformer,
        "upernet": cfg.model.hf.checkpoint.upernet,
    }.get(name, None)

    if not model_name:
        raise ValueError(
            "cfg.model.name is empty. Set it to one of: "
            "unet, unetpp, deeplabv3plus, fpn, segformer, mask2former, upernet"
        )

    model, processor= build_model(
        model_name=model_name,  # use the function argument
        num_classes=cfg.model.num_classes,
        device=device,

        # SMP knobs (ignored by HF builders)
        encoder_name=cfg.model.smp.encoder_name,
        encoder_weights=cfg.model.smp.encoder_weights,
        in_channels=cfg.model.smp.in_channels,

        # HF knobs (factory must accept & forward these)
        hf_name=hf_ckpt,
        ignore_index=cfg.model.ignore_class_index,
        reduce_labels=cfg.model.hf.processor.reduce_labels,
        do_rescale=cfg.model.hf.processor.do_rescale,
        do_normalize=cfg.model.hf.processor.do_normalize,
        de_resize=cfg.model.hf.processor.do_resize,
        reinit_decoder=cfg.model.hf.options.reinit_decoder,
        freeze_backbone=cfg.model.hf.options.freeze_backbone_stage0_1,

    )

    criterion, scheduler, optimizer = get_loss_and_optimizer(model)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    enc_name = cfg.model.smp.encoder_name
    enc_wts = cfg.model.smp.encoder_weights
    dataset = cfg.project.dataset

    created_writer = False
    if writer is None:
        log_dir = utils.build_log_dir(
            cfg,
            model_name=model_name,
            encoder_name=enc_name,
            encoder_weights=enc_wts,
            dataset_name=dataset,
        )
        writer = SummaryWriter(log_dir=log_dir)
        created_writer = True
        logger.info(f"TensorBoard: {log_dir}")

    model, best_miou = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=cfg.training.num_epochs,
        writer=writer,
        processor=processor,
    )

    ckpt_path = utils.build_checkpoint_path(
        cfg,
        model_name=model_name,
        encoder_name=enc_name,
        encoder_weights=enc_wts,
        dataset_name=dataset,
        epoch="final",
        miou=best_miou
    )
    ckpt_path = utils.save_checkpoint(model, processor, cfg, best_miou)
    logger.info(f" Model saved to {ckpt_path}")

    eval_processor = processor

    test_metrics = evaluate(
        model, test_loader, device, writer=writer, epoch=None,
        num_classes=cfg.model.num_classes, ignore_index=cfg.model.ignore_class_index,
        rare_class_ids=rare_class_ids, processor=eval_processor,
        boundary_tolerance_px=3, log_prefix="Test",
    )
    logger.info(
        f" Test mIoU={test_metrics['mIoU_macro']:.4f} | BF1={test_metrics['BoundaryF1']:.4f} | OA={test_metrics['OA_micro']:.4f}")

    visualisation.visualize_prediction(model, test_loader, device)

    if created_writer:
        writer.close()
#=======================================
# main function
#=======================================

def main():
    """parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["flair", "dlr"], default="dlr")
    args = parser.parse_args()"""
    dataset_name = cfg.project.dataset.lower()
    """dataset_choice = args.dataset or dataset_name"""

    torch.manual_seed(cfg.training.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch:", torch.__version__, "CUDA build:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())

    if dataset_name == "flair":
        train_dataset, val_dataset, test_dataset= prepare_datasets_from_csvs(
            train_csv_path=cfg.data.flair.train_csv,
            val_csv_path= cfg.data.flair.val_csv,
            base_dir=cfg.data.flair.base_dir)
    else:  # fallback to DLR dataset
         train_dataset,val_dataset, test_dataset = dlr_preprocessing.load_data_dlr(cfg.data.dlr.root_dir, dataset_type="SS_Dense", model_name=cfg.model.name)

    if dataset_name.lower() == "flair":
        RARE_CLASS_IDS = cfg.data.flair.rare_class_ids
    else:
        RARE_CLASS_IDS = cfg.data.dlr.rare_class_ids
    """
    logger.info("Checking FLAIR label range...")
    all_labels = torch.cat([train_dataset[i][1].flatten() for i in range(20)])  # Sample 20 masks
    logger.info("Unique labels in train set:", torch.unique(all_labels))
    logger.info("cfg.model.num_classes =", cfg.model.num_classes)"""

    logger.info(f"Dataset chosen is : {dataset_name}")
    logger.info(f"\nTrain samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    """sample_image, sample_mask = train_dataset[0]
    logger.info("Label range (first mask):", sample_mask.min().item(), "to", sample_mask.max().item())
    logger.info("Unique labels (first mask):", torch.unique(sample_mask).tolist())"""

    log_dir = utils.build_log_dir(
        cfg,
        model_name=cfg.model.name,
        encoder_name=cfg.model.smp.encoder_name,
        encoder_weights=cfg.model.smp.encoder_weights,
        dataset_name=cfg.project.dataset,
    )
    writer = SummaryWriter(log_dir=log_dir)

    try:
        train_and_evaluate(cfg.model.name, train_dataset, val_dataset, test_dataset, device, writer, rare_class_ids=RARE_CLASS_IDS)
    except KeyboardInterrupt:
        logger.info("Training interrupted manually.")
    finally:
        writer.close()

# ================================
# Training Script
# ================================
if __name__ == "__main__":
    main()
