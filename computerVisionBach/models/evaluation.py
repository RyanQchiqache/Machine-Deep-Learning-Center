import time
from tqdm import tqdm
from typing import Dict, Iterable, Optional, Sequence
import logging
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy


def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    epoch: Optional[int] = None,
    writer: Optional["SummaryWriter"] = None,
    *,
    num_classes: int,
    ignore_index: int = 255,
    rare_class_ids: Optional[Sequence[int]] = None,
    processor: Optional[object] = None,  # HF processor for Mask2Former/SegFormer, etc.
    boundary_tolerance_px: int = 3,
    log_prefix: str = "Val",
) -> Dict[str, float]:
    """
    evaluation:
    - mIoU (macro), per-class IoU
    - Overall Accuracy (micro) + Accuracy (macro)
    - Boundary-F1 with N-pixel tolerance (default 3px)
    - Rare-class mIoU (subset of class IDs)
    - Efficiency: Params (M), Peak VRAM (GB), Inference Latency (ms/img)

    Returns a dict with all summary metrics. Also logs scalars to TensorBoard if writer is provided.

    Notes:
    - Uses torchmetrics with ignore_index for consistent denominators.
    - Works for SMP models (expects logits [B, C, H, W]) and HF models via `processor`.
    - Measures latency as average forward-pass time per image (including HF preprocessing if used).
    """

    model.eval()
    is_cuda = device.type == "cuda"
    # -------- Efficiency bookkeeping
    # Params (trainable only, in millions)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # Peak VRAM: reset before loop and capture after
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # -------- Torchmetrics (DDP-safe if you configure torchmetrics accordingly in your setup)
    iou_macro = MulticlassJaccardIndex(
        num_classes=num_classes, average="macro", ignore_index=ignore_index
    ).to(device)
    iou_per_class = MulticlassJaccardIndex(
        num_classes=num_classes, average=None, ignore_index=ignore_index
    ).to(device)
    acc_macro = MulticlassAccuracy(
        num_classes=num_classes, average="macro", ignore_index=ignore_index
    ).to(device)
    acc_micro = MulticlassAccuracy(
        num_classes=num_classes, average="micro", ignore_index=ignore_index
    ).to(device)

    # Boundary-F1 accumulators (TP/FP/FN over the whole dataset)
    bf1_tp = 0
    bf1_fp = 0
    bf1_fn = 0

    # Rare-class accumulators (compute per-batch IoU for rare subset, then average)
    have_rare = rare_class_ids is not None and len(rare_class_ids) > 0
    rare_ids = torch.tensor(rare_class_ids, device=device, dtype=torch.long) if have_rare else None
    rare_iou_sum = 0.0
    rare_batches = 0

    total_images = 0
    total_forward_time_s = 0.0

    import numpy as np

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _pred_any(images: torch.Tensor, out_size_hw=None):
        """
        Returns either:
          - logits [B, C, h, w]  (SegFormer / UPerNet)
          - labels [B, H, W]     (Mask2Former-universal post-processed)
        """
        if processor is None:
            return model(images)

        # If your dataset normalized tensors, de-normalize to [0,1] for HF processor
        imgs = images
        if imgs.min() < 0 or imgs.max() > 1:
            imgs = imgs * IMAGENET_STD + IMAGENET_MEAN
        imgs = (imgs.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # N,H,W,C uint8

        # Build inputs (no segmentation_maps during eval)
        inputs = processor(images=list(imgs), return_tensors="pt")
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        outputs = model(**inputs)

        # If model exposes per-pixel logits (e.g., SegFormer/UPerNet), use them
        if hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits  # [B, C, h, w]

        # Mask2Former-universal: post-process to semantic label maps [H,W] per image
        target_sizes = [out_size_hw] * inputs["pixel_values"].shape[0] if out_size_hw is not None else None
        if hasattr(processor, "post_process_semantic_segmentation"):
            sem = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        else:
            # older name in some versions
            sem = processor.post_process_semantic(outputs, target_sizes=target_sizes)

        # sem is a list of tensors or numpy arrays [H,W]; stack to [B,H,W] on device
        sem = [torch.as_tensor(s, device=device, dtype=torch.long) for s in sem]
        return torch.stack(sem, dim=0)  # [B, H, W]

    def _make_boundary_map(mask: torch.Tensor) -> torch.Tensor:
        """
        mask: [H, W] integer labels, ignore_index possible.
        Returns a boolean tensor of boundary pixels (union across all classes except ignore).
        Method: 4-neighborhood gradient -> edges; then clean ignore regions.
        """
        # Exclude ignore regions by temporarily setting them to neighbor-safe value
        mi = mask.clone()
        ignore = (mi == ignore_index)
        # For diffing, fill ignore with a unique negative that won't match neighbors
        mi[ignore] = -1

        # 4-neighborhood differences
        dh = torch.zeros_like(mi, dtype=torch.bool)
        dv = torch.zeros_like(mi, dtype=torch.bool)
        dh[:, 1:] = mi[:, 1:] != mi[:, :-1]
        dv[1:, :] = mi[1:, :] != mi[:-1, :]
        edges = dh | dv

        # Remove edges that are just transitions to ignore
        # Reconstruct where ignore touches; mark those edges off
        # Any pixel that is ignore and has a non-ignore neighbor would create an edge; drop them
        # Build neighbor ignore maps
        ign_right = torch.zeros_like(ignore)
        ign_right[:, :-1] = ignore[:, 1:]
        ign_down = torch.zeros_like(ignore)
        ign_down[:-1, :] = ignore[1:, :]
        # Edges adjacent to ignore → remove
        edges = edges & ~(ignore | ign_right | ign_down)

        return edges

    def _dilate_bool(mask_bool: torch.Tensor, r: int) -> torch.Tensor:
        """Binary dilation via max-pool (square structuring element)."""
        if r <= 0:
            return mask_bool
        # Convert to float for pooling
        x = mask_bool.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        k = 2 * r + 1
        pad = r
        y = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return (y.squeeze(0).squeeze(0) > 0.5)

    @torch.no_grad()
    def _boundary_f1_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tuple[int, int, int]:
        """
        pred, target: [H, W] integer labels with ignore_index.
        Computes TP/FP/FN for boundary detection with tolerance.
        """
        gt_edges = _make_boundary_map(target)
        pr_edges = _make_boundary_map(pred)

        # Tolerant matching (dilate the other side by 'r' pixels)
        gt_dil = _dilate_bool(gt_edges, boundary_tolerance_px)
        pr_dil = _dilate_bool(pr_edges, boundary_tolerance_px)

        # Matches
        tp = (pr_edges & gt_dil).sum().item()
        fp = (pr_edges & ~gt_dil).sum().item()
        fn = (gt_edges & ~pr_dil).sum().item()
        return tp, fp, fn

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluation", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()

            out = _pred_any(images, out_size_hw=masks.shape[-2:])
            if out.dim() == 4:
                logits = out
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds = torch.argmax(logits, dim=1)  # [B,H,W]
            elif out.dim() == 3:
                preds = out.long()  # already per-pixel class IDs [B,H,W]
            else:
                raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")

            # Argmax over classes
            #preds = torch.argmax(logits, dim=1)  # [B, H, W]

            if is_cuda:
                torch.cuda.synchronize(device)
            end = time.perf_counter()

            # Latency accounting
            total_forward_time_s += (end - start)
            total_images += images.size(0)

            # ---- Torchmetrics updates: feed full tensors (metrics handle ignore_index)
            iou_macro.update(preds, masks)
            iou_per_class.update(preds, masks)
            acc_macro.update(preds, masks)
            acc_micro.update(preds, masks)

            # ---- Boundary-F1 accumulation (per image)
            # Compute on each image to avoid huge memory; ignore_index handled internally in helper
            b = preds.size(0)
            for i in range(b):
                tp, fp, fn = _boundary_f1_batch(preds[i], masks[i])
                bf1_tp += tp
                bf1_fp += fp
                bf1_fn += fn

            # ---- Rare-class mIoU (optional)
            if have_rare:
                # Per-class IoU for this batch (vector), then select rare IDs that appear in dataset taxonomy
                batch_iou = MulticlassJaccardIndex(
                    num_classes=num_classes, average=None, ignore_index=ignore_index
                ).to(device)
                batch_iou.update(preds, masks)
                per_cls = batch_iou.compute()  # [C]
                # Filter only valid rare ids within [0, num_classes)
                valid_ids = rare_ids[(rare_ids >= 0) & (rare_ids < num_classes)]
                if valid_ids.numel() > 0:
                    rare_iou = per_cls[valid_ids].mean().item()
                    rare_iou_sum += rare_iou
                    rare_batches += 1
                del batch_iou

    # -------- Compute final metrics
    miou_macro = float(iou_macro.compute().item())
    per_class_ious = iou_per_class.compute()  # tensor [C]
    oa_macro = float(acc_macro.compute().item())
    oa_micro = float(acc_micro.compute().item())

    # Boundary F1
    if (bf1_tp + bf1_fp + bf1_fn) == 0:
        boundary_f1 = 0.0
    else:
        precision = bf1_tp / max(1, (bf1_tp + bf1_fp))
        recall = bf1_tp / max(1, (bf1_tp + bf1_fn))
        boundary_f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    # Rare-class mIoU
    if have_rare and rare_batches > 0:
        miou_rare = float(rare_iou_sum / rare_batches)
    elif have_rare:
        miou_rare = float("nan")  # no rare classes encountered
    else:
        miou_rare = float("nan")

    # Peak VRAM and latency
    if is_cuda:
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_vram_gb = 0.0
    latency_ms_per_img = 1000.0 * (total_forward_time_s / max(1, total_images))

    # -------- Logging
    # Console (brief)
    print(f"\n✓ {log_prefix}: mIoU={miou_macro:.4f} | OA_micro={oa_micro:.4f} | OA_macro={oa_macro:.4f} "
          f"| BF1@{boundary_tolerance_px}px={boundary_f1:.4f} | Rare mIoU={miou_rare:.4f} "
          f"| Params={params_m:.2f}M | PeakVRAM={peak_vram_gb:.2f}GB | Latency={latency_ms_per_img:.1f}ms/img")

    # TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{log_prefix}/mIoU_macro", miou_macro, epoch)
        writer.add_scalar(f"{log_prefix}/OA_micro", oa_micro, epoch)
        writer.add_scalar(f"{log_prefix}/OA_macro", oa_macro, epoch)
        writer.add_scalar(f"{log_prefix}/BoundaryF1_{boundary_tolerance_px}px", boundary_f1, epoch)
        writer.add_scalar(f"{log_prefix}/mIoU_rare", miou_rare, epoch)
        writer.add_scalar(f"{log_prefix}/Params_M", params_m, epoch)
        writer.add_scalar(f"{log_prefix}/PeakVRAM_GB", peak_vram_gb, epoch)
        writer.add_scalar(f"{log_prefix}/Latency_ms_per_img", latency_ms_per_img, epoch)
        # Per-class IoU histograms/scalars
        for k, val in enumerate(per_class_ious.tolist()):
            writer.add_scalar(f"{log_prefix}/IoU_Class_{k}", float(val), epoch)

    logger = logging.getLogger(__name__)

    per_cls = per_class_ious.detach().float().cpu().tolist()
    logger.info(f"{log_prefix} per-class IoUs (epoch {epoch if epoch is not None else '-'})")
    for k, v in enumerate(per_cls):
        if k == ignore_index:
            continue
        if v != v:
            logger.info(f"  └─ Class {k:02d} IoU: nan")
        else:
            logger.info(f"  └─ Class {k:02d} IoU: {v:.4f}")

    # -------- Reset torchmetrics (safe if you reuse evaluator)
    iou_macro.reset()
    iou_per_class.reset()
    acc_macro.reset()
    acc_micro.reset()

    return {
        "mIoU_macro": miou_macro,
        "OA_micro": oa_micro,
        "OA_macro": oa_macro,
        "BoundaryF1_px": float(boundary_tolerance_px),
        "BoundaryF1": float(boundary_f1),
        "mIoU_rare": float(miou_rare),
        "Params_M": float(params_m),
        "PeakVRAM_GB": float(peak_vram_gb),
        "Latency_ms_per_img": float(latency_ms_per_img),
    }
