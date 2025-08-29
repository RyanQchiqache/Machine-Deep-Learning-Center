# ===== Unified Model Loading & Inference (training-parity HF configs) =====
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)

# Your SMP import
from computerVisionBach.models.model_pipeline import smp

# ---------- Config ----------
IMAGENET_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# HF defaults aligned with your training builders
HF_DEFAULTS = {
    "segformer": "nvidia/segformer-b2-finetuned-ade-512-512",
    "upernet":   "openmmlab/upernet-swin-small",
    "mask2former": "facebook/mask2former-swin-small-ade-semantic",
}

# Default processor kwargs (training parity)
HF_PROC_DEFAULTS = {
    "segformer": dict(reduce_labels=False, do_rescale=True, size={"height": 512, "width": 512}),
    "upernet":   dict(reduce_labels=False, do_rescale=True),  # size not required
    "mask2former": dict(reduce_labels=False, do_rescale=True),
}

# ---------- Data structures ----------
@dataclass
class ProcessorSpec:
    name_or_path: str
    reduce_labels: bool = False
    do_rescale: bool = True
    size: Optional[Dict[str, int]] = None  # e.g., {"height":512, "width":512}

    def cache_key(self) -> Tuple[Any, ...]:
        # hashable key for caching
        size_key = None if self.size is None else (self.size.get("height"), self.size.get("width"))
        return (self.name_or_path, self.reduce_labels, self.do_rescale, size_key)

@dataclass
class ModelBundle:
    name: str                  # "unet", "deeplabv3+", "segformer", "upernet", "mask2former", "unet_resnet"
    model: torch.nn.Module
    num_classes: int
    in_channels: int
    expects_5ch: bool
    is_hf: bool                # whether to use HF processors
    processor_spec: Optional[ProcessorSpec] = None  # HF processor construction spec

# ---------- Small utilities ----------
def _safe_rgb_uint8(np_patch: np.ndarray) -> np.ndarray:
    """Ensure (H,W,3) uint8 in [0,255]. If more than 3 channels, keep first 3."""
    if np_patch.ndim != 3:
        raise ValueError(f"Expected HWC, got {np_patch.shape}")
    h, w, c = np_patch.shape
    if c < 3:
        raise ValueError("HF models require 3-channel RGB input.")
    if c > 3:
        np_patch = np_patch[:, :, :3]
    if np_patch.dtype != np.uint8:
        mx = float(np.max(np_patch)) if np_patch.size else 1.0
        if mx <= 1.0:
            np_patch = (np_patch * 255.0).astype(np.uint8)
        else:
            np_patch = np.clip(np_patch, 0, 255).astype(np.uint8)
    return np_patch

def _m2f_postprocess_semantic(processor, outputs, H: int, W: int) -> np.ndarray:
    """Mask2Former post-processing with version compatibility."""
    if hasattr(processor, "post_process_semantic"):
        seg = processor.post_process_semantic(outputs, target_sizes=[(H, W)])[0]["segmentation"]
    elif hasattr(processor, "post_process_semantic_segmentation"):
        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
    else:
        raise AttributeError("Processor lacks semantic post-processing method.")
    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    return seg.astype(np.int64)

# ---------- Model factories ----------
def _build_smp_unet(num_classes: int, in_channels: int, encoder_name: str = "resnet50"):
    return smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet",
                    in_channels=in_channels, classes=num_classes)

def _build_smp_deeplab(num_classes: int, in_channels: int, encoder_name: str = "resnet50"):
    return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights="imagenet",
                             in_channels=in_channels, classes=num_classes)

def _build_hf(model_name: str,
              num_classes: int,
              ckpt_or_hub: Optional[str] = None,
              class_names: Optional[list] = None,
              ignore_index: int = 255):
    """
    Mirrors your training builders: constructs model and returns (model, ProcessorSpec).
    """
    hub = ckpt_or_hub or HF_DEFAULTS[model_name]
    proc_kwargs = HF_PROC_DEFAULTS.get(model_name, {}).copy()
    id2label = {i: (class_names[i] if class_names else f"Class_{i}") for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    if model_name == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained(
            hub, num_labels=num_classes, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True
        )
    elif model_name == "upernet":
        model = UperNetForSemanticSegmentation.from_pretrained(
            hub, num_labels=num_classes, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True
        )
    elif model_name == "mask2former":
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            hub, num_labels=num_classes, id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unsupported HF model: {model_name}")

    model.config.ignore_index = ignore_index

    proc_spec = ProcessorSpec(
        name_or_path=hub,
        reduce_labels=bool(proc_kwargs.get("reduce_labels", False)),
        do_rescale=bool(proc_kwargs.get("do_rescale", True)),
        size=proc_kwargs.get("size"),
    )
    return model, proc_spec

# ---------- Checkpoint loading for SMP ----------
def _load_smp_checkpoint_if_any(model: torch.nn.Module,
                                ckpt_path: Optional[str],
                                num_classes: int,
                                in_channels: int) -> None:
    """
    Loads a local .pth if provided. Adapts head (classes) and first conv (in_channels) as needed.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    sd = torch.load(ckpt_path, map_location="cpu")

    # Drop head if class count mismatches
    head_w = sd.get("segmentation_head.0.weight")
    if head_w is not None and head_w.shape[0] != num_classes:
        sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head.")}

    # Adapt first conv if in_channels mismatch (RGB+IR+Elev, etc.)
    conv_keys = [k for k in sd.keys() if k.endswith("encoder.conv1.weight")]
    if conv_keys:
        conv_key = conv_keys[0]
        w = sd[conv_key]  # [out_c, in_c_ckpt, k, k]
        in_c_ckpt = w.shape[1]
        if in_c_ckpt != in_channels:
            if in_channels < in_c_ckpt:
                sd[conv_key] = w[:, :in_channels, :, :]
            else:
                reps = in_channels - in_c_ckpt
                extra = w[:, :1, :, :].repeat(1, reps, 1, 1)
                sd[conv_key] = torch.cat([w, extra], dim=1)

    model.load_state_dict(sd, strict=False)

# ---------- Public: load_model ----------
def load_model(model_name: str,
               num_classes: int,
               in_channels: int = 3,
               encoder_name: Optional[str] = None,
               ckpt_path: Optional[str] = None,
               device: Optional[torch.device] = None,
               # HF parity knobs (optional; defaults mirror training)
               class_names: Optional[list] = None,
               ignore_index: int = 255) -> ModelBundle:
    """
    Load any supported model with desired num_classes/in_channels and an optional checkpoint.
    - SMP: local .pth via ckpt_path
    - HF: hub id or local directory via ckpt_path (falls back to defaults)
    """
    model_key = model_name.lower()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_key in {"unet", "unet_resnet"}:
        enc = encoder_name or "resnet50"
        model = _build_smp_unet(num_classes, in_channels, encoder_name=enc)
        _load_smp_checkpoint_if_any(model, ckpt_path, num_classes, in_channels)
        return ModelBundle(
            name="unet_resnet" if model_key == "unet_resnet" else "unet",
            model=model.to(device).eval(),
            num_classes=num_classes,
            in_channels=in_channels,
            expects_5ch=(in_channels == 5),
            is_hf=False,
        )

    if model_key == "deeplabv3+":
        enc = encoder_name or "resnet50"
        model = _build_smp_deeplab(num_classes, in_channels, encoder_name=enc)
        _load_smp_checkpoint_if_any(model, ckpt_path, num_classes, in_channels)
        return ModelBundle(
            name="deeplabv3+",
            model=model.to(device).eval(),
            num_classes=num_classes,
            in_channels=in_channels,
            expects_5ch=(in_channels == 5),
            is_hf=False,
        )

    if model_key in {"segformer", "upernet", "mask2former"}:
        model, proc_spec = _build_hf(model_key, num_classes, ckpt_or_hub=ckpt_path,
                                     class_names=class_names, ignore_index=ignore_index)
        return ModelBundle(
            name=model_key,
            model=model.to(device).eval(),
            num_classes=num_classes,
            in_channels=3,            # HF models are RGB-only
            expects_5ch=False,
            is_hf=True,
            processor_spec=proc_spec,
        )

    raise ValueError(f"Unsupported model: {model_name}")

# ---------- Processor cache (by spec) ----------
_PROCESSOR_CACHE: Dict[Tuple[Any, ...], AutoImageProcessor] = {}

def _get_processor(spec: ProcessorSpec):
    key = spec.cache_key()
    if key not in _PROCESSOR_CACHE:
        kwargs = dict(reduce_labels=spec.reduce_labels, do_rescale=spec.do_rescale)
        if spec.size is not None:
            kwargs["size"] = spec.size
        _PROCESSOR_CACHE[key] = AutoImageProcessor.from_pretrained(spec.name_or_path, **kwargs)
    return _PROCESSOR_CACHE[key]

# ---------- Public: predict one patch ----------
def predict_patch(np_patch: np.ndarray,
                  bundle: ModelBundle,
                  device: Optional[torch.device] = None) -> np.ndarray:
    """
    Run inference on a single HWC patch and return (H,W) int64 class map.
    Works for SMP and HF (SegFormer/UPerNet/Mask2Former).
    """
    device = device or next(bundle.model.parameters()).device
    H, W = np_patch.shape[:2]

    # HF path (RGB only; processor handles normalization/resizing)
    if bundle.is_hf:
        rgb = _safe_rgb_uint8(np_patch)
        processor = _get_processor(bundle.processor_spec)

        with torch.no_grad():
            inputs = processor(images=[rgb], return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            outputs = bundle.model(**inputs)

            if bundle.name == "mask2former":
                return _m2f_postprocess_semantic(processor, outputs, H, W)

            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("HF model outputs missing `.logits` (non-Mask2Former).")
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
            return pred.astype(np.int64)

    # SMP path (BCHW float; ImageNet normalize for 3ch)
    with torch.no_grad():
        x = torch.tensor(np_patch.transpose(2, 0, 1), dtype=torch.float32)
        x = x / 255.0 if x.max() > 1.0 else x
        if x.shape[0] == 3:  # only normalize RGB
            x = IMAGENET_NORMALIZE(x)
        x = x.unsqueeze(0).to(device)

        logits = bundle.model(x)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
        return pred.astype(np.int64)
