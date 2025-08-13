from __future__ import annotations
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import UperNetForSemanticSegmentation, SegformerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from computerVisionBach.models.Unet_SS.SS_models.Unet import UNet
from functools import partial

import torch

# ---- optional: central defaults
DEFAULTS = {
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "in_channels": 3,
}

# ---- helpers for HF models so return signature is consistent
def _build_mask2former(num_classes, device, class_names=None, **_):
    name = "facebook/mask2former-swin-small-ade-semantic"
    processor = Mask2FormerImageProcessor.from_pretrained(name, reduce_labels=False, do_rescale=False)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    id2label = {i: c for i, c in enumerate(class_names)}
    label2id = {c: i for i, c in id2label.items()}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = 255
    return model.to(device), processor

def _build_segformer(num_classes, device, **_):
    name = "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return model.to(device), processor

def _build_upernet(num_classes, device, **_):
    name = "openmmlab/upernet-swin-small"
    # upernet uses different processors depending on backbone; SegformerImageProcessor works for many cases
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    model = UperNetForSemanticSegmentation.from_pretrained(
        name, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return model.to(device), processor

# ---- SMP builders (return processor=None)
def _build_unet(num_classes, device, encoder_name, encoder_weights, in_channels, **_):
    m = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return m.to(device), None

def _build_unetpp(num_classes, device, encoder_name, encoder_weights, in_channels, **_):
    m = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
    return m.to(device), None

def _build_fpn(num_classes, device, encoder_name, encoder_weights, in_channels, **_):
    m = smp.FPN(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
    return m.to(device), None

def _build_deeplabv3plus(num_classes, device, encoder_name, encoder_weights, in_channels, encoder_output_stride=16, **_):
    m = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        encoder_output_stride=encoder_output_stride,
    )
    return m.to(device), None

def _build_custom_unet(num_classes, device, in_channels, **_):
    # your own UNet class
    m = UNet(in_channels, num_classes)
    return m.to(device), None

# ---- registry with aliases
_MODEL_REGISTRY = {
    # SMP
    "unet": _build_unet,
    "unet_resnet": partial(_build_unet, encoder_name="resnet101"),
    "unet_resnet101": partial(_build_unet, encoder_name="resnet101"),
    "unetpp": _build_unetpp,
    "unet++": _build_unetpp,
    "fpn": _build_fpn,
    "deeplabv3+": _build_deeplabv3plus,
    "deeplabv3plus": _build_deeplabv3plus,
    "custom_unet": _build_custom_unet,

    # HF
    "mask2former": _build_mask2former,
    "segformer": _build_segformer,
    "upernet": _build_upernet,
}

def build_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    *,
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
    in_channels: int | None = None,
    **extra,
):
    """
    Returns: (model, processor_or_None)
    - encoder_* and in_channels are used by SMP builders; ignored by HF builders.
    - extra is forwarded (e.g., encoder_output_stride for DeepLabV3+).
    """
    key = model_name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {model_name}. Available: {sorted(_MODEL_REGISTRY.keys())}")

    # fill defaults for SMP builders
    kwargs = dict(
        encoder_name=encoder_name or DEFAULTS["encoder_name"],
        encoder_weights=encoder_weights or DEFAULTS["encoder_weights"],
        in_channels=in_channels or DEFAULTS["in_channels"],
        hf_name=extra.get("hf_name"),
        ignore_index=extra.get("ignore_index", 255),
        reduce_labels=extra.get("reduce_labels", False),
        do_rescale=extra.get("do_rescale", False),
        **extra,
    )

    builder = _MODEL_REGISTRY[key]
    return builder(num_classes=num_classes, device=device, **kwargs)
