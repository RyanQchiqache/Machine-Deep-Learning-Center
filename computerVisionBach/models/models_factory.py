from __future__ import annotations
import torch
import inspect
import segmentation_models_pytorch as smp


from functools import partial
from omegaconf import OmegaConf
from transformers import SegformerForSemanticSegmentation
from transformers import UperNetForSemanticSegmentation, SegformerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
)
from computerVisionBach.models.Unet_SS.SS_models.Unet import UNet
from transformers import AutoImageProcessor
from torch import nn
from loguru import logger

cfg = OmegaConf.load("/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/config/config.yaml")
OmegaConf.resolve(cfg)

# ---- optional: central defaults
DEFAULTS = {
    "encoder_name": "resnet34",
    "encoder_weights": "imagenet",
    "in_channels": 3,
}


def freeze_backbone_stages(model) -> None:
    """
    Freeze Swin patch embedding (+ its norm) and the first two Swin stages (layers 0 and 1)
    for Hugging Face Mask2Former Swin backbones.
    """
    prefixes = (
        "model.pixel_level_module.encoder.embeddings",
        "model.pixel_level_module.encoder.encoder.layers.0",
        "model.pixel_level_module.encoder.encoder.layers.1",
    )

    logger.info("Freezing Swin patch embedding + first two stages (0, 1)...")
    num_frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            if param.requires_grad:
                param.requires_grad = False
                num_frozen += 1
                logger.debug(f"Froze: {name}")
    logger.info(f"Freezing completed. Params frozen: {num_frozen}")



def reinit_decoder(model):
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


# ---- helpers for HF models so return signature is consistent

def _ensure_label_maps(num_classes, class_names=None):
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    assert len(class_names) == num_classes, "class_names must have length == num_classes"
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in id2label.items()}
    return id2label, label2id

def _build_mask2former(num_classes, device, class_names=None,
                       hf_name=None, ignore_index=255,
                       reduce_labels=False, do_rescale=False, do_resize=False,
                       reinit_decoder: bool = False, freeze_backbone: bool = True,
                       **_):
    name = hf_name or "facebook/mask2former-swin-small-ade-semantic"
    processor = AutoImageProcessor.from_pretrained(
        name, reduce_labels=reduce_labels, do_rescale=do_rescale, do_resize=do_resize
    )
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    id2label = {i: c for i, c in enumerate(class_names)}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        name,
        num_labels=num_classes,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = ignore_index

    if freeze_backbone:
        freeze_backbone_stages(model)
        logger.success("Mask2Former Swin patch_embed + stages (0,1) frozen.")
    if reinit_decoder:
        reinit_decoder(model)
        logger.success("Mask2Former pixel decoder re-initialized.")

    return model.to(device), processor

def _build_segformer(num_classes, device, class_names=None,
                     hf_name=None, ignore_index=255,
                     reduce_labels=False, do_rescale=False, **_):
    name = hf_name or "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(
        name,
        reduce_labels=reduce_labels,   # MUST be False for your 0..19 labels
        do_rescale=do_rescale,         # OK to keep True for uint8 inputs
        size = {"height": 512, "width": 512}
    )
    id2label = {i: (class_names[i] if class_names else str(i)) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = ignore_index  # 255
    return model.to(device), processor

def _build_upernet(num_classes, device, class_names=None,
                     hf_name=None, ignore_index=255,
                     reduce_labels=False, do_rescale=False,do_resize=False, **_):
    name = hf_name or "openmmlab/upernet-swin-small"
    processor = AutoImageProcessor.from_pretrained(
        name, reduce_labels=reduce_labels, do_rescale=do_rescale, do_resize=do_resize
    )
    id2label = {i: (class_names[i] if class_names else str(i)) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}
    model = UperNetForSemanticSegmentation.from_pretrained(
        name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_index = ignore_index
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

def _build_deeplabv3plus(num_classes, device, encoder_name, encoder_weights, in_channels,
                         encoder_output_stride=None, **_):
    args = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
        encoder_output_stride=encoder_output_stride if encoder_output_stride is not None else 16,
    )


    m = smp.DeepLabV3Plus(**args)
    return m.to(device), None


def _build_custom_unet(num_classes, device, in_channels, **_):
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

    # Base knobs (safe defaults)
    base = {
        "encoder_name": encoder_name or DEFAULTS["encoder_name"],
        "encoder_weights": encoder_weights or DEFAULTS["encoder_weights"],
        "in_channels": in_channels or DEFAULTS["in_channels"],
        "ignore_index": extra.get("ignore_index", 255),
        "reduce_labels": extra.get("reduce_labels", False),
        "do_rescale": extra.get("do_rescale", False),
        # Pass-through: any other keys in `extra` will be filtered per builder
    }

    # Merge user extras last
    base.update(extra)

    # Keep only args the builder can accept
    builder = _MODEL_REGISTRY[key]

    def _filtered_kwargs(fn, kwargs):
        sig = inspect.signature(fn)
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    kwargs = _filtered_kwargs(builder, base)

    # --- Build exactly once
    model, processor = builder(num_classes=num_classes, device=device, **kwargs)

    # --- Verification (generic)
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Params — frozen: {frozen}, trainable: {trainable}")

    # Optional spot-checks (only for Mask2Former)
    if key == "mask2former":
        to_check = (
            "model.pixel_level_module.encoder.embeddings.patch_embeddings",
            "model.pixel_level_module.encoder.embeddings.norm",
            "model.pixel_level_module.encoder.encoder.layers.0",
            "model.pixel_level_module.encoder.encoder.layers.1",
        )
        shown = 0
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in to_check):
                logger.debug(f"{name} requires_grad={p.requires_grad}")
                shown += 1
                if shown >= 20:
                    break

    # --- Return the same instance you modified
    return model, processor