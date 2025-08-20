"""
Streamlit App: Aerial/Satellite Semantic Segmentation (FLAIR & DLR)
- Handles JPG/PNG and GeoTIFF (TIF/TIFF)
- Supports SMP (Unet, DeepLabV3+) and HF Transformers (SegFormer, UPerNet)
- Patch-wise full-image inference or random-crop preview
- Dataset-aware preprocessing (FLAIR vs DLR)

Notes
-----
* FLAIR images are typically GeoTIFFs and some FLAIR models may expect >3 channels (e.g., RGB+IR+Elevation).
* DLR images are typically JPG/PNG (3-channel RGB).
* Adjust checkpoint paths under `MODEL_CHECKPOINTS` as needed.
* `utils.COLOR_MAP_dense` and `utils.class_to_rgb` are assumed to be available
  in your repository and consistent with your trained models.
"""

# put this at the VERY TOP of vis_app.py, before importing from computerVisionBach
import sys
from pathlib import Path

# project root is the directory that contains the 'computerVisionBach' package
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Machine-Deep-Learning-Center
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import os
import io
import gc
import atexit
import tempfile
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import rasterio
import cv2

import streamlit as st
from torchvision import transforms
from patchify import patchify, unpatchify
from transformers import SegformerImageProcessor
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    UperNetForSemanticSegmentation,
)

# Project imports (assumed to exist in your repo)
from computerVisionBach.models.Unet_SS import utils
from computerVisionBach.models.Unet_SS.satellite_dataset.flair_dataset import FlairDataset
from computerVisionBach.models.model_pipeline import smp

# Constants:
N_CLASSES = 19
# =============================
# App Config & Globals
# =============================
st.set_page_config(page_title="🛰️ Semantic Segmentation", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset options
DATASETS = ["DLR", "FLAIR"]

# Model options
MODELS = [
    "deeplabv3+",   # SMP + resnet50 (DLR by default)
    "unet",         # SMP + resnet50 (DLR by default)
    "segformer",    # HF Segformer B2
    "upernet",      # HF UPerNet ConvNeXt small
    "unet_resnet",
    "mask2former",
]

# Model checkpoints (adjust paths for your machine)
MODEL_CHECKPOINTS: Dict[str, str] = {
    "unet": "computerVisionBach/models/Unet_SS/checkpoints/unet_resnet50_model.pth",
    "deeplabv3+": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/deeplabv3+_model_dlr_101.pth", #"/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/deeplabv3+_model_dlr_resnet50_65epochs.pth"
    "upernet": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/upernet_model.pth",
    "unet_resnet": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/unet_resnet34_model_flair_unet_resnet34.pth", #"/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/unet_resnet101_model_flair_unet.pth",
    "segformer": "",  # leave empty to use HF weights
    "Mask2former": "/home/ryqc/data/experiments/segmentation/checkpoints/dlr/Mask2former/enc-swin_t_w-ade20k/Mask2former_dlr_2025-08-19_14-05-02.pth",
}

# Normalization for SMP models trained on ImageNet encoders
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Color map (DLR dense example). Must match your model's class order.
COLOR_MAP_RGB_DLR = {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}

COLOR_MAP_RGB_FLAIR = {k: v for k, v in FlairDataset.COLOR_MAP.items()}


# =========================
# Legend helper
# ============================
# --- Legend helpers ---

def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

# DLR: build (name, hex) from your utils.COLOR_MAP_dense {idx: [name, hex]}
DLR_LEGEND = {i: (v[0], v[1]) for i, v in utils.COLOR_MAP_dense.items()}

# FLAIR: names in 0..18 order (match your COLOR_MAP keys)
FLAIR_NAMES_19 = [
    "building", "pervious surface", "impervious surface", "bare soil", "water",
    "coniferous", "deciduous", "brushwood", "vineyard", "herbaceous vegetation",
    "agricultural land", "plowed land", "swimming pool", "snow", "clear cut",
    "mixed", "ligneous", "greenhouse", "other"
]

# Make hex colors for FLAIR maps (you had RGB tuples)
FLAIR_HEX_19 = {i: rgb_to_hex(rgb) for i, rgb in FlairDataset.COLOR_MAP.items()}
# For your 13-class relabel (original 1..13 -> new 0..12), we drop original idx 0 ("building")
FLAIR_HEX_13 = {i: FLAIR_HEX_19[i + 1] for i in range(13)}
FLAIR_NAMES_13 = {i: FLAIR_NAMES_19[i + 1] for i in range(13)}

def get_legend_items(dataset: str, num_classes: int):
    """
    Returns list of (name, hex) tuples for the current dataset/class-count.
    """
    if dataset == "DLR":
        return [(name, hex_color) for name, hex_color in DLR_LEGEND.values()]

    if dataset == "FLAIR" and num_classes == 19:
        return [(FLAIR_NAMES_19[i], FLAIR_HEX_19[i]) for i in range(19)]

    if dataset == "FLAIR" and num_classes == 13:
        return [(FLAIR_NAMES_13[i], FLAIR_HEX_13[i]) for i in range(13)]

    # fallback
    return [(f"class {i}", f"#{(i*37)%255:02x}{(i*91)%255:02x}{(i*53)%255:02x}")
            for i in range(num_classes)]


def render_legend(dataset: str, num_classes: int, cols=4):
    """
    Nicely render legend as colored chips in a grid (no numbers shown).
    """
    items = get_legend_items(dataset, num_classes)
    grid = st.columns(cols)
    for idx, (name, hex_color) in enumerate(items):
        with grid[idx % cols]:
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;margin:6px 0;">
                  <div style="width:18px;height:18px;border-radius:4px;background:{hex_color};
                              border:1px solid rgba(255,255,255,0.15);margin-right:8px;"></div>
                  <div style="font-size:0.92rem;">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )




# =============================
# Utility Functions
# =============================

def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def safe_to_uint8(img: np.ndarray) -> np.ndarray:
    """Scale/clip float array to uint8 for display if needed."""
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    maxv = arr.max() if arr.size else 1.0
    if maxv <= 0:
        maxv = 1.0
    arr = np.clip(arr / maxv * 255.0, 0, 255).astype(np.uint8)
    return arr


def read_uploaded_image(uploaded_file, dataset: str, model_choice: str) -> Tuple[np.ndarray, Image.Image]:
    """Read an uploaded image file into a HWC numpy array and a displayable PIL Image.

    - For GeoTIFF (TIF/TIFF), use rasterio and preserve bands.
    - For JPG/PNG, use PIL.
    - Enforce channel count expectations based on dataset/model.
    """
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()

    if ext in ["jpg", "jpeg", "png"]:
        img = Image.open(uploaded_file).convert("RGB")
        np_img = np.array(img)

        if dataset == "FLAIR" and model_choice == "unet_resnet34_flair":
            # Expect 5 channels (RGB + IR + Elevation). If not provided,
            # create dummy IR/Elevation so the model can run.
            h, w, _ = np_img.shape
            dummy_ir = np.zeros((h, w, 1), dtype=np_img.dtype)
            dummy_elev = np.zeros((h, w, 1), dtype=np_img.dtype)
            np_img = np.concatenate([np_img, dummy_ir, dummy_elev], axis=2)
            st.warning("Uploaded RGB; added dummy IR + Elevation to meet 5-channel input.")

        disp = img

    elif ext in ["tif", "tiff"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        atexit.register(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))

        with rasterio.open(tmp_path) as src:
            arr = src.read()  # (bands, H, W)
        np_img = np.transpose(arr, (1, 2, 0))  # HWC

        # Expected channels is driven by the selected model’s in_channels
        expected_c = st.session_state.get("in_channels", 3)

        if np_img.shape[2] >= expected_c:
            np_img = np_img[:, :, :expected_c]
        else:
            if st.session_state.get("pad_missing", True):
                # pad zeros to reach expected_c
                h, w, c = np_img.shape
                pad = np.zeros((h, w, expected_c - c), dtype=np_img.dtype)
                np_img = np.concatenate([np_img, pad], axis=2)
                st.warning(f"TIFF has {c} bands; padded to {expected_c}.")
            else:
                st.error(f"TIFF has {np_img.shape[2]} bands, but model expects {expected_c}.")
                st.stop()

        disp = Image.fromarray(safe_to_uint8(np_img[:, :, :3]))

    else:
        st.error("Unsupported file type. Please upload JPG/PNG/TIF/TIFF.")
        st.stop()

    return np_img, disp

# =============================
# HF processor helpers (no unhashable args)
# =============================

def _hf_name_or_path(model) -> str:
    """
    Derive a stable hub name/path string for the loaded HF model.
    Falls back to sensible defaults by model_type.
    """
    # Prefer the exact thing you loaded
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    if name_or_path:
        return name_or_path

    # Fallbacks if the config lost path info
    mtype = getattr(getattr(model, "config", None), "model_type", None)
    if mtype == "mask2former":
        return "facebook/mask2former-swin-small-ade-semantic"
    if mtype == "segformer":
        return "nvidia/segformer-b2-finetuned-ade-512-512"
    if mtype == "upernet":
        return "openmmlab/upernet-convnext-small"
    # Generic fallback (works for SegFormer-style processors)
    return "nvidia/segformer-b2-finetuned-ade-512-512"


@st.cache_resource(show_spinner=False)
def get_processor_by_name(name_or_path: str):
    """
    Cached by a plain string → hashable → no UnhashableParamError.
    """
    return AutoImageProcessor.from_pretrained(name_or_path)


# =============================
# Model Loading
# =============================
@st.cache_resource(show_spinner=True)
def load_model(model_name: str, dataset: str, num_classes: int, encoder_name: str = None, in_channels: int = 3):
    expects_5ch = (in_channels == 5)

    if model_choice == "unet_resnet":
        # Build a generic Unet+ResNet with chosen params
        enc = encoder_name or "resnet34"
        model = smp.Unet(
            encoder_name=enc,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        ckpt = MODEL_CHECKPOINTS.get("unet_resnet", "") # single, generic checkpoint path
        print(ckpt)

        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")

            # 1) Drop head if num_classes mismatch
            head_w = sd.get("segmentation_head.0.weight")
            if head_w is not None and head_w.shape[0] != num_classes:
                sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head.")}

            # 2) Adapt first conv if channel count mismatch
            conv_key = "encoder.conv1.weight"
            if conv_key in sd:
                w = sd[conv_key]  # [out_c, in_c_ckpt, k, k]
                in_c_ckpt = w.shape[1]
                if in_c_ckpt != in_channels:
                    if in_channels < in_c_ckpt:
                        # Reduce channels: take first in_channels (or average if you prefer)
                        sd[conv_key] = w[:, :in_channels, :, :]
                    else:
                        # Increase channels: repeat/average to fill
                        reps = in_channels - in_c_ckpt
                        extra = w[:, :1, :, :].repeat(1, reps, 1, 1)  # repeat first channel
                        sd[conv_key] = torch.cat([w, extra], dim=1)

            model.load_state_dict(sd, strict=False)

        model.eval()
        return model.to(device), num_classes, expects_5ch

    # ---- other models unchanged, but use num_classes ----
    if model_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        ckpt = MODEL_CHECKPOINTS.get("deeplabv3+", "")
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            head_w = sd.get("segmentation_head.0.weight")
            if head_w is not None and head_w.shape[0] != num_classes:
                sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head.")}
            model.load_state_dict(sd, strict=False)

    elif model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
        ckpt = MODEL_CHECKPOINTS.get("unet", "")
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            head_w = sd.get("segmentation_head.0.weight")
            if head_w is not None and head_w.shape[0] != num_classes:
                sd = {k: v for k, v in sd.items() if not k.startswith("segmentation_head.")}
            model.load_state_dict(sd, strict=False)

    elif model_name == "mask2former":
        # HF models are RGB-only; do not advertise 5ch
        expects_5ch = False
        name = "facebook/mask2former-swin-small-ade-semantic"
        class_names = [f"Class_{i}" for i in range(num_classes)]
        id2label = {i: c for i, c in enumerate(class_names)}
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            name,
            num_labels=num_classes,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,
        )
    elif model_name == "segformer":
        expects_5ch = False
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "upernet":
        expects_5ch = False
        model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small",
            num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        st.error(f"Unsupported model: {model_name}")
        st.stop()

    model.eval()
    return model.to(device), num_classes, expects_5ch




# =============================
# Inference Helpers
# =============================

# =============================
# Inference Helpers (REPLACED)
# =============================

def _to_model_input(tensor_image_chw: torch.Tensor) -> torch.Tensor:
    """Normalize to ImageNet stats for SMP backbones (3ch only)."""
    return NORMALIZE(tensor_image_chw)

def _ensure_rgb_uint8(np_patch: np.ndarray) -> np.ndarray:
    """
    Ensures (H, W, 3) uint8 in [0,255] for HF processors.
    If more than 3 channels are present, uses the first 3 with a warning.
    If float, scales to [0,255].
    """
    if np_patch.ndim != 3:
        raise ValueError(f"Expected HWC, got shape {np_patch.shape}")
    H, W, C = np_patch.shape
    if C < 3:
        raise ValueError("HF models require 3-channel RGB input.")
    if C > 3:
        # Keep first 3 channels and warn once via Streamlit
        st.warning("Input has >3 channels; using first 3 for HF model.", icon="!!!!")
        np_patch = np_patch[:, :, :3]

    if np_patch.dtype != np.uint8:
        # If values look like 0..1 floats, scale up; otherwise clip to 0..255
        mx = float(np.max(np_patch)) if np_patch.size else 1.0
        if mx <= 1.0:
            np_patch = (np_patch * 255.0).astype(np.uint8)
        else:
            np_patch = np.clip(np_patch, 0, 255).astype(np.uint8)
    return np_patch

def _m2f_postprocess_semantic(processor, outputs, H: int, W: int):
    """
    Version‑safe post‑processing for Mask2Former semantic maps.
    Returns a (H, W) int64 numpy array.
    """
    if hasattr(processor, "post_process_semantic"):
        seg = processor.post_process_semantic(outputs, target_sizes=[(H, W)])[0]["segmentation"]
    elif hasattr(processor, "post_process_semantic_segmentation"):
        # Older HF versions (e.g., 4.26–4.27)
        seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
    else:
        raise AttributeError(
            "This ImageProcessor lacks semantic post‑processing. "
            "Upgrade transformers or provide a compatible processor."
        )
    # seg can be Tensor or numpy; normalize to numpy int64
    if hasattr(seg, "detach"):
        seg = seg.detach().cpu().numpy()
    return seg.astype(np.int64)



def _forward_model(model: torch.nn.Module, np_patch: np.ndarray) -> np.ndarray:
    """
    Run model on a single patch (HWC). Returns argmax mask (H, W).
    - SMP models: BCHW float, normalized (RGB only)
    - SegFormer/UPerNet (HF): AutoImageProcessor + .logits upsample -> argmax
    - Mask2Former (HF): AutoImageProcessor + post_process_semantic (no manual upsample)
    """
    H, W = np_patch.shape[:2]

    # ================= HF MODELS =================
    if hasattr(model, "config") and hasattr(model.config, "model_type"):
        model_type = model.config.model_type

        # Guard: HF models are RGB only
        if np_patch.shape[-1] != 3:
            # Try to adapt; if still not 3ch, stop
            try:
                rgb_patch = _ensure_rgb_uint8(np_patch)
            except Exception as e:
                st.error(f"HF model expects 3-channel RGB input. {e}")
                st.stop()
        else:
            rgb_patch = _ensure_rgb_uint8(np_patch)

        processor = get_processor_by_name(_hf_name_or_path(model))

        with torch.no_grad():
            inputs = processor(images=[rgb_patch], return_tensors="pt")  # do_rescale handled by processor
            # Move tensors to device
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            outputs = model(**inputs)

            if model_type == "mask2former":
                out = _m2f_postprocess_semantic(processor, outputs, H, W)
                return out

            # SegFormer / UPerNet path: upsample logits to (H,W)
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # (B,C,h,w)
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
                return pred.astype(np.int64)

            # Fallback (shouldn’t happen)
            st.error("Unexpected HF model output: missing .logits and not Mask2Former.")
            st.stop()

    # ================= SMP MODELS =================
    with torch.no_grad():
        # SMP expects BCHW floats in [0,1]
        x = torch.tensor(np_patch.transpose(2, 0, 1), dtype=torch.float32)
        x = x / 255.0 if x.max() > 1.0 else x

        # Normalize only for 3ch (your NORMALIZE is ImageNet stats)
        if x.shape[0] == 3:
            x = _to_model_input(x)

        x = x.unsqueeze(0).to(device)
        logits = model(x)  # (B,C,h,w)

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()[0]
        return pred.astype(np.int64)



def predict_full_image_patchwise(np_img: np.ndarray, crop: int) -> np.ndarray:
    """Predict a full image by splitting into non-overlapping patches of size `crop`."""
    h, w, c = np_img.shape
    pad_h = (crop - h % crop) % crop
    pad_w = (crop - w % crop) % crop

    padded = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    patches = patchify(padded, (crop, crop, c), step=crop)

    mask_rows = []
    for i in range(patches.shape[0]):
        row = []
        for j in range(patches.shape[1]):
            pred = _forward_model(st.session_state.model, patches[i, j, 0])
            row.append(pred)
        mask_rows.append(row)

    mask_array = np.stack([np.stack(r, axis=0) for r in mask_rows], axis=0)
    full_mask = unpatchify(mask_array, padded.shape[:2])
    return full_mask[:h, :w]


def predict_random_crop(np_img: np.ndarray, crop: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = np_img.shape
    if h < crop or w < crop:
        raise ValueError("Image smaller than crop size.")
    top = np.random.randint(0, h - crop + 1)
    left = np.random.randint(0, w - crop + 1)
    crop_img = np_img[top: top + crop, left: left + crop]
    pred = _forward_model(st.session_state.model, crop_img)
    return crop_img, pred


def colorize_mask(mask: np.ndarray, dataset: str) -> np.ndarray:
    cmap = COLOR_MAP_RGB_FLAIR if dataset == "FLAIR" else COLOR_MAP_RGB_DLR
    return utils.class_to_rgb(mask, cmap)

def get_colormap(dataset: str, num_classes: int):
    if dataset == "DLR":
        # DLR map comes as hex strings -> convert to RGB tuples once
        return {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}

    # FLAIR
    if num_classes == 19:
        return {k: v for k, v in FlairDataset.COLOR_MAP.items()}
    elif num_classes == 13:
        # Your relabeling uses original 1..13 -> 0..12
        return {i: FlairDataset.COLOR_MAP[i + 1] for i in range(13)}
    else:
        # fallback: truncate or synthesize colors
        base = {k: v for k, v in FlairDataset.COLOR_MAP.items()}
        return {i: base.get(i, (int((i*37)%255), int((i*91)%255), int((i*53)%255))) for i in range(num_classes)}

# =============================
# UI Layout
# =============================

st.markdown(
    """
    <h1 style='text-align: center; color: #14C4FF; font-size: 3rem;'>🚀 Real-Time Aerial Image Segmentation</h1>
    <p style='text-align: center; font-size: 1.1rem;'>Upload satellite or aerial imagery (JPG/PNG/TIF/TIFF). Choose dataset/model and run inference.</p>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("BEV Segmentation AI")
    dataset = st.selectbox("Dataset", DATASETS, index=0, help="Pick the dataset the image belongs to (affects preprocessing & colors).")
    model_choice = st.selectbox("Model", MODELS, index=0)
    crop_size = st.slider("Crop Size", min_value=128, max_value=1024, step=64, value=512)

    # Extra controls for generic Unet+ResNet
    if model_choice == "unet_resnet":
        encoder_name = st.selectbox("Encoder", ["resnet34", "resnet50", "resnet101"], index=0)
        in_channels_ui = st.number_input("Input channels", min_value=1, max_value=8, value=3, step=1)
    else:
        encoder_name = None
        in_channels_ui = 3

    # Choose number of classes explicitly
    if dataset == "DLR":
        num_classes_ui = st.number_input("Num classes", min_value=1, max_value=255, value=20, step=1)
    else:  # FLAIR
        num_classes_ui = st.selectbox("Num classes", options=[13, 19], index=0)

    # (Re)load model if any key setting changed
    key_tuple = (model_choice, dataset, int(num_classes_ui), encoder_name, int(in_channels_ui))
    if "_loaded_key" not in st.session_state or st.session_state._loaded_key != key_tuple:
        free_cuda()
        model, num_classes, expects_5ch = load_model(
            model_choice, dataset, int(num_classes_ui),
            encoder_name=encoder_name, in_channels=int(in_channels_ui)
        )
        st.session_state.model = model
        st.session_state.num_classes = num_classes
        st.session_state.in_channels = int(in_channels_ui)
        st.session_state.expects_5ch = expects_5ch
        st.session_state._loaded_key = key_tuple

    st.subheader("Legend")
    render_legend(dataset, int(st.session_state.get("num_classes", num_classes_ui)))

# File uploader
uploaded = st.file_uploader("📤 Upload an aerial/urban image (JPG/PNG/TIF/TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded:
    st.stop()

# Read file based on dataset/model expectations
np_img, disp_img = read_uploaded_image(uploaded, dataset, model_choice)

# Display original
st.subheader("🖼️ Original Image")
st.image(disp_img, use_column_width=True)

# Inference method
mode = st.radio("Select Inference Method", ["Full Image (patch-wise)", "Random Crop"], index=1)

if st.button("Run Segmentation"):
    with st.spinner("Segmenting..."):
        if mode == "Random Crop":
            crop_img, pred_mask = predict_random_crop(np_img, crop_size)
            base_rgb = crop_img[:, :, :3] if crop_img.shape[2] > 3 else crop_img
        else:
            pred_mask = predict_full_image_patchwise(np_img, crop_size)
            base_rgb = np_img[:, :, :3] if np_img.shape[2] > 3 else np_img

        pred_rgb = utils.class_to_rgb(pred_mask, get_colormap(dataset, int(st.session_state.num_classes)))
        if pred_rgb.shape[:2] != base_rgb.shape[:2]:
            pred_rgb = cv2.resize(pred_rgb, (base_rgb.shape[1], base_rgb.shape[0]))

        overlay = cv2.addWeighted(safe_to_uint8(base_rgb), 0.6, safe_to_uint8(pred_rgb), 0.4, 0)

    st.subheader("Segmentation Results")
    c1, c2 = st.columns(2)
    c1.image(pred_rgb, caption="Predicted Mask", use_column_width=True)
    c2.image(overlay, caption="Overlay", use_column_width=True)

    # Download overlay
    buf = io.BytesIO()
    Image.fromarray(safe_to_uint8(overlay)).save(buf, format="PNG")
    st.download_button("💾 Download Overlay", buf.getvalue(), file_name="segmentation_overlay.png", mime="image/png")


# =============================
# Optional Styling
# =============================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0e1117;
        color: #E0E0E0;
        font-size: 16px;
    }
    h1, h2, h3 { color: #E3E8F0; font-weight: 600; margin-bottom: 0.5rem; }
    .stButton>button { background-color: #14C4FF; color: white; border-radius: 12px; font-size: 1.1rem; padding: 0.6rem 1.6rem; transition: all 0.3s ease-in-out; }
    .stButton>button:hover { background-color: #0072FF; transform: scale(1.05); }
    section[data-testid="stSidebar"] { background-color: #1e2633; border-right: 1px solid #39414f; padding: 1rem; }
    .stFileUploader > div > div { background-color: #1a1f27; border: 1px dashed #3A3F4B; color: #E0E0E0; padding: 0.6rem; border-radius: 8px; }
    .stRadio > div > label { background-color: #1f2630; border: 1px solid #2a2f3b; border-radius: 6px; padding: 0.5rem 1rem; margin-bottom: 6px; color: #dcdcdc; }
    .stRadio > div > label:hover { background-color: #2a2f3b; }
    .stSelectbox div, .stSlider, .stSelectbox label { color: #E0E0E0 !important; }
    .element-container p { color: #aaaaaa; font-size: 0.9rem; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1f2630; }
    ::-webkit-scrollbar-thumb { background-color: #3A3F4B; border-radius: 8px; }
    input, textarea, select { background-color: #1f2630 !important; color: #E0E0E0 !important; border: 1px solid #3A3F4B !important; border-radius: 6px; padding: 0.4rem 0.6rem; }
    .main > div > div > div > h1 { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)
