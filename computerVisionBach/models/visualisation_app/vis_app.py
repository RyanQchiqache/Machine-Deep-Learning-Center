# ==============================
# vis_app.py  — Streamlit UI
# Uses: computerVisionBach/vis_core/uni_infer.py
# ==============================

# --- put this at the VERY TOP ---
import sys
from pathlib import Path

# vis_app.py -> visualisation_app (0) -> models (1) -> computerVisionBach (2) -> REPO ROOT (3)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]          # directory that CONTAINS 'computerVisionBach'
PKG_DIR   = REPO_ROOT / "computerVisionBach"

# Prepend both, to be extra robust
for p in (str(REPO_ROOT), str(PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- std / third-party ----
import os
import io
import gc
import atexit
import tempfile
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from PIL import Image
import rasterio
import cv2
import streamlit as st
from patchify import patchify, unpatchify

# ---- project imports ----
from computerVisionBach.models.Unet_SS import utils
from computerVisionBach.models.Unet_SS.satellite_dataset.flair_dataset import FlairDataset

# unified inference (your new file)
from computerVisionBach.models.visualisation_app.uni_infer import (
    load_model, ModelBundle, predict_patch
)

# =============================
# App Config & Globals
# =============================
st.set_page_config(page_title="🛰️ Semantic Segmentation", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset options
DATASETS = ["DLR", "FLAIR"]

# Model options
MODELS = [
    "deeplabv3+",    # SMP + ResNet
    "unet",          # SMP + ResNet
    "unet_resnet",   # generic Unet+ResNet (customizable)
    "upernet",       # HF
    "mask2former",   # HF
]

MODEL_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "DLR": {
        "unet": "computerVisionBach/models/Unet_SS/checkpoints/unet_resnet50_model.pth",
        "deeplabv3+": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/deeplabv3+_model_dlr_resnet50_65epochs.pth",
        "upernet": "",
        "unet_resnet": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/unet_resnet50_model_dlr_resnet50.pth",
        "mask2former": "/home/ryqc/data/experiments/segmentation/checkpoints/dlr/mask2former/enc-Swin_AKD20k_w-70_12/mask2former_dlr_2025-08-29_15-07-04_hf",
    },
    "FLAIR": {
        "unet": "",
        "deeplabv3+": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/deeplabv3+_model_flair_deeplabv3+.pth",
        "upernet": "",
        "unet_resnet": "/home/ryqc/data/Machine-Deep-Learning-Center/computerVisionBach/models/Unet_SS/checkpoints/unet_resnet34_model_flair_unet_resnet34.pth",
        "mask2former": "/home/ryqc/data/experiments/segmentation/checkpoints/flair/mask2former/enc-Swin_AKD20k_w-70_12_flair/mask2former_flair_2025-09-02_14-10-47_hf",
    },
}


# =============================
# Small utilities
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

def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

# Read uploaded image (supports JPG/PNG/TIF/TIFF), preserves bands for TIFF
def read_uploaded_image(uploaded_file, dataset: str, model_choice: str) -> Tuple[np.ndarray, Image.Image]:
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()

    if ext in ["jpg", "jpeg", "png"]:
        img = Image.open(uploaded_file).convert("RGB")
        np_img = np.array(img)
        disp = img

    elif ext in ["tif", "tiff"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        atexit.register(lambda: os.path.exists(tmp_path) and os.remove(tmp_path))

        with rasterio.open(tmp_path) as src:
            arr = src.read()  # (bands, H, W)
        np_img = np.transpose(arr, (1, 2, 0))  # HWC

        # The expected in_channels is determined by the currently loaded model (if any)
        expected_c = st.session_state.get("in_channels", 3)
        if np_img.shape[2] >= expected_c:
            np_img = np_img[:, :, :expected_c]
        else:
            # pad missing channels with zeros
            h, w, c = np_img.shape
            pad = np.zeros((h, w, expected_c - c), dtype=np_img.dtype)
            np_img = np.concatenate([np_img, pad], axis=2)
            st.warning(f"TIFF has {c} bands; padded to {expected_c}.")

        disp = Image.fromarray(safe_to_uint8(np_img[:, :, :3]))
    else:
        st.error("Unsupported file type. Please upload JPG/PNG/TIF/TIFF.")
        st.stop()

    return np_img, disp

# =========================
# Legend helpers
# =========================
DLR_LEGEND = {i: (v[0], v[1]) for i, v in utils.COLOR_MAP_dense.items()}

FLAIR_NAMES_19 = [
    "building", "pervious surface", "impervious surface", "bare soil", "water",
    "coniferous", "deciduous", "brushwood", "vineyard", "herbaceous vegetation",
    "agricultural land", "plowed land", "swimming pool", "snow", "clear cut",
    "mixed", "ligneous", "greenhouse", "other"
]
FLAIR_HEX_19 = {i: rgb_to_hex(rgb) for i, rgb in FlairDataset.COLOR_MAP.items()}

FLAIR_NAMES_13 = {i: FLAIR_NAMES_19[i] for i in range(13)}
FLAIR_NAMES_12 = {i: FLAIR_NAMES_19[i] for i in range(12)}

FLAIR_HEX_13 = {i: FLAIR_HEX_19[i] for i in range(13)}
FLAIR_HEX_12 = {i: FLAIR_HEX_19[i] for i in range(12)}


def get_legend_items(dataset: str, num_classes: int):
    if dataset == "DLR":
        return [(name, hex_color) for name, hex_color in DLR_LEGEND.values()]
    if dataset == "FLAIR" and num_classes in (12, 13, 19):
        return [(FLAIR_NAMES_19[i], FLAIR_HEX_19[i]) for i in range(num_classes)]
    return [(f"class {i}", f"#{(i*37)%255:02x}{(i*91)%255:02x}{(i*53)%255:02x}") for i in range(num_classes)]


def render_legend(dataset: str, num_classes: int, cols=4):
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

def get_colormap(dataset: str, num_classes: int):
    if dataset == "DLR":
        return {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}
    if dataset == "FLAIR" and num_classes in (12, 13, 19):
        return {i: FlairDataset.COLOR_MAP[i] for i in range(num_classes)}
    # fallback
    base = {k: v for k, v in FlairDataset.COLOR_MAP.items()}
    return {i: base.get(i, (int((i*37)%255), int((i*91)%255), int((i*53)%255))) for i in range(num_classes)}



# =============================
# Patch-wise inference (uses predict_patch from uni_infer)
# =============================
def predict_full_image_patchwise(np_img: np.ndarray, crop: int) -> np.ndarray:
    h, w, c = np_img.shape
    pad_h = (crop - h % crop) % crop
    pad_w = (crop - w % crop) % crop

    padded = np.pad(np_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    patches = patchify(padded, (crop, crop, c), step=crop)

    rows = []
    for i in range(patches.shape[0]):
        row = []
        for j in range(patches.shape[1]):
            pred = predict_patch(patches[i, j, 0], st.session_state.bundle, device=device)
            row.append(pred)
        rows.append(np.stack(row, axis=0))

    mask_array = np.stack(rows, axis=0)
    full_mask = unpatchify(mask_array, padded.shape[:2])
    return full_mask[:h, :w]

def predict_random_crop(np_img: np.ndarray, crop: int):
    h, w, _ = np_img.shape
    if h < crop or w < crop:
        raise ValueError("Image smaller than crop size.")
    top = np.random.randint(0, h - crop + 1)
    left = np.random.randint(0, w - crop + 1)
    crop_img = np_img[top: top + crop, left: left + crop]
    pred = predict_patch(crop_img, st.session_state.bundle, device=device)
    return crop_img, pred

def resolve_checkpoint(dataset: str, model_choice: str) -> Optional[str]:
    ckpt = MODEL_CHECKPOINTS.get(dataset, {}).get(model_choice, "") or None
    if not ckpt:
        return None

    is_hf = model_choice in {"segformer", "upernet", "mask2former"}
    looks_like_path = ckpt.startswith("/") or ckpt.startswith("./") or ckpt.startswith("../")

    if is_hf:
        # HF: accept dir OR hub id. If it looks like a path, require a dir.
        if looks_like_path and not os.path.isdir(ckpt):
            st.warning(f"Checkpoint for {model_choice} should be a directory or a hub id; using default weights.")
            return None
        return ckpt
    else:
        # SMP: must be a file path
        if not os.path.isfile(ckpt):
            st.warning(f"Checkpoint file not found for {model_choice}; using default weights.")
            return None
        return ckpt

# =============================
# UI
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
    dataset = st.selectbox("Dataset", DATASETS, index=0,
                           help="Pick the dataset the image belongs to (affects preprocessing & colors).")
    model_choice = st.selectbox("Model", MODELS, index=0)
    crop_size = st.slider("Crop Size", min_value=128, max_value=1024, step=64, value=512)

    # Extra controls for generic Unet+ResNet
    if model_choice == "unet_resnet":
        encoder_name = st.selectbox("Encoder", ["resnet34", "resnet50", "resnet101"], index=0)
        in_channels_ui = st.number_input("Input channels", min_value=1, max_value=8, value=3, step=1)
    else:
        encoder_name = None
        in_channels_ui = 3  # HF models are RGB-only; SMP defaults to 3 unless you trained >3

    # Number of classes
    if dataset == "DLR":
        num_classes_ui = st.number_input("Num classes", min_value=1, max_value=255, value=20, step=1)
    else:  # FLAIR
        num_classes_ui = st.selectbox("Num classes", options=[12, 13, 19], index=0)

    # (Re)load model if any key setting changed
    key_tuple = (model_choice, dataset, int(num_classes_ui), encoder_name, int(in_channels_ui))
    if "_loaded_key" not in st.session_state or st.session_state._loaded_key != key_tuple:
        free_cuda()
        ckpt = resolve_checkpoint(dataset, model_choice)

        class_names = (
            FLAIR_NAMES_19[:int(num_classes_ui)]
            if (dataset == "FLAIR" and int(num_classes_ui) in (12, 13, 19))
            else [name for name, _hex in get_legend_items(dataset, int(num_classes_ui))]
        )

        bundle = load_model(
            model_name=model_choice,
            num_classes=int(num_classes_ui),
            in_channels=int(in_channels_ui),
            encoder_name=encoder_name,
            ckpt_path=ckpt,
            device=device,
            class_names=class_names
        )
        st.session_state.bundle = bundle
        st.session_state.model = bundle.model                # keep for backwards compatibility
        st.session_state.num_classes = bundle.num_classes
        st.session_state.in_channels = bundle.in_channels
        st.session_state.expects_5ch = bundle.expects_5ch
        st.session_state._loaded_key = key_tuple

    st.subheader("Legend")
    render_legend(dataset, int(st.session_state.get("num_classes", num_classes_ui)))

# File uploader
uploaded = st.file_uploader(" Upload an aerial/urban image (JPG/PNG/TIF/TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

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
    st.download_button("💾 Download Overlay", buf.getvalue(),
                       file_name="segmentation_overlay.png", mime="image/png")

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
