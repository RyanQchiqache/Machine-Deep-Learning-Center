import sys
import os
import tempfile
import atexit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from transformers.modeling_utils import PreTrainedModel
from transformers import SegformerImageProcessor
import streamlit as st
import numpy as np
from PIL import Image
import torch
from io import BytesIO
import rasterio
import cv2

from computerVisionBach.models.Unet_SS import utils
from computerVisionBach.models.Unet_SS.model_pipeline import UNet, smp, PATCH_SIZE, N_CLASSES

# ----------------------------------------
# ⚙️ Config
# ----------------------------------------
st.set_page_config(page_title="🛰️ Semantic Segmentation", layout="wide")
color_map_rgb = {k: utils.hex_to_rgb(v[1]) for k, v in utils.COLOR_MAP_dense.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------
# Load trained model
# ----------------------------------------
@st.cache_resource
def load_model(model_name="deeplabv3+"):
    if model_name == "unet":
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=N_CLASSES,
        )
        ckpt = "computerVisionBach/models/Unet_SS/checkpoints/unet_resnet50_model.pth"
    elif model_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=N_CLASSES
        )
        ckpt = "computerVisionBach/models/Unet_SS/checkpoints/deeplabv3+_model.pth"
    elif model_name == "segformer":
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=N_CLASSES,
            ignore_mismatched_sizes=True,
        )
        ckpt = "checkpoints/segformer_model.pth"
    elif model_name == "upernet":
        from transformers import UperNetForSemanticSegmentation
        model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small",
            num_labels=N_CLASSES,
            ignore_mismatched_sizes=True,
        )
        ckpt = "computerVisionBach/models/Unet_SS/checkpoints/upernet_model.pth"
    else:
        raise ValueError("Unsupported model name.")

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model.to(device)



# ----------------------------------------
# Predict full image using patch-wise logic
# ----------------------------------------


processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

def predict_image_with_patches(image_np, model):
    from patchify import patchify, unpatchify

    h, w, _ = image_np.shape
    h_crop = (h // PATCH_SIZE) * PATCH_SIZE
    w_crop = (w // PATCH_SIZE) * PATCH_SIZE
    image_cropped = image_np[:h_crop, :w_crop]

    patches = patchify(image_cropped, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    mask_patches = []

    with torch.no_grad():
        for i in range(patches.shape[0]):
            row = []
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]

                if isinstance(model, PreTrainedModel):
                    input_processed = processor(images=patch, return_tensors="pt", do_rescale=False).to(device)
                    output = model(**input_processed).logits
                else:
                    tensor = torch.tensor(patch.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
                    output = model(tensor)

                if output.shape[-2:] != patch.shape[:2]:
                    output = torch.nn.functional.interpolate(output, size=patch.shape[:2], mode="bilinear", align_corners=False)

                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                row.append(pred)
            mask_patches.append(row)

    mask_array = np.stack([np.stack(row, axis=0) for row in mask_patches], axis=0)
    full_mask = unpatchify(mask_array, (h_crop, w_crop))

    if (h_crop != h) or (w_crop != w):
        padded = np.zeros((h, w), dtype=np.uint8)
        padded[:h_crop, :w_crop] = full_mask
        return padded

    return full_mask



# ----------------------------------------
# Main Streamlit Interface
# ----------------------------------------
st.sidebar.title("BEV Segmentation AI")
model_choice = st.sidebar.selectbox("Choose model", ["deeplabv3+", "unet", "segformer", "upernet"])
model = load_model(model_choice)

st.markdown("""
    <h1 style='text-align: center; color: #14C4FF; font-size: 3rem;'>🚀 Real-Time Aerial Image Segmentation</h1>
    <p style='text-align: center; font-size: 1.2rem;'>Upload satellite or aerial image and segment it using a pretrained model.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an aerial or urban image", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.stop()

# ----------------------------------------
# Read image
# ----------------------------------------
file_ext = uploaded_file.name.split(".")[-1].lower()
if file_ext in ["jpg", "jpeg", "png"]:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
elif file_ext in ["tif", "tiff"]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    atexit.register(lambda: os.remove(tmp_path) if os.path.exists(tmp_path) else None)
    with rasterio.open(tmp_path) as src:
        img = src.read()
        img = np.transpose(img, (1, 2, 0))
    if img.shape[2] >= 3:
        image_np = img[:, :, :3]
    else:
        st.error("TIF must have at least 3 channels.")
        st.stop()
    image = Image.fromarray((image_np / image_np.max() * 255).astype(np.uint8))
else:
    st.error("Unsupported file type.")
    st.stop()

# Show original
st.subheader("🖼️ Original Image")
st.image(image, use_column_width=True)

# ----------------------------------------
# Inference
# ----------------------------------------
if st.button("🧠 Run Segmentation"):
    with st.spinner("Segmenting..."):
        pred_mask = predict_image_with_patches(image_np, model)
        pred_rgb = utils.class_to_rgb(pred_mask, color_map_rgb)
        overlay = cv2.addWeighted(image_np, 0.6, pred_rgb, 0.4, 0)

    st.subheader("🧩 Segmentation Results")
    col1, col2 = st.columns(2)
    col1.image(pred_rgb, caption="Predicted Mask", use_column_width=True)
    col2.image(overlay, caption="Overlay", use_column_width=True)

    buffer = BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    st.download_button("💾 Download Overlay", buffer.getvalue(), file_name="segmentation_overlay.png", mime="image/png")

# ----------------------------------------
# Optional: Styling
# ----------------------------------------
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0F1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #14C4FF;
        color: white;
        border-radius: 12px;
        font-size: 1.1rem;
        padding: 0.6rem 1.6rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0072FF;
        transform: scale(1.05);
    }
    .stFileUploader>div>div {
        color: #FAFAFA;
    }
    </style>
""", unsafe_allow_html=True)
