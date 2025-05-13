"""import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- CONFIG ---
DLR_IMAGE_DIRS = [
    "DLR_semantic_segmentation/SS_Dense/train",
    "DLR_semantic_segmentation/SS_Dense/val",
    "DLR_semantic_segmentation/SS_Dense/test"
]
DLR_MASK_DIRS = [
    "DLR_semantic_segmentation/SS_Dense/train/labels/rgb",
    "DLR_semantic_segmentation/SS_Dense/val/labels/rgb",
]
UAE_TILES_DIR = "Semantic segmentation dataset"
OUTPUT_IMAGE_DIR = "UnifiedPatches/images"
OUTPUT_MASK_DIR = "UnifiedPatches/masks"
PATCH_SIZE = 512
STRIDE = 512

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# --- COLOR TO CLASS MAPPING (DLR) ---
COLOR2CLASS_DLR = {
    (244, 35, 232): 4,  # low vegetation
    (102, 102, 156): 1,  # paved road
    (190, 153, 153): 1,  # non paved road
    (153, 153, 153): 1,  # paved parking
    (250, 170, 30): 1,  # non paved parking
    (220, 20, 60): 2,  # building
    (125, 0, 142): 3,  # car
    (170, 200, 40): 3,  # trailer
    (200, 60, 100): 3,  # van
    (150, 18, 80): 3,  # truck
    (81, 180, 81): 3,  # long truck
    (190, 241, 21): 3,  # bus
    (70, 70, 70): 4  # tree
}

UAE_LABEL_MAP = {
    0: 5,  # Unlabeled → optional ignore class or map to background
    1: 0,  # Water → background or separate if needed
    2: 4,  # Land (unpaved area) → vegetation/terrain
    3: 1,  # Road
    4: 2,  # Building
    5: 4  # Vegetation
}

img_id = 1


# --- UTILITIES ---
def remap_dlr_mask(rgb_mask):
    arr = np.array(rgb_mask)
    new_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
    for color, class_id in COLOR2CLASS_DLR.items():
        mask = np.all(arr == color, axis=-1)
        new_mask[mask] = class_id
    return new_mask


def remap_uae_mask(mask):
    arr = np.array(mask)
    new_mask = np.zeros_like(arr)
    for old, new in UAE_LABEL_MAP.items():
        new_mask[arr == old] = new
    return new_mask


def save_patch(img_patch, mask_patch):
    global img_id
    img_out = cv2.resize(img_patch, (PATCH_SIZE, PATCH_SIZE))
    mask_out = cv2.resize(mask_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{img_id:05d}.jpg", img_out)
    Image.fromarray(mask_out).save(f"{OUTPUT_MASK_DIR}/{img_id:05d}.png")
    img_id += 1


# --- PROCESS DLR CROPS ---
print("Processing DLR crops...")
for img_dir, mask_dir in zip(DLR_IMAGE_DIRS, DLR_MASK_DIRS):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    for file in tqdm(img_files):
        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", ".png"))

        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path)
        mask_rgb = Image.open(mask_path).convert("RGB")
        mask = remap_dlr_mask(mask_rgb)

        H, W = img.shape[:2]
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                img_patch = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                mask_patch = mask[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                save_patch(img_patch, mask_patch)

# --- PROCESS UAE FIXED SIZE ---
print("Processing UAE resize/pad...")
for tile in sorted(os.listdir(UAE_TILES_DIR)):
    tile_path = os.path.join(UAE_TILES_DIR, tile)
    img_dir = os.path.join(tile_path, "images")
    mask_dir = os.path.join(tile_path, "masks")

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        continue

    for file in tqdm(sorted(os.listdir(img_dir))):
        if not file.endswith(".jpg"): continue

        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file.replace(".jpg", ".png"))

        img = cv2.imread(img_path)
        mask = Image.open(mask_path)
        mask_arr = remap_uae_mask(mask)

        img = cv2.resize(img, (PATCH_SIZE, PATCH_SIZE))
        mask_arr = cv2.resize(mask_arr, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"{OUTPUT_IMAGE_DIR}/{img_id:05d}.jpg", img)
        Image.fromarray(mask_arr).save(f"{OUTPUT_MASK_DIR}/{img_id:05d}.png")
        img_id += 1

print(f"\n✅ Done. Total patches created: {img_id - 1}")




"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def normalize_band(band):
    """Normalize a single band to the range [0, 255]."""
    band = band.astype(np.float32)
    band = (band - np.min(band)) / (np.max(band) - np.min(band)) * 255
    return band.astype(np.uint8)


def process_tif_image(input_path, output_dir):
    """Process and visualize a .tif image."""
    try:
        # Open the .tif file using rasterio
        with rasterio.open(input_path) as src:
            print(f"Processing: {os.path.basename(input_path)}")
            band_count = src.count
            print(f"Number of bands: {band_count}")

            # Check if the file has enough bands (RGB)
            if band_count < 3:
                print("The file does not contain enough bands to form an RGB image.")
                return

            # Read the first three bands (assuming RGB) and normalize
            red = normalize_band(src.read(1))
            green = normalize_band(src.read(2))
            blue = normalize_band(src.read(3))

            # Stack bands into an RGB image
            rgb_image = np.dstack((red, green, blue))

            # Display the image
            plt.figure(figsize=(8, 8))
            plt.imshow(rgb_image)
            plt.title(f"Processed: {os.path.basename(input_path)}")
            plt.axis('off')
            plt.show()

            # Save the image as PNG
            output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_path))[0]}.png")
            plt.imsave(output_path, rgb_image.astype(np.uint8))
            print(f"Saved processed image to: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_directory(directory):
    """Process all .tif files in a given directory."""
    output_dir = os.path.join(directory, "processed_images")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            input_path = os.path.join(directory, filename)
            process_tif_image(input_path, output_dir)


# Set your LandCover.ai dataset directory path here
dataset_dir = "M-33-7-A-d-2-3.tif"
process_directory(dataset_dir)

