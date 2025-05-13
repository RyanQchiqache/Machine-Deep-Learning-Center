"""
Computer Vision SoSe 25 — Task 1: Simple Image Operations
Author: Ryan Qchiqache
LMU Munich, Prof. Dr. Björn Ommer

This script demonstrates fundamental image processing steps:
1. Load an image
2. Inspect and plot dimensions
3. Apply a random crop
4. Convert the crop to grayscale
5. Insert grayscale patch into original
6. Resize the result
I will have 2 examples with 2 random crops:
- first without pytorch tensors etc..
- second with pytorch (using permute :()
"""
from typing import Tuple, Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os


# Utility functions

def convert_to_np_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a NumPy array."""
    return np.array(image)


def print_channels(image: np.ndarray) -> Tuple[int, int, int]:
    """Return the dimensions of an image array as (height, width, channels)."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return height, width, channels


def random_crop(image: np.ndarray, crop_size: int) -> Tuple[np.ndarray, int, int]:
    """Crop the image randomly to the specified size."""
    height, width, _ = print_channels(image)
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    crop = image[y:y + crop_size, x:x + crop_size]
    return crop, x, y


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a NumPy image array to grayscale."""
    gray_image = Image.fromarray(image).convert("L")
    return np.array(gray_image)


def ensure_rgb(image: Image.Image) -> np.ndarray:
    """Ensure the image has 3 color channels (RGB)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def insert_patch(image: np.ndarray, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """Insert a grayscale patch into the original image."""
    patch_rgb = np.stack([patch] * 3, axis=-1)
    image_copy = image.copy()
    image_copy[y:y + patch.shape[0], x:x + patch.shape[1]] = patch_rgb
    return image_copy


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize the image by a given scale factor."""
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized = Image.fromarray(image).resize(new_size)
    return np.array(resized)


def plot_image(image: Union[Image.Image, np.ndarray], title: str) -> None:
    """Display an image using Matplotlib.pyplot"""
    if isinstance(image, np.ndarray):
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main(image_path: str, crop_size: int = 256) -> None:
    """Main function"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path)
    image_np = ensure_rgb(image)
    height, width, channels = print_channels(image_np)
    print(f"Loaded image: {height} x {width} x {channels}")

    plot_image(image_np, f"Original Image: {height} x {width} x {channels}")

    crop, x, y = random_crop(image_np, crop_size)
    plot_image(crop, f"Random Crop ({crop_size}x{crop_size}) at ({x}, {y})")

    gray_crop = convert_to_grayscale(crop)
    plot_image(gray_crop, "Grayscale Crop")

    modified_image = insert_patch(image_np, gray_crop, x, y)
    plot_image(modified_image, "Image with Grayscale Patch")

    resized_image = resize_image(modified_image, scale=0.5)
    plot_image(resized_image, f"Resized Image: {resized_image.shape[0]} x {resized_image.shape[1]}")


if __name__ == "__main__":
    # can change the crop_size in here (e.g.main("Capybara.jpg", 126))
    main("Capybara.jpg")















# 1. Load the image
img: Image.Image = Image.open("Capybara.jpg")
img_np: np.ndarray = np.array(img)
# 2. Print and plot with dimensions
height, width= img_np.shape[:2]
channels = 1 if img_np.ndim == 2 else img_np.shape[2]

plt.imshow(img_np)
plt.title(f"Original Image: {height}×{width}×{channels}")
plt.axis('off')
plt.show()

# 3. Plot a random 256x256 crop
crop_size = 256
x = random.randint(0, width - crop_size)
y = random.randint(0, height - crop_size)
crop_np = img_np[y:y+crop_size, x:x+crop_size]

plt.imshow(crop_np)
plt.title(f"Random Crop: 256×256 at ({x}, {y})")
plt.axis('off')
plt.show()

# 4. Convert the crop to grayscale and plot
crop_img = Image.fromarray(crop_np)
crop_gray = crop_img.convert("L")  # 'L' = grayscale
crop_gray_np = np.array(crop_gray)

plt.imshow(crop_gray_np, cmap='gray')
plt.title("Grayscale Crop (256×256)")
plt.axis('off')
plt.show()

# 5. Insert grayscale patch back into original image
# Convert to RGB if needed
if img.mode != 'RGB':
    img = img.convert('RGB')
    img_np = np.array(img)

# Convert grayscale to 3 channels (RGB) to paste back
crop_gray_rgb = np.stack([crop_gray_np]*3, axis=-1)

# Insert the patch back
img_copy = img_np.copy()
img_copy[y:y+crop_size, x:x+crop_size] = crop_gray_rgb

plt.imshow(img_copy)
plt.title("Image with Grayscale Patch Reinserted")
plt.axis('off')
plt.show()

# 6. Resize to half
half_size = (img_copy.shape[1] // 2, img_copy.shape[0] // 2)
resized = Image.fromarray(img_copy).resize(half_size)

plt.imshow(resized)
plt.title(f"Resized Image with Patch: {half_size[1]}×{half_size[0]}")
plt.axis('off')
plt.show()

