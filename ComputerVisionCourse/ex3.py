"""import cv2
from typing import List, Tuple
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image_path: str = "/Users/ryanqchiqache/PycharmProjects/Machine-Learning-Learning-Center/ComputerVisionCourse/exercise03/input/Chihuahua.jpg"
image = Image.open(image_path)
image_np = np.array(image)
print(image_np.shape)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")


def take_random_patch(image: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = image.shape[:2]
    if H < patch_size or W < patch_size:
        raise ValueError(f"Image is smaller than patch size")

    top = np.random.randint(0, H - patch_size + 1)
    left = np.random.randint(0, W - patch_size + 1)

    patch = image[top: top + patch_size, left: left + patch_size]

    return patch, (top, left)


def take_center_patch(image: np.ndarray, patch_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    H, W = image.shape[:2]
    top = (H - patch_size) // 2
    left = (W - patch_size) // 2
    patch = image[top: top + patch_size, left: left + patch_size]

    return patch, (top, left)


def gray_scale_patch(patch: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    print(f"gray:{gray.shape}, gray_rgb:{gray_rgb.shape}")
    return gray_rgb


def insert_patch(image: np.ndarray, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """"""Insert a grayscale patch into the original image.""""""
    patch_rgb = patch
    image_copy = image.copy()
    image_copy[y:y + patch.shape[0], x:x + patch.shape[1]] = patch_rgb
    return image_copy


def put_patches_in_images(image: np.ndarray, patches: List[np.ndarray],
                          positions: List[Tuple[int, int]]) -> Image.Image:
    canvas = image.copy()
    for patch, (top, left) in zip(patches, positions):
        canvas = insert_patch(canvas, patch, left, top)
    return Image.fromarray(canvas)


center_patch, rand_pos = take_center_patch(image_np, patch_size=256)
random_patch, center_pos = take_random_patch(image_np, patch_size=256)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(center_patch)
ax[0].set_title("center_patch")
ax[0].axis("off")
ax[1].imshow(random_patch)
ax[1].set_title(f"random patch ")
ax[1].axis("off")

plt.tight_layout()
plt.show()

gray_random: np.ndarray = gray_scale_patch(random_patch)
gray_center: np.ndarray = gray_scale_patch(center_patch)

patched_up_image: Image.Image = put_patches_in_images(image_np, [gray_random, gray_center], [center_pos, rand_pos])
plt.imshow(patched_up_image)
plt.title("final image with patches in grayscale")
plt.axis("off")
plt.show()"""

import numpy as np
from pprint import pprint

# Input image
x = np.array([
    [1, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1]
])

# Kernels
w1 = np.array([[1, 0], [0, 1]])
w2 = np.array([[1, 0], [1, 0]])
w3 = np.array([[1, 1], [0, 0]])
kernels = [w1, w2, w3]
bias = -1

def apply_conv_relu(x, kernel, bias, stride=1):
    kH, kW = kernel.shape
    H, W = x.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i*stride:i*stride+kH, j*stride:j*stride+kW]
            out[i, j] = np.sum(patch * kernel) + bias

    return np.maximum(0, out)

# Apply each kernel with stride (you can change stride here)
stride = 1
outputs = []
for kernel in kernels:
    result = apply_conv_relu(x, kernel, bias, stride)
    outputs.append(result)

# Stack into a 3D tensor
tensor_output = np.stack(outputs, axis=0)

# Print
print("Final output tensor shape:", tensor_output.shape)
pprint(tensor_output)
