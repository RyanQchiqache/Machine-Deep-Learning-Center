"""
Computer Vision SoSe 25 - Task 3: Introduction to PyTorch
Author: Ryan Qchiqache
LMU Munich, Prof. Dr. Björn Ommer

This script demonstrates:
1. Tensor format conversion between NumPy and PyTorch
2. Manual 2D convolution vs. PyTorch's nn.Conv2d
3. Application of a Gaussian filter using nn.Conv2d
"""

import numpy as np
import torch
from torch import nn, Tensor
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple


def convert_image_to_tensor(image_path: str) -> Tuple[Tensor, np.ndarray]:
    """
    Load an RGB image, convert it to a tensor, and demonstrate format conversions.
    """
    image = Image.open(image_path).convert("RGB")
    tensor_chw = TF.to_tensor(image)
    print(tensor_chw.shape)

    tensor_hwc = tensor_chw.permute(1, 2, 0)
    tensor_back = tensor_hwc.permute(2, 0, 1)

    image_reconstructed = tensor_back.permute(1, 2, 0).numpy()
    Image.fromarray((image_reconstructed * 255).astype(np.uint8)).save("restored.jpg")

    return tensor_chw, image_reconstructed


def manual_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manually perform a 2D convolution on a single-channel image.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    output = np.zeros((H - kH + 1, W - kW + 1), dtype=np.float32)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = image[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * kernel)

    return output


def compare_manual_vs_torch_conv():
    """
    Compare manual convolution output with PyTorch nn.Conv2d output.
    """
    x_np = np.random.rand(5, 5, 1).astype(np.float32)
    w_np = np.random.rand(2, 2, 1).astype(np.float32)

    x_tensor = torch.from_numpy(x_np.transpose(2, 0, 1)).unsqueeze(0)  # [1, 1, 5, 5]
    w_tensor = torch.from_numpy(w_np.transpose(2, 0, 1)).unsqueeze(0)  # [1, 1, 2, 2]

    conv = nn.Conv2d(1, 1, kernel_size=2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(w_tensor)

    output_torch = conv(x_tensor).detach().numpy()[0, 0]
    output_manual = manual_convolve2d(x_np[:, :, 0], w_np[:, :, 0])

    max_diff = np.max(np.abs(output_torch - output_manual))
    print(f"Max difference between manual and torch Conv2d: {max_diff:.6f}")


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def apply_gaussian_filter(image_path: str, kernel_size: int = 7, sigma: float = 1.5):
    """
    Apply a Gaussian filter to a grayscale image using both manual and nn.Conv2d methods.
    """
    image_gray = Image.open(image_path).convert("L")
    img_np = np.array(image_gray).astype(np.float32) / 255.0

    kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)

    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(kernel_tensor)

    result_torch = conv(img_tensor).detach().squeeze().numpy()
    result_manual = manual_convolve2d(img_np, kernel)

    return result_torch, result_manual


def plot_gaussian(result_image, result_torch):

    plt.figure(figsize=(6,6))
    plt.imshow(result_image, cmap="gray")
    plt.title("Manual Gaussian Blur")
    plt.axis("off")

    plt.figure(figsize=(6, 6))
    plt.imshow(result_torch, cmap='gray')
    plt.title("PyTorch Gaussian Blur")
    plt.axis('off')
    plt.show()



def main():
    """
    Entry point: perform Task 3 steps sequentially.
    """
    image_path = "Capybara.jpg"

    print("--- Task 3.1: Format Conversion ---")
    _, _ = convert_image_to_tensor(image_path)

    print("--- Task 3.2: Custom vs PyTorch Convolution ---")
    compare_manual_vs_torch_conv()

    print("--- Task 3.3: Gaussian Filter via nn.Conv2d ---")
    manual, torch = apply_gaussian_filter(image_path)
    plot_gaussian(manual, torch)


if __name__ == "__main__":
    main()
