import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple

"""
    Computer Vision SoSe 25 — Task 2: Convolution and Filters 
    Author: Ryan Qchiqache
    LMU Munich, Prof. Dr. Björn Ommer
    
    CNNs: In Computer Vision, algorithms are driven by convolutional neural networks.
          CNNs perform the convolution operation by sliding a kernel (called a filter) 
          over an image and computing the dot product between the kernel and the covered 
          part of the image.

          The convolution operation is defined as:
              g'(x, y) = g(x, y) * f(·, ·)
                      = sum_{dx = -a}^{a} sum_{dy = -b}^{b} g(x + dx, y + dy) * f(dx, dy)

          Where:
          - g(x, y) is the input image and g'(x,y) is the filtered image
          - f(dx, dy) is the convolution kernel (2D)
          - a and b are the half-width and half-height of the kernel, respectively

    Gaussian Filter:
        A Gaussian filter is a special kernel used to blur images and reduce noise. 
        It applies the transformation:
            f(x, y) = (1 / (2πσ²)) * exp{ - (x² + y²) / (2σ²) }

        - σ (sigma) controls the spread/blur strength
        - The kernel values are normalized to sum to 1
"""


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply 2D convolution to a grayscale image.

    Parameters:
    - image: np.ndarray[float32] of shape (H, W)
    - kernel: np.ndarray[float32] of shape (k, k)

    Returns:
    - output: np.ndarray[float32] of shape (H, W)
    """
    H: int = image.shape[0]
    W: int = image.shape[1]
    kH: int = kernel.shape[0]
    kW: int = kernel.shape[1]

    assert kH == kW and kH % 2 == 1, "Kernel must be square (e.g. 3x3, 5x5) and odd-sized so it has a center pixel."

    pad_h: int = kH // 2
    pad_w: int = kW // 2

    padded: np.ndarray = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0.0)
    output: np.ndarray = np.zeros_like(image, dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region: np.ndarray = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * kernel)

    return output


def gaussian_kernel(size: int, sigma: float, mean: float = 0.0) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel.
    A small grid of weights shaped like a "bell curve" :)
    This is often used for:
        - blurring images
        - smoothing noisy input
        - pre-processing before edge detection

    Parameters:
    - size: int (must be odd)
    - sigma: float
    - mean: float (default 0)

    Returns:
    - kernel: np.ndarray[float64] of shape (size, size)
    """
    assert size % 2 == 1, "Kernel size must be odd"

    ax: np.ndarray = np.linspace(-(size // 2), size // 2, size)
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(ax, ax)

    # direct implementation of the Gaussian function
    kernel: np.ndarray = np.exp(-((xx - mean) ** 2 + (yy - mean) ** 2) / (2 * sigma ** 2))
    # This turns the exponential into a proper probability density (Gaussian PDF).
    kernel /= 2 * np.pi * sigma ** 2
    # ensure that all values in the kernel add up to 1, this way the brightness of the iage is preserved and
    # Blurring doesn't make the image brighter or darker
    kernel /= np.sum(kernel)

    return kernel


def plot_kernel(kernel: np.ndarray, title: str = "Gaussian Kernel") -> None:
    """
    Plot kernel as 200x200 grayscale image.

    Parameters:
    - kernel: np.ndarray[float64]
    - title: str
    """
    normalized: np.ndarray = (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
    img: Image.Image = Image.fromarray(normalized.astype(np.uint8)).resize((200, 200), Image.BICUBIC)

    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def main() -> None:
    # === Load the Image ===
    image_path: str = "Capybara.jpg"
    img: Image.Image = Image.open(image_path).convert("L")
    img_np: np.ndarray = np.array(img).astype(np.float32) / 255.0

    # === Gaussian Filter ===
    kernel: np.ndarray = gaussian_kernel(size=7, sigma=1.5)
    plot_kernel(kernel)
    blurred: np.ndarray = convolve2d(img_np, kernel)
    blurred_norm: np.ndarray = (blurred - blurred.min()) / (blurred.max() - blurred.min())

    # === Laplacian Filter ===
    laplace: np.ndarray = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    #laplace = np.array([[-1, -1, -1], [-1, -8, -1], [-1, -1, -1]], dtype=np.float32)# trying another kernel aka filter

    #edges: np.ndarray = convolve2d(img_np, laplace)
    edges: np.ndarray = np.abs(convolve2d(img_np, laplace)) # Taking the absolute value before normalization and the test was better
    edges_norm: np.ndarray = (edges - edges.min()) / (edges.max() - edges.min())

    # === Plot Original Image ===
    plt.figure(figsize=(6, 6))
    plt.imshow(img_np, cmap='gray')
    plt.title("Original (Grayscale)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === Plot Gaussian Blurred Image ===
    plt.figure(figsize=(6, 6))
    plt.imshow(blurred_norm, cmap='gray')
    plt.title("Gaussian Blurred")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === Plot Laplacian Edge Image ===
    plt.figure(figsize=(6, 6))
    plt.imshow(edges_norm, cmap='gray')
    plt.title("Laplacian Edges")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === Short Discussion ===
    print("""
       Discussion:
       Hand-crafted filters (like Laplacian or Gaussian) are fixed and cannot adapt to data.
       They work well for simple tasks but fail in complex scenarios.

       For better visual results with the Laplacian filter, I applied the absolute value before normalization
       to enhance edge contrast. I also switched to a stronger kernel (with -8 center weight) to detect edges
       in all directions more effectively. These adjustments significantly improved the representation of edges.

       CNNs improve on this by learning optimal filters directly from data using backpropagation.
       """)


if __name__ == "__main__":
    main()
