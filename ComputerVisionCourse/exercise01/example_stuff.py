import matplotlib.pyplot as plt
import numpy as np


image = np.array([
    [1, 2, 3, 4, 5],
    [5, 6, 7, 8, 9],
    [9, 8, 7, 6, 5],
    [5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5]
], dtype=np.float32)

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

padded = np.pad(image,((1,1),(1,1)), mode="constant", constant_values=0)
print(padded)
output = np.zeros_like(image)
print(output)

H, W = image.shape
kW, kH = kernel.shape

for i in range(H):
    for j in range(W):
        region = padded[i:i+kW, j:j+kW]
        output[i,j] = np.sum(region*kernel)
        print(f"At pixel ({i},{j}):\n{region}")


print("Original Image:")
print(image)

print("\nKernel:")
print(kernel)

print("\nConvolved Output:")
print(output)

print("================================================")
# ================================================
# Gaussian kernel
def gaussian_kernel(size=3, sigma= 1.0, mean=0.0):
    assert size % 2 == 1
    ax = np.linspace(-(size//2), size//2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-((xx - mean) ** 2 + (yy - mean) ** 2) / (2 * sigma ** 2))
    kernel /= 2 * np.pi * sigma ** 2
    kernel /= np.sum(kernel)

    return kernel


#create a 3x3 kernel
kernel = gaussian_kernel(3,1.0)
print(f"3x3 Gaussian kernel")
print(np.round(kernel, 4))

plt.imshow(kernel, cmap='gray')
plt.title("3x3 gaussian kernel")
plt.colorbar()
plt.show()


