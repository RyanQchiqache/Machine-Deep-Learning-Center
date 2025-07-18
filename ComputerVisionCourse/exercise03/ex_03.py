"""
Computer Vision SoSe 25 - Exercise 3: Detection and Segmentation
Author: Ryan Qchiqache
MatrNr: 12443412
LMU Munich, Prof. Dr. Björn Ommer

 This script implements object detection and segmentation using a pretrained ResNet-50, without retraining or modifying the network.
  Task 1: Object Detection using Grad-CAM
    1. Load pretrained ResNet-50 and print model layers
    2. Use Grad-CAM to highlight class-relevant regions via gradients and activations
    3. Register forward and backward hooks to capture layer4 outputs
    4. Compute and visualize Grad-CAM heatmaps and bounding boxes
    5. Compare class activations across different classes and top-5 predictions
  Task 2: Segmentation using Class Activation Maps
    1. Extract forward activations from the final convolutional layer
    2. Compute class activation maps using classifier weights
    3. Resize CAMs to input resolution and threshold for segmentation
    4. Visualize per-class segmentation masks and color-coded overlays
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List
import urllib.request

# ---------------------------------------------
# Task 1: Object Detection
# ---------------------------------------------

# Task 1.1: Load Pretrained Model and Print Layers
LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
class_names = urllib.request.urlopen(LABELS_URL).read().decode('utf-8').splitlines()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
model.eval()
print(model)

# Task 1.2 & 1.3: Register Hooks to Store Activations and Gradients
gradients: List[Tensor] = []
activations: List[Tensor] = []


def forward_hook(module: Module, input: Tuple[Tensor], output: Tensor) -> None:
    activations.append(output.detach())


def backward_hook(module: Module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> None:
    gradients.append(grad_output[0].detach())


target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Task 1.4: Grad-CAM Calculation and Visualization
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def calculate_grad_cam(gradients, activations):
    grad = gradients[-1]
    act = activations[-1]
    weights = grad.mean(dim=(2, 3))
    cam = (weights[0].view(-1, 1, 1) * act[0]).sum(0)
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.cpu().numpy()


def extract_bounding_boxes(cam, img, threshold=0.05):
    heatmap = (cam * 255).astype(np.uint8)
    _, binary_map = cv2.threshold(heatmap, int(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_np = np.array(img.resize((224, 224)))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img_np


img_path = 'input/Chihuahua.jpg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)

model.zero_grad()
gradients.clear()
activations.clear()
output = model(input_tensor)
pred_class = output.argmax().item()
print(f"Predicted class: {pred_class} - {class_names[pred_class]}")

output[0, pred_class].backward()
cam = calculate_grad_cam(gradients, activations)
# Extract bounding boxes for the detected class
bounding_box = extract_bounding_boxes(cam, img)

# Resize original image and convert to OpenCV BGR format
resized_img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)

# Convert CAM to heatmap — ensure shape and type are correct
cam_resized = cv2.resize(cam, (224, 224))  # Resize CAM to match image size
cam_uint8 = np.uint8(255 * cam_resized)  # Convert to 8-bit grayscale

# Apply color map to CAM (jet → BGR)
heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # (224, 224, 3)

# Convert original image to OpenCV BGR format
resized_img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)

# Sanity check shapes before blending
assert heatmap.shape == resized_img.shape, f"Mismatch: heatmap={heatmap.shape}, image={resized_img.shape}"

# Blend the image with the heatmap
overlay = cv2.addWeighted(resized_img, 0.5, heatmap, 0.5, 0)

# Convert to RGB for matplotlib display
overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# Plot Grad-CAM heatmap, bounding box, and smooth overlay
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM Heatmap")

plt.subplot(1, 3, 2)
plt.imshow(bounding_box)
plt.title("Detected Region")

plt.subplot(1, 3, 3)
plt.imshow(overlay_rgb)  # <- use RGB version here!
plt.title("Smooth Overlay")

plt.tight_layout()
plt.show()
"""visual = extract_bounding_boxes(cam, img)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM Heatmap")
plt.subplot(1, 2, 2)
plt.imshow(visual)
plt.title("Detected Region")
plt.show()"""

# Task 1.5 & 1.6: Top-5 Predictions Visualization
top5_classes = output[0].topk(5).indices.tolist()

for cls_id in top5_classes:
    model.zero_grad()
    gradients.clear()
    activations.clear()
    output = model(input_tensor)
    output[0, cls_id].backward(retain_graph=True)
    cam = calculate_grad_cam(gradients, activations)
    visual = extract_bounding_boxes(cam, img)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(cam, cmap='jet')
    plt.title(f"Grad-CAM: {class_names[cls_id]}")
    plt.subplot(1, 2, 2)
    plt.imshow(visual)
    plt.title("Bounding Box")
    plt.show()

# ---------------------------------------------
# Task 2: Segmentation with Classification Network
# ---------------------------------------------

# Task 2.1: Register Hook for Last Conv Layer
model.eval()
final_conv = model.layer4[-1]
activations = []


def task2_forward_hook(module: Module, input: Tuple[Tensor], output: Tensor):
    activations.append(output.detach())


final_conv.register_forward_hook(task2_forward_hook)

# Task 2.2: Load Image and Store Activation
img_path = 'ComputerVisionCourse/exercise03/input/Bulldog.png'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)
activations.clear()

with torch.no_grad():
    output = model(input_tensor)

top_classes = output[0].topk(3).indices.tolist()  #(idk like "Bulldog, Boxer, Golden retriever or smth")

# Task 2.3–2.5: Compute CAMs and Overlay
feature_map = activations[-1][0]
fc_weights = model.fc.weight.detach()  #shape (num_classes, C)
segmentation = np.zeros((224, 224), dtype=np.uint8)
original_img = np.array(img.resize((224, 224)))
overlay = np.zeros_like(original_img, dtype=np.uint8)

for i, cls in enumerate(top_classes):
    weights = fc_weights[cls]
    cam = torch.einsum('c,chw->hw', weights, feature_map)
    cam = torch.clamp(cam, min=0)# like relu
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cam.cpu().numpy()

    cam_resized = cv2.resize(cam, (224, 224))
    mask = (cam_resized > 0.3).astype(np.uint8)
    segmentation[mask == 1] = i + 1

    color = np.random.randint(0, 255, size=(3,))
    for c in range(3):
        overlay[:, :, c][mask == 1] = color[c]

blended = cv2.addWeighted(original_img, 0.5, overlay, 0.5, 0)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(segmentation, cmap='tab10')
plt.title('Segmentation Map')
plt.subplot(1, 3, 3)
plt.imshow(blended)
plt.title('Overlay')
plt.tight_layout()
plt.show()
