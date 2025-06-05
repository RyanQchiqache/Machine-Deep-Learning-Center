import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Task 1.1:

# Download the ImageNet class index file
import urllib.request

LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
class_names = urllib.request.urlopen(LABELS_URL).read().decode('utf-8').splitlines()


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
model.eval()
##-----------------Your code here-----------------##
# Print the layers of ResNet-50
print(model)

##-----------------Your code above-----------------##


# Task 1.2 and 1.3:
# Implement functions or code snippet to store the intermediate 
# activation (forward pass) and gradient (backward pass) for the `layer4`


##-----------------Your code here-----------------##
# We define hook storage
gradients: List[Tensor] = []
activations: List[Tensor] = []


def forward_hook(module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
    activations.append(output.detach())


def backward_hook(module: Module, grad_input: Tuple[Tensor, ...], grad_output: Tuple[Tensor, ...]) -> None:
    gradients.append(grad_output[0].detach())


# since we want to implement storage of activations and gradients from layer4 using hooks
target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)




##-----------------Your code above-----------------##

# Task 1.4-1
# Preprocessing for the input image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
img_path = 'exercise03/input/Chihuahua.jpg'
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)


# You dont have to touch this function
def extract_bounding_boxes(cam, img, threshold=0.3):
    """
    Extract bounding boxes from the Grad-CAM heatmap and draw them on the image.
    You can adjust the threshold to control the sensitivity of the bounding box extraction.
    The higher the threshold, the fewer boxes will be detected.
    """
    # Convert the Grad-CAM heatmap to an 8-bit image
    heatmap = (cam * 255).astype(np.uint8)
    # Apply a binary threshold to the heatmap to isolate the most salient regions
    _, binary_map = cv2.threshold(heatmap, threshold * 255, 255, cv2.THRESH_BINARY)

    # Find contours in the binary map to detect connected regions
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Resize the original image to match the Grad-CAM heatmap size (224x224)
    img_np = np.array(img.resize((224, 224)))

    # Collect bounding boxes
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])  # [x1, y1, x2, y2]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_np


def calculate_grad_cam(gradients, activations):
    """
    Calculate the Grad-CAM heatmap.
    """

    ##-----------------Your code here-----------------##

    #Step 1: we get the last stored activation and gradient
    grad = gradients[-1]  # which is [1, C, H, W]
    act = activations[-1]  # which is also like grad

    #Step 2: we compute the weights (GAP (global average pooling over H and W))
    weights = torch.mean(grad, dim=(2, 3))  # which is [1, C]

    #Step 3: weight sum of activations
    cam = torch.zeros(act.shape[2:], dtype=torch.float32)
    for i in range(weights.shape[1]):
        cam += weights[0, i] * act[0, i]

    # Step 4: we apply relu from scratch :)
    cam = cam * (cam > 0)

    #Step5: we normalize to [0, 1]
    cam -= cam.min()
    cam /= cam.max() + 1e-8

    # then we convert to numpy
    cam_np = cam.cpu().numpy()

    return cam_np

    ##-----------------Your code above-----------------##


# Task 1.4-2: no code needed here if you implement calculate_grad_cam correctly

# Forward pass
model.zero_grad()
gradients.clear()
activations.clear()
output = model(input_tensor)

pred_class = output.argmax().item()
print(f"Predicted class: {pred_class} - {class_names[pred_class]}")

# Backward pass to get gradients for the predicted class
output = model(input_tensor)
model.zero_grad()
class_loss = output[0, pred_class]
class_loss.backward()

# Grad-CAM calculation
cam = calculate_grad_cam(gradients, activations)

# Extract bounding boxes
bounding_box = extract_bounding_boxes(cam, img)

# Show result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cam, cmap='jet')
plt.title("Grad-CAM Heatmap")

plt.subplot(1, 2, 2)
plt.imshow(bounding_box)
plt.title("Detected Region")
plt.show()

# task 1.5: before handing another class, you would like to clean up your gradients and activations
##-----------------Your code here-----------------##
output = model(input_tensor)
model.zero_grad()
gradients.clear()
activations.clear()
##-----------------Your code above-----------------##

# Forward pass
output = model(input_tensor)
# id of the predicted class
top5_classes = output.argsort(descending=True)[0, :5].tolist()

for idx in range(5):
    print(f"Predicted class: {top5_classes[idx]} - {class_names[top5_classes[idx]]}")

##-----------------Your code here-----------------##
second_class = top5_classes[1]
model.zero_grad()
gradients.clear()
activations.clear()
output = model(input_tensor)
output[0, second_class].backward()
cam = calculate_grad_cam(gradients, activations)
visual = extract_bounding_boxes(cam, img)


plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(cam, cmap='jet')
plt.title(f"Grad-CAM: {class_names[second_class]}")

plt.subplot(1, 2, 2)
plt.imshow(visual)
plt.title("Bounding Box")
plt.show()

##-----------------Your code above-----------------##


# task 1.6: same principle as before.
##-----------------Your code here-----------------##
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

# Task 2

# Task 2.1
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
model.eval()

##-----------------Your code here-----------------##
final_conv = model.layer4[-1]
activations = []

def task2_forward_hook(module: Module, input: Tuple[Tensor, ...], output: Tensor) -> None:
    activations.append(output.detach())

final_conv.register_forward_hook(task2_forward_hook)

##-----------------Your code above-----------------##


# Task 2.2
activations.clear()

# === Load and preprocess image ===
img_path = 'exercise03/input/Bulldog.png'  # Replace with your image path
img = Image.open(img_path).convert('RGB')  # Ensure RGB format
input_tensor = preprocess(img).unsqueeze(0)  # Apply transforms and add batch dimension

##-----------------Your code here-----------------##
# store the activations of the final conv layer
##-----------------Your code above-----------------##


# === Forward pass through the model ===
with torch.no_grad():
    output = model(input_tensor)  # Forward pass without gradients

# === Select top-k predicted classes ===
topk = 3  # Number of top classes to visualize, you can change this
top_classes = output[0].topk(topk).indices.tolist()  # Get top 3 predicted class indices

# Task 2.3
##-----------------Your code here-----------------##
# === Extract the activation map and classifier weights ===
if not activations:
    raise RuntimeError("Activation hook did not capture output.")

feature_map = activations[-1][0]  # shape: [2048, H, W]
fc_weights = model.fc.weight.detach()  # shape: [1000, 2048]
segmentation = np.zeros((224, 224), dtype=np.uint8)
original_img = np.array(img.resize((224, 224)))
overlay = np.zeros_like(original_img, dtype=np.uint8)
##-----------------Your code above-----------------##


# === Loop over top-3 classes to generate class-specific CAMs and segmentations ===
for i, cls in enumerate(top_classes):
    weights = fc_weights[cls]  # Get weights for this class (shape: [2048])

    ##-----------------Your code here-----------------##
    # Compute CAM via weighted sum over feature map channels
    cam = torch.einsum('c,chw->hw', weights, feature_map)  # shape: [H, W]
    cam = torch.clamp(cam, min=0)  # ReLU
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cam.cpu().numpy()
    ##-----------------Your code above-----------------##

    # Task 2.4
    ##-----------------Your code here-----------------##
    cam_resized = cv2.resize(cam, (224, 224))
    mask = (cam_resized > 0.3).astype(np.uint8)  # Threshold for binary mask

    segmentation[mask == 1] = i + 1  # assigning a unique class label (1, 2, 3...)

    # Assign a color (BGR) — you can customize this
    color = np.random.randint(0, 255, size=(3,))
    for c in range(3):
        overlay[:, :, c][mask == 1] = color[c]

    ##-----------------Your code above-----------------##

# Task 2.5
##-----------------Your code here-----------------##
# === Assign black color to background (pixels not covered by any top class) ===
# from chatgpt since I didn't know how to do this at all :/
blended = cv2.addWeighted(original_img, 0.5, overlay, 0.5,0)
##-----------------Your code above-----------------##
# === Show the results ===
plt.figure(figsize=(12, 4))

# show the original image* :)
plt.subplot(1, 3, 1)
plt.imshow(original_img)
plt.title('Original')

# show the segmentation map
plt.subplot(1, 3, 2)
plt.imshow(segmentation, cmap='tab10')
plt.title('Segmentation Map')

# show the overlay
plt.subplot(1, 3, 3)
plt.imshow(blended)
plt.title('Overlay')

plt.tight_layout()
plt.show()

