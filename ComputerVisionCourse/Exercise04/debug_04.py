import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import torchvision
import cv2
from PIL import Image
# =====================================
# Configuration
# =====================================
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use torchvision transforms (CPU-friendly)
train_tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ToTensor(),
])
test_tf = T.ToTensor()

dataset_vis = CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
classes = dataset_vis.classes

# collect 10 images per class for a grid
label_images = {c: [] for c in range(10)}
for img, label in dataset_vis:
    if len(label_images[label]) < 10:
        label_images[label].append(img)
    if all(len(lst) == 10 for lst in label_images.values()):
        break

fig, axs = plt.subplots(10, 10, figsize=(18, 18))
for class_idx in range(10):
    for i in range(10):
        img = label_images[class_idx][i].permute(1, 2, 0).numpy()
        ax = axs[class_idx][i]
        ax.imshow(img)
        ax.axis("off")
        if i == 0:
            ax.set_title(classes[class_idx])
plt.tight_layout()
plt.show()

# actual train/test datasets
train_data = CIFAR10(root="./data", train=True, download=False, transform=train_tf)
test_data  = CIFAR10(root="./data", train=False, download=False, transform=test_tf)

# =====================================
# Network (LeNet-style for CIFAR-10)
# =====================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),   # 32->28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 28->14
            nn.Conv2d(6, 16, 5, 1),  # 14->10
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 10->5
        )
        self.classification = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.classification(x)

def get_cr_optim(model: nn.Module):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    return criterion, optimizer

def create_dataloader(dataset):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: Adam, criterion: nn.CrossEntropyLoss, device: torch.device):
    model.train()
    running_correct, total = 0, 0
    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        preds = outputs.argmax(1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * running_correct / total

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

def train(model, train_loader, test_loader, epochs, criterion, optimizer, device):
    train_h= []
    test_h = []
    for epoch in range(epochs):
        train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc  = evaluate(model, test_loader, device)
        logger.info(f"Epoch {epoch+1}/{epochs} — Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        train_h.append(train_acc)
        test_h.append(test_acc)
    return train_h, test_h

def plot_history(train_hist, test_hist):
    plt.figure(figsize=(7,4))
    xs = range(1, len(train_hist) + 1)
    plt.plot(xs, train_hist, label="Train accuracy")
    plt.plot(xs, test_hist, label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Test Accuracy over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#==============================
# activation and saliency Maps
# =============================

model_resnet = torchvision.models.resnet50(weights=False).to(device)
print(model_resnet)

image_pug = cv2.imread("/home/q/qchiqache/PycharmProjects/Machine-Deep-Learning-Center/ComputerVisionCourse/Exercise04/pug.jpg", cv2.IMREAD_COLOR)
image_pug_cv = cv2.cvtColor(image_pug, cv2.COLOR_RGB2BGR)

pil_img = Image.fromarray(image_pug_cv)


transforming_dog = T.Compose([
    T.Resize((224,224)), T.ToTensor()
])
transformed_pug = transforming_dog(pil_img).unsqueeze(0).to(device)

image_np = (transformed_pug
             .squeeze(0)
            .permute(1,2,0)
            .cpu()
            .numpy())
print(transformed_pug.shape)
print(image_np.shape)

# Activation map
activation_map = {}
def get_activation(name:str):
    def hook (model, input, output):
        activation_map[name] = output.detach()
    return hook
model_resnet.conv1.register_forward_hook(get_activation("conv1"))
model_resnet.layer3[0].conv1.register_forward_hook(get_activation("layer3_conv1"))
model_resnet.layer4[0].conv1.register_forward_hook(get_activation("layer4_conv1"))

def debug_hook(module, input, output):
    print(f"\n[HOOK] Module: {module.__class__.__name__}")
    print(f"Input type: {type(input)}, len={len(input)}, shape(s)={[i.shape for i in input]}")
    print(f"Output type: {type(output)}, shape={output.shape}")
    print(f"First 5 output values (flattened): {output.flatten()[:5]}")

model_resnet.conv1.register_forward_hook(debug_hook)


def visualize_activations(activation_tensor, layer_name, num_channels=6):
    """
    Visualize the first `num_channels` feature maps from a layer.
    """
    act = activation_tensor.squeeze(0).cpu()

    plt.figure(figsize=(15, 5))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(act[i], cmap='viridis')
        plt.axis('off')
        plt.title(f"{layer_name} - ch {i}")
    plt.tight_layout()
    plt.show()

output=model_resnet(transformed_pug)

print("conv1 output shape:", activation_map['conv1'].shape)
print("conv1 output shape:", activation_map['layer3_conv1'].shape)
print("conv1 output shape:", activation_map['layer4_conv1'].shape)
visualize_activations(activation_map["conv1"], "conv1", 6)
visualize_activations(activation_map["layer3_conv1"], "layer3_conv1", 6)
visualize_activations(activation_map["layer4_conv1"], "layer4_conv1", 6)

if __name__ == "__main__":
    # ==============================
    # Simple CNN Model training
    # ==============================
    """model = Net().to(device)
    print(model)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", parameters)

    train_loader = create_dataloader(train_data)
    test_loader  = create_dataloader(test_data)

    criterion, optimizer = get_cr_optim(model)

    try:
        train_h, test_h = train(model, train_loader, test_loader, EPOCHS, criterion, optimizer, device)
        plot_history(train_h, test_h)
    except KeyboardInterrupt:
        logger.info("Training interrupted.")"""


    #==============================
    # activation and saliency Maps
    # =============================



