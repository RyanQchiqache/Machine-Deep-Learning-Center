import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torchvision.utils
from tqdm import tnrange
from torch.optim import Adam
import matplotlib.pyplot as plt
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision.transforms import transforms as T

# Constants
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transforms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.433), (0.5, 0.5, 0.5)),
    T.Resize((32, 32))
])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# model, DNN
class DeepNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model: nn.Module, train_loader: DataLoader, val_loader, EPOCHS, LR, device) -> None:
    criterion: nn = nn.CrossEntropyLoss()
    optimizer: torch.optim = Adam(model.parameters(), LR)

    model.train()

    for epoch in tnrange(EPOCHS + 1):
        running_loss: float = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                output = model(inputs)
                loss = criterion(output, 1)
                val_loss += loss.item()

                _, predicted = torch.max(output, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        model.train()


def teest(model: nn.Module, test_loader: DataLoader, device):
    model.to(device)
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            _, predicted = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.3f}")

    return accuracy


def show_images(images, labels, class_names, n=4):
    """Show n images from a batch with their labels."""
    images = images[:n]
    labels = labels[:n]
    images = images * 0.5 + torch.tensor([0.485, 0.456, 0.433]).view(1, 3, 1, 1)

    images = torch.clamp(images, 0, 1)

    fig, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    for i in range(n):
        img_np = images[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img_np)
        axs[i].set_title(class_names[labels[i].item()])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()


def predict_image(model, image_path, class_names, transform, device):
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    predicted_class = class_names[predicted.item()]
    print(f"Predicted class: {predicted_class}")


if __name__ == '__main__':
    print(f"Device: {device}")
    custom_image_path = "path/to/your/image.jpg"
    # Datasets
    full_train_set = dataset.CIFAR10(root="./data", train=True, download=True, transform=transforms)

    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size

    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])

    test_set = dataset.CIFAR10(root="./data", train=False, download=True, transform=transforms)

    # Dataloaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)

    # Show sample images
    """sample_batch = next(iter(train_loader))
    show_images(sample_batch[0], sample_batch[1], classes, n=4)"""

    # Training
    model = DeepNN(num_classes=NUM_CLASSES).to(device)
    try:
        train(model, train_loader, val_loader, EPOCHS, LR, device)
        teest(model, test_loader, device)
        predict_image(model, custom_image_path, classes, transforms, device)
    except KeyboardInterrupt:
        print("Training was stopped manually")