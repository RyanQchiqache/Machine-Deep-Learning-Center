import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# --- Config ---
BATCH_SIZE = 256
EPOCHS = 10  # feel free to increase this for better results if you have the compute
TEMPERATURE = 0.5
EMBEDDING_DIM = 128
lr = 1e-3


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.transforms = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            index (int): Index of the sample in the base dataset.
        Returns:
            tuple: A tuple containing two augmented views of the same image.
        """
        x, _ = self.base_dataset[index]

        # TODO: apply the transformations to the image `x` to create two augmented views `xi` and `xj`
        x_i, x_j = self.transforms(x), self.transforms(x)
        return x_i, x_j

    def __len__(self):
        return len(self.base_dataset)


# --- Simple Modern ConvNet Encoder ---
class SimpleConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # very simple ConvNet architecture with the goal of being fast to train
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.shape[0], -1)


# --- Encoder and Projection Head ---
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleConvEncoder()
        self.projector = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, EMBEDDING_DIM))

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return F.normalize(z, dim=1)


# --- Contrastive Loss (NT-Xent) ---
def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Args:
        z1 (torch.Tensor): First set of embeddings, shape [B, D].
        z2 (torch.Tensor): Second set of embeddings, shape [B, D].
        temperature (float): Temperature parameter for scaling the similarity scores.
    Returns:
        torch.Tensor: A single scalar. The computed contrastive loss.
    """
    # TODO implement the contrastive loss function
    # you can check Algorithm 1 in the SimCLR paper (https://arxiv.org/pdf/2002.05709) for reference
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # Computing the  cosine similarity matrix: sim(i, j) = z_i · z_j / ||z_i|| ||z_j||
    sim = torch.mm(z, z.T)  # [2B, 2B]

    # Removing the self-similarity by masking diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)  # large negative value so exp(-large) ≈ 0

    # Applyiiing temperature scaling
    sim = sim / temperature

    # Labels: positives are at positions [i, i + B] and [i + B, i]
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)], dim=0).to(z.device)

    # Compute loss: cross-entropy over similarity rows
    loss = F.cross_entropy(sim, positives)

    # if I want symmetrized loss..
    """"loss_1 = F.cross_entropy(sim[:batch_size], positives[:batch_size])
    loss_2 = F.cross_entropy(sim[batch_size:], positives[batch_size:])
    return (loss_1 + loss_2) / 2"""

    return loss


def train(model, dataloader, epochs, optimizer, device, temperature):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch: {epoch + 1}", leave=True)
        for xi, xj in progress_bar:
            xi, xj = xi.to(device), xj.to(device)
            zi, zj = model(xi), model(xj)
            loss = contrastive_loss(z1=zi, z2=zj, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {total_loss:.4f}/{len(dataloader):.4f}")

        print("Saving model...")
        torch.save(model.state_dict(), "simclr_model.pth")
        print("Model saved as simclr_model.pth")


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device "{device}".')

    model = SimCLRModel()
    if not os.path.exists("simclr_model.pth"):
        print("Model file simclr_model.pth does not exist. Starting training from scratch.")
    else:
        print("Model file simclr_model.pth found. Loading existing model.")
        model.load_state_dict(torch.load("simclr_model.pth", map_location="cpu"))

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True)
    train_loader = DataLoader(SimCLRDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    try:
        train(model, train_loader, EPOCHS, optimizer, device, TEMPERATURE)
    except KeyboardInterrupt:
        print("training was interrupted by user")


if __name__ == "__main__":
    main()
