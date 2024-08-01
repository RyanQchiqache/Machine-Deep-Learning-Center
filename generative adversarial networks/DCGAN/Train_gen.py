import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Model_gen import Discriminator, Generator, initialize_weights
import os
import time
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Saving logs and exiting...')
    writer_real.close()
    writer_fake.close()
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def main():
    """
    Run tensorboard --logdir=logs --load_fast=true in the Terminal to check the tensorboard
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 32  # Reduced batch size for debugging
    IMAGE_SIZE = 64  # Size to which images will be resized
    CHANNELS_IMAGE = 1  # Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB)
    Z_DIM = 100  # Dimensionality of the latent vector (input to the generator)
    EPOCHS = 5
    FEATURES_DISC = 64  # Number of features in the discriminator
    FEATURES_GEN = 64  # Number of features in the generator

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE)),  # Resize images to the specified size
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(  # Normalize images to the range [-1, 1]
                [0.5 for _ in range(CHANNELS_IMAGE)],  # Mean normalization value for each channel
                [0.5 for _ in range(CHANNELS_IMAGE)]   # Standard deviation normalization value for each channel
            ),
        ]
    )

    # Dataset and DataLoader
    try:
        dataset = datasets.MNIST(
            root="dataset/",  # Root directory of the dataset
            train=True,  # Specify training set
            transform=transform,  # Apply defined transformations
            download=True,  # Download dataset if not available locally
        )
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Start with 0 workers
    )
    print(f"Dataloader initialized with batch size {BATCH_SIZE}.")

    # Initialize the Discriminator and Generator models
    gen = Generator(Z_DIM, CHANNELS_IMAGE, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMAGE, FEATURES_DISC).to(device)

    # Initialize weights
    initialize_weights(gen)
    initialize_weights(disc)
    print("Models initialized and weights set.")

    # Define optimizers for both models
    optimizer_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    print("Optimizers defined.")

    # Loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()
    print("Loss function defined.")

    # For logging to TensorBoard
    global writer_real, writer_fake  # Make writers global to access them in the signal handler
    writer_real = SummaryWriter(log_dir='logs/real')
    writer_fake = SummaryWriter(log_dir='logs/fake')
    step = 0

    # Create directory for saving models
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints directory created at {checkpoint_dir}.")

    # Fixed noise for consistent evaluation
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    print("Fixed noise generated for evaluation.")

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        for batch_idx, (real, _) in enumerate(dataloader):
            print(f"Processing batch {batch_idx}")
            batch_start_time = time.time()
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))  # Loss for real data
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # Loss for fake data
            loss_disc = (loss_disc_real + loss_disc_fake) / 2  # Combined loss
            disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))  # Generator loss
            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Print losses and log to TensorBoard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} "
                    f"Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}, "
                    f"Batch Time: {time.time() - batch_start_time:.2f} seconds"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)  # Generate images using fixed noise for consistent visualization
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                    # Log scalar values for losses
                    writer_real.add_scalar("Discriminator Loss", loss_disc.item(), global_step=step)
                    writer_fake.add_scalar("Generator Loss", loss_gen.item(), global_step=step)

                    # Ensure data is written to the logs
                    writer_real.flush()
                    writer_fake.flush()

                step += 1

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{EPOCHS}] completed in {epoch_duration:.2f} seconds.")

        # Save the model checkpoints
        torch.save(gen.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch}.pth")
        torch.save(disc.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch}.pth")
        print(f"Model checkpoints saved for epoch {epoch}.")

    # Close the TensorBoard writers
    writer_real.close()
    writer_fake.close()
    print("Training completed and TensorBoard writers closed.")

if __name__ == '__main__':
    main()
