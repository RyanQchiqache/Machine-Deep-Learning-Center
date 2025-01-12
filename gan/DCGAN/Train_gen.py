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
import logging
import colorlog

def setup_logger():
    """Setup logger with colorlog."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
    ))
    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

logger = setup_logger()

def signal_handler(sig, frame):
    logger.info('You pressed Ctrl+C! Saving logs and exiting...')
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
    logger.info(f"Using device: {device}")

    # Hyperparameters
    LEARNING_RATE = 2e-4  # seen in most GAN models
    BATCH_SIZE = 64       # augmenting batch to try out
    IMAGE_SIZE = 64       # Size to which images will be resized
    CHANNELS_IMAGE = 1    # Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB)
    Z_DIM = 100           # Dimensionality of the latent vector (input to the generator)
    EPOCHS = 5           # Number of epochs
    FEATURES_DISC = 64    # Number of features in the discriminator
    FEATURES_GEN = 64     # Number of features in the generator
    BETA1 = 0.5           # For Adam optimizer

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE)),           # Resize images to the specified size
            transforms.ToTensor(),                     # Convert images to PyTorch tensors
            transforms.Normalize(                      # Normalize images to the range [-1, 1]
                [0.5 for _ in range(CHANNELS_IMAGE)],  # Mean normalization value for each channel
                [0.5 for _ in range(CHANNELS_IMAGE)]   # Standard deviation normalization value for each channel
            ),
        ]
    )

    # Dataset and DataLoader
    try:
        dataset = datasets.MNIST(
            root="dataset/",      # Root directory of the dataset
            train=True,           # Specify training set
            transform=transform,  # Apply defined transformations
            download=True,        # Download dataset if not available locally
        )
        logger.info("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Start with 0 workers
    )
    logger.info(f"Dataloader initialized with batch size {BATCH_SIZE}.")

    # Initialize the Discriminator and Generator models
    gen = Generator(Z_DIM, CHANNELS_IMAGE, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMAGE, FEATURES_DISC).to(device)

    # Initialize weights
    initialize_weights(gen)
    initialize_weights(disc)
    logger.info("Models initialized and weights set.")

    # Define optimizers for both models
    optimizer_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    logger.info("Optimizers defined.")

    # Loss function (Binary Cross Entropy)
    criterion = nn.BCELoss()
    logger.info("Loss function defined.")

    # For logging to TensorBoard
    global writer_real, writer_fake  # Make writers global to access them in the signal handler
    writer_real = SummaryWriter(log_dir='logs/real')
    writer_fake = SummaryWriter(log_dir='logs/fake')
    step = 0

    # Create directory for saving models
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints directory created at {checkpoint_dir}.")

    # Fixed noise for consistent evaluation
    fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
    logger.info("Fixed noise generated for evaluation.")

    # Training Loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        logger.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        for batch_idx, (real, _) in enumerate(dataloader):
            logger.debug(f"Processing batch {batch_idx}")
            batch_start_time = time.time()
            real = real.to(device)
            noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
            fake = gen(noise)

            # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            ## Discriminator update##
            # Forward pass fake batch through Discriminator
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))  # Loss for real data

            # Forward pass real batch through Discriminator
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))  # Loss for fake data

            #Total Discriminator loss
            loss_disc = (loss_disc_real + loss_disc_fake) / 2  # Combined loss

            # Backpropagation for Discriminator
            disc.zero_grad() # Zero the gradients
            loss_disc.backward() # Backpropagate the loss
            optimizer_disc.step() # Update the weights

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
            ### Generator Update ###
            # Forward pass fake batch through Discriminator again, but do not detach this time
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))  # Generator loss

            # Backpropagation for Generator
            gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Print losses and log to TensorBoard or Logging to TensorBoard
            if batch_idx % 100 == 0:
                logger.info(
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
        logger.info(f"Epoch [{epoch}/{EPOCHS}] completed in {epoch_duration:.2f} seconds.")

        # Save the model checkpoints
        torch.save(gen.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch}.pth")
        torch.save(disc.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch}.pth")
        logger.info(f"Model checkpoints saved for epoch {epoch}.")

    # Close the TensorBoard writers
    writer_real.close()
    writer_fake.close()
    logger.info("Training completed and TensorBoard writers closed.")

if __name__ == '__main__':
    main()
