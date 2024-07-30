import torch
import torch.nn as nn

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, channel_image, feature_channel):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # Initial convolutional layer
            nn.Conv2d(channel_image, feature_channel, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # Adding additional convolutional blocks
            self.block_(feature_channel, feature_channel * 2, 4, 2, 1),
            self.block_(feature_channel * 2, feature_channel * 4, 4, 2, 1),
            self.block_(feature_channel * 4, feature_channel * 8, 4, 2, 1),
            # Final convolutional layer to produce a single output
            nn.Conv2d(feature_channel * 8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    # Define a helper method to create a block with Conv2d, BatchNorm2d, and LeakyReLU layers
    def block_(self, channel_in, channel_out, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(0.2),
        )

    # Forward pass through the discriminator
    def forward(self, x):
        return self.discriminator(x)


# Define the Generator network
class Generator(nn.Module):
    def __init__(self, z_dimension, channel_image, feature_channel):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            # Initial block to transform latent vector
            self.block_(z_dimension, feature_channel * 16, 4, 1, 0),
            self.block_(feature_channel * 16, feature_channel * 8, 4, 2, 1),
            self.block_(feature_channel * 8, feature_channel * 4, 4, 2, 1),
            self.block_(feature_channel * 4, feature_channel * 2, 4, 2, 1),
            # Final ConvTranspose2d layer to produce the image
            nn.ConvTranspose2d(feature_channel * 2, channel_image, 4, 2, 1, bias=False),
            nn.Tanh()  # Tanh activation to get values in the range [-1, 1]
        )

    # Define a helper method to create a block with ConvTranspose2d, BatchNorm2d, and ReLU layers
    def block_(self, channel_in, channel_out, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(True),  # ReLU activation
        )

    # Forward pass through the generator
    def forward(self, x):
        return self.generator(x)


# Function to initialize the weights of the model
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)


# Function to test the Discriminator and Generator
def test():
    # Define the dimensions for testing
    N, channels, height, width = 8, 3, 64, 64
    z_dim = 100  # Latent vector size for generator

    # Create random input tensor for the discriminator
    t = torch.randn(N, channels, height, width)
    discriminator = Discriminator(channels, 8)
    initialize_weights(discriminator)
    # Ensure the discriminator output shape is as expected
    assert discriminator(t).shape == (N, 1, 1, 1)

    # Create random latent vector for the generator
    generator = Generator(z_dim, channels, 8)
    initialize_weights(generator)
    s = torch.randn(N, z_dim, 1, 1)
    # Ensure the generator output shape is as expected
    assert generator(s).shape == (N, channels, height, width)

    # Print success message if all assertions pass
    print('success')

# Run the test function
if __name__ == "__main__":
    test()
