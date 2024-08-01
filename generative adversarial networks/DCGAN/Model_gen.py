import torch
import torch.nn as nn

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, channel_image, feature_channel):
        """
        Initialize the Discriminator model.

        Parameters:
        channel_image (int): Number of channels in the input image.
        feature_channel (int): Number of feature channels after the first convolution.
        """
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            # Initial convolutional layer
            nn.Conv2d(channel_image, feature_channel, 4, 2, 1, bias=False),  # 32x32 -> 16x16 or 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            # Adding additional convolutional blocks
            self.block_(feature_channel, feature_channel * 2, 4, 2, 1),
            self.block_(feature_channel * 2, feature_channel * 4, 4, 2, 1),
            self.block_(feature_channel * 4, feature_channel * 8, 4, 2, 1),
            # Final convolutional layer to produce a single output
            nn.Conv2d(feature_channel * 8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def block_(self, channel_in, channel_out, kernel_size, stride, padding):
        """
        Helper method to create a block with Conv2d, BatchNorm2d, and LeakyReLU layers.

        Parameters:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.

        Returns:
        nn.Sequential: Sequential block containing Conv2d, BatchNorm2d, and LeakyReLU layers.
        """
        return nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Forward pass through the Discriminator.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the discriminator.
        """
        print(f"Input: {x.shape}")
        for layer in self.discriminator:
            x = layer(x)
            print(f"After layer {layer}: {x.shape}")
        return x

# Testing the modified Discriminator
def test_discriminator():
    """
    Test the Discriminator model to ensure it produces the expected output shapes.
    """
    N, channels, height, width = 8, 3, 32, 32  # For CIFAR-10
    t = torch.randn(N, channels, height, width)
    discriminator = Discriminator(channels, 8)
    print(discriminator)
    output = discriminator(t)
    print(f"Output shape: {output.shape}")

    N, channels, height, width = 8, 3, 64, 64  # For larger input size
    t = torch.randn(N, channels, height, width)
    output = discriminator(t)
    print(f"Output shape: {output.shape}")

test_discriminator()

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, z_dimension, channel_image, feature_channel):
        """
        Initialize the Generator model.

        Parameters:
        z_dimension (int): Dimension of the latent vector.
        channel_image (int): Number of channels in the output image.
        feature_channel (int): Number of feature channels after the first transpose convolution.
        """
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

    def block_(self, channel_in, channel_out, kernel_size, stride, padding):
        """
        Helper method to create a block with ConvTranspose2d, BatchNorm2d, and ReLU layers.

        Parameters:
        channel_in (int): Number of input channels.
        channel_out (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding for the convolution.

        Returns:
        nn.Sequential: Sequential block containing ConvTranspose2d, BatchNorm2d, and ReLU layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(True),  # ReLU activation
        )

    def forward(self, x):
        """
        Forward pass through the Generator.

        Parameters:
        x (torch.Tensor): Input tensor (latent vector).

        Returns:
        torch.Tensor: Output tensor (generated image).
        """
        return self.generator(x)


def initialize_weights(model):
    """
    Initialize the weights of the model.

    Parameters:
    model (nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)


def test():
    """
    Test the Discriminator and Generator models to ensure they produce the expected output shapes.
    """
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


if __name__ == "__main__":
    # Run the test function if the script is executed directly
    test()
