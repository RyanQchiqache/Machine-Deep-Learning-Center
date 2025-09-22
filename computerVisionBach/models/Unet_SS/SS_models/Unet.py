import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.dropout(self.conv(x))

class DoubleConv(nn.Module):
    """Two 3x3 conv layers with ReLU, no BatchNorm or Dropout, padding=0 (valid conv)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),  # padding=0
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def crop_and_concat(self, enc_feat, dec_feat):
        """Crop encoder features to match decoder size, then concatenate."""
        _, _, H, W = dec_feat.shape
        enc_feat_cropped = torchvision.transforms.CenterCrop([H, W])(enc_feat)
        return torch.cat([enc_feat_cropped, dec_feat], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.upconv4(b)
        d4 = self.crop_and_concat(e4, d4)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = self.crop_and_concat(e3, d3)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = self.crop_and_concat(e2, d2)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = self.crop_and_concat(e1, d1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)

        """if return_features:
            return out, {'enc1': e1, 'enc2': e2, 'enc3': e3, 'enc4': e4,
                         'bottleneck': b, 'dec4': d4, 'dec3': d3, 'dec2': d2, 'dec1': d1}"""

        return out