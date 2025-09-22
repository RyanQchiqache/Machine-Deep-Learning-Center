import torch
import torch.nn as nn
import torchvision.models as models

class UNetResNet34(nn.Module):
    def __init__(self, num_classes=15, in_channels=3):
        super(UNetResNet34, self).__init__()
        base_model = models.resnet34(weights=None)
        self.base_layers = list(base_model.children())

        # Replace the first conv layer to support custom in_channels
        self.base_layers[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder1 = nn.Sequential(*self.base_layers[:3])   # conv1 + bn1 + relu
        self.encoder2 = nn.Sequential(*self.base_layers[3:5])  # maxpool + layer1
        self.encoder3 = self.base_layers[5]
        self.encoder4 = self.base_layers[6]
        self.encoder5 = self.base_layers[7]

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder5 = self._decoder_block(256 + 256, 256)
        self.decoder4 = self._decoder_block(256 + 128, 128)
        self.decoder3 = self._decoder_block(128 + 64, 64)
        self.decoder2 = self._decoder_block(64 + 64, 64)
        self.decoder1 = self._decoder_block(64, 32)

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        center = self.center(e5)

        d5 = self.decoder5(torch.cat([center, e4], dim=1))
        d4 = self.decoder4(torch.cat([d5, e3], dim=1))
        d3 = self.decoder3(torch.cat([d4, e2], dim=1))
        d2 = self.decoder2(torch.cat([d3, e1], dim=1))
        d1 = self.decoder1(d2)

        return self.final(d1)