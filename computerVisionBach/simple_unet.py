from keras import backend as K
import torch
import torch.nn as nn

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_classes=4):
        super(UNet, self).__init__()

        self.c1 = ConvBlock(in_channels, 16, dropout=0.2)
        self.p1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(16, 32, dropout=0.2)
        self.p2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(32, 64, dropout=0.2)
        self.p3 = nn.MaxPool2d(2)

        self.c4 = ConvBlock(64, 128, dropout=0.2)
        self.p4 = nn.MaxPool2d(2)

        self.c5 = ConvBlock(128, 256, dropout=0.3)

        self.u6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c6 = ConvBlock(256, 128, dropout=0.2)

        self.u7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c7 = ConvBlock(128, 64, dropout=0.2)

        self.u8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.c8 = ConvBlock(64, 32, dropout=0.2)

        self.u9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.c9 = ConvBlock(32, 16, dropout=0.2)

        self.out = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        return self.out(c9)
