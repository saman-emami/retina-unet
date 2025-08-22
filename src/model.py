import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=Config.dropout,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: up-sampled features, x2: skip connection features
        x1 = self.up(x1)

        diff_y = x2.size(dim=2) - x1.size(dim=2)
        diff_x = x2.size(dim=3) - x1.size(dim=3)

        left_padding = diff_x // 2
        right_padding = diff_x - left_padding
        up_padding = diff_y // 2
        down_padding = diff_y - up_padding

        x1 = F.pad(x1, [left_padding, right_padding, up_padding, down_padding])

        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=Config.in_channels, num_classes=Config.num_classes):
        super().__init__()

        # Encoder path
        self.input_conv = DoubleConv(in_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        # Decoder path
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output segmentation map
        output = self.output_conv(x)
        return output
