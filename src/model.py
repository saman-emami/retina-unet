import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=Config.dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
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
    def __init__(
        self,
        in_channels=Config.in_channels,
        num_classes=Config.num_classes,
        layers=Config.layers,
    ):
        super().__init__()

        # Input layer
        self.input_conv = DoubleConv(in_channels, layers[0])

        self.encoder_path = [
            DownSample(*channels) for channels in zip(layers[:-1], layers[1:])
        ]

        reversed_layers = layers[::-1]

        self.decoder_path = [
            UpSample(*channels)
            for channels in zip(reversed_layers[:-1], reversed_layers[1:])
        ]

        # Output layer
        self.output_conv = nn.Conv2d(layers[0], num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.input_conv(x)

        # Encoder
        encodings = [x1]
        for i, down_sample in enumerate(self.encoder_path):
            encodings[i + 1] = down_sample(encodings[i])

        reversed_encodings = encodings[::-1]

        # Decoder with skip connections
        x = reversed_encodings[0]
        for i, up_sample in enumerate(self.decoder_path):
            x = up_sample(x, reversed_encodings[i + 1])

        # Output segmentation map
        output = self.output_conv(x)
        return output
