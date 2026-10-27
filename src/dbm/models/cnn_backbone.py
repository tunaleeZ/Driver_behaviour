"""Small CNN backbone for per-frame feature extraction."""

from __future__ import annotations

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """Four convolutional blocks followed by global average pooling."""

    def __init__(self, channels: tuple[int, ...] | list[int] = (32, 64, 128, 256),
                 input_channels: int = 3):
        super().__init__()
        if not channels or any(int(c) <= 0 for c in channels):
            raise ValueError("channels must contain positive values")
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            layers.extend((nn.Conv2d(in_channels, int(out_channels), 3, padding=1),
                           nn.BatchNorm2d(int(out_channels)), nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)))
            in_channels = int(out_channels)
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = in_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map (batch, channels, height, width) to (batch, output_dim)."""
        return self.pool(self.features(images)).flatten(1)
