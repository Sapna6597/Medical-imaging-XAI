"""A compact CNN sized for MedMNIST, with a Grad-CAM-friendly final conv block."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """3-block convolutional network.

    Keeps an ~1/8 spatial resolution feature map (e.g., 8x8 for 64x64 input) so
    Grad-CAM produces a meaningful localization heatmap. ``target_layer`` exposes
    the last convolution for Grad-CAM hooks.
    """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        super().__init__()
        self.block1 = self._conv_block(n_channels, 32)
        self.block2 = self._conv_block(32, 64)
        self.block3 = self._conv_block(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)
        # Last conv layer used as the Grad-CAM target.
        self.target_layer: nn.Module = self.block3[0]

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(n_channels: int, n_classes: int) -> SmallCNN:
    """Construct the classifier."""
    return SmallCNN(n_channels=n_channels, n_classes=n_classes)
