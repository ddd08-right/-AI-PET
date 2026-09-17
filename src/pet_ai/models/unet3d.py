"""Small native PyTorch 3D U-Net baseline.

This module intentionally implements a compact two-level 3D U-Net for
engineering education and CPU smoke tests. It is not nnU-Net, is not novel, and
is not a clinical model.

Tensor convention:
    input:  [B, 2, D, H, W]
    output: [B, 1, D, H, W] raw logits
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    """Two Conv3d layers with GroupNorm and ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet3D(nn.Module):
    """A deliberately small two-level 3D U-Net returning raw segmentation logits."""

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 4,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.encoder1 = ConvBlock3D(in_channels, c1)
        self.down1 = nn.MaxPool3d(kernel_size=2)
        self.encoder2 = ConvBlock3D(c1, c2)
        self.down2 = nn.MaxPool3d(kernel_size=2)
        self.bottleneck = ConvBlock3D(c2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock3D(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock3D(c1 + c1, c1)
        self.output_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward propagation.

        For the default input [B, 2, 32, 32, 32]:
            enc1:      [B, 4, 32, 32, 32]
            enc2:      [B, 8, 16, 16, 16]
            bottleneck:[B, 16, 8, 8, 8]
            logits:    [B, 1, 32, 32, 32]
        """

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.down1(enc1))
        bottleneck = self.bottleneck(self.down2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        return self.output_conv(dec1)
