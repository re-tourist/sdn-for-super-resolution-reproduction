"""Pure electronic bottleneck baseline aligned with Stage1/2 SR semantics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ElectronicBottleneckAutoencoder(nn.Module):
    """HR -> encoder -> low-res latent -> decoder -> reconstructed HR."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_channels: int = 1,
        sr_factor: int = 4,
    ) -> None:
        super().__init__()
        if sr_factor != 4:
            raise ValueError(
                f"This baseline currently supports sr_factor=4 only, got {sr_factor}."
            )
        self.sr_factor = sr_factor
        self.latent_channels = latent_channels

        # Encoder: 96 -> 48 -> 24
        self.enc_block0 = ConvBlock(in_channels, base_channels)
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.enc_block1 = ConvBlock(base_channels * 2, base_channels * 2)
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.latent_proj = nn.Conv2d(base_channels * 2, latent_channels, kernel_size=1)

        # Decoder: 24 -> 48 -> 96
        self.dec_block0 = ConvBlock(latent_channels, base_channels * 2)
        self.dec_block1 = ConvBlock(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_block0(x)
        h = self.enc_down1(h)
        h = self.enc_block1(h)
        h = self.enc_down2(h)
        latent = self.latent_proj(h)
        return latent

    def decode(self, latent: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
        h = self.dec_block0(latent)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec_block1(h)
        h = F.interpolate(h, size=output_hw, mode="bilinear", align_corners=False)
        out = self.out_conv(h)
        # `atan` keeps the output bounded in [0, 1] but avoids the early saturation
        # that caused this sparse grayscale task to collapse to all-black reconstructions.
        return torch.atan(out) / torch.pi + 0.5

    def forward(
        self, x: torch.Tensor, return_latent: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decode(latent, output_hw=(x.shape[-2], x.shape[-1]))
        if return_latent:
            return recon, latent
        return recon
