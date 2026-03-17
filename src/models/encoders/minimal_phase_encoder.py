"""Minimal phase encoder for the Stage 4 closed-loop protocol.

This module implements a deliberately small, debuggable encoder for the frozen
Stage 4 closed loop:

`HR input -> minimal encoder -> phi_lr -> optical decoder -> I_out_roi -> loss`

It is intentionally not the final paper-aligned encoder. Its only job is to
accept an HR image tensor and produce a phase-domain tensor `phi_lr` that can
be consumed by `DiffractiveDecoder.forward_from_phase(...)`.

Stage is represented by protocol/config and experiment scope, not by a
stage-named source directory. This module lives under `src/models/encoders/`
for that reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_hw(hw: int | Sequence[int], *, name: str) -> tuple[int, int]:
    """Normalize an integer or a length-2 sequence into `(height, width)`."""
    if isinstance(hw, int):
        if hw < 1:
            raise ValueError(f"{name} must be >= 1, got {hw}.")
        return (hw, hw)

    if not isinstance(hw, Sequence) or isinstance(hw, (str, bytes)) or len(hw) != 2:
        raise TypeError(f"{name} must be an int or a length-2 sequence, got {hw!r}.")

    height = int(hw[0])
    width = int(hw[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} values must be >= 1, got {(height, width)}.")
    return (height, width)


class ConvBlock(nn.Module):
    """Small reusable conv block aligned with the existing repo style."""

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


class MinimalPhaseEncoder(nn.Module):
    """Minimal Stage 4 encoder that outputs phase-domain `phi_lr`.

    The frozen Stage 4 mainline expects a single-channel HR image tensor as
    input and a single-channel phase tensor `phi_lr` as output. This module
    keeps the raw-to-phase mapping explicit and intentionally avoids paper-final
    complexity.

    Stage should be expressed through protocol/config, not through source-code
    directory structure. This is why the encoder lives under
    `src/models/encoders/` rather than a stage-named module path.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        target_hw: int | Sequence[int],
        base_channels: int = 16,
        hidden_channels: int | None = None,
        phase_range: tuple[float, float] = (-math.pi, math.pi),
        interpolation_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}.")

        hidden = int(hidden_channels) if hidden_channels is not None else base_channels * 2
        if hidden < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden}.")

        phase_min = float(phase_range[0])
        phase_max = float(phase_range[1])
        if not phase_min < phase_max:
            raise ValueError(
                "phase_range must be strictly increasing, "
                f"got {(phase_min, phase_max)}."
            )

        self.in_channels = int(in_channels)
        self.target_hw = _normalize_hw(target_hw, name="target_hw")
        self.base_channels = int(base_channels)
        self.hidden_channels = hidden
        self.phase_range = (phase_min, phase_max)
        self.interpolation_mode = interpolation_mode
        self.align_corners = bool(align_corners)

        self.stem = ConvBlock(self.in_channels, self.base_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.base_channels, self.hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.body = ConvBlock(self.hidden_channels, self.hidden_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.base_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.raw_phase_head = nn.Conv2d(self.base_channels, 1, kernel_size=1)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x)!r}.")
        if x.ndim != 4:
            raise ValueError(
                "MinimalPhaseEncoder expects input shape [B,C,H,W], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                "Input channel count does not match encoder configuration, "
                f"got {tuple(x.shape)} with in_channels={self.in_channels}."
            )

    def _resize_to_target(self, features: torch.Tensor) -> torch.Tensor:
        resize_kwargs: dict[str, object] = {
            "size": self.target_hw,
            "mode": self.interpolation_mode,
        }
        if self.interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            resize_kwargs["align_corners"] = self.align_corners
        return F.interpolate(features, **resize_kwargs)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into low-resolution feature maps before phase projection."""
        self._validate_input(x)
        features = self.stem(x)
        features = self.downsample(features)
        features = self.body(features)
        features = self._resize_to_target(features)
        return self.proj(features)

    def encode_raw_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the pre-mapped raw phase tensor for debugging."""
        features = self.encode_features(x)
        return self.raw_phase_head(features)

    def map_raw_phase(self, raw_phase: torch.Tensor) -> torch.Tensor:
        """Map raw phase logits into the configured phase-domain range."""
        if raw_phase.ndim != 4 or raw_phase.shape[1] != 1:
            raise ValueError(
                "raw_phase must have shape [B,1,H,W], "
                f"got {tuple(raw_phase.shape)}."
            )

        phase_min, phase_max = self.phase_range
        phase_center = 0.5 * (phase_min + phase_max)
        phase_half_span = 0.5 * (phase_max - phase_min)
        return phase_center + phase_half_span * torch.tanh(raw_phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode HR input into phase-domain `phi_lr`."""
        raw_phase = self.encode_raw_phase(x)
        phi_lr = self.map_raw_phase(raw_phase)
        return phi_lr

    def extra_repr(self) -> str:
        phase_min, phase_max = self.phase_range
        return (
            f"in_channels={self.in_channels}, "
            f"target_hw={self.target_hw}, "
            f"base_channels={self.base_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"phase_range=({phase_min:.4f}, {phase_max:.4f}), "
            f"interpolation_mode='{self.interpolation_mode}'"
        )


@torch.no_grad()
def _run_self_check() -> None:
    """Run a minimal local sanity check for shape and phase range."""
    encoder = MinimalPhaseEncoder(
        in_channels=1,
        target_hw=(24, 24),
        base_channels=8,
        hidden_channels=16,
        phase_range=(-math.pi, math.pi),
    )
    x_hr = torch.rand(2, 1, 96, 96, dtype=torch.float32)
    phi_lr = encoder(x_hr)

    expected_shape = (2, 1, 24, 24)
    if tuple(phi_lr.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected phi_lr shape: got {tuple(phi_lr.shape)}, expected {expected_shape}."
        )

    phase_min, phase_max = encoder.phase_range
    if float(phi_lr.min()) < phase_min - 1e-6 or float(phi_lr.max()) > phase_max + 1e-6:
        raise AssertionError(
            "phi_lr is outside the configured phase range: "
            f"range={encoder.phase_range}, "
            f"observed=({float(phi_lr.min())}, {float(phi_lr.max())})."
        )

    print("MinimalPhaseEncoder self-check passed.")
    print(f"phi_lr.shape={tuple(phi_lr.shape)}")
    print(f"phi_lr.range=({float(phi_lr.min()):.6f}, {float(phi_lr.max()):.6f})")


if __name__ == "__main__":
    _run_self_check()


__all__ = ["MinimalPhaseEncoder"]
