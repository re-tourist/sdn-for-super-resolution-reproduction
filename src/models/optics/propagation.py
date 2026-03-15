"""Optical propagation primitives for the Stage 3 decoder skeleton.

This module keeps only the optics core that is worth reusing from
`external/sdn_upstream`:

- centered FFT / IFFT propagation flow derived from upstream `FFT` / `IFFT`
- Rayleigh-Sommerfeld transfer kernel derived from upstream `getH2`
- phase-only mask modulation derived from upstream `sublayer.forward`

Task heads, datasets, electrical decoders, and any layer-index special cases
are intentionally removed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from src.models.optics.phase_utils import (
    ensure_complex_field,
    normalize_hw,
    normalize_spacing,
    phase_to_field,
)


def fft2_centered(field: torch.Tensor) -> torch.Tensor:
    """Apply the same centered FFT layout used by the upstream optics code."""
    shifted = torch.fft.fftshift(field, dim=(-2, -1))
    transformed = torch.fft.fft2(shifted, dim=(-2, -1))
    return torch.fft.fftshift(transformed, dim=(-2, -1))


def ifft2_centered(spectrum: torch.Tensor) -> torch.Tensor:
    """Apply the inverse centered FFT layout used by the upstream optics code."""
    shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))
    transformed = torch.fft.ifft2(shifted, dim=(-2, -1))
    return torch.fft.fftshift(transformed, dim=(-2, -1))


def build_rayleigh_sommerfeld_transfer_kernel(
    propagation_hw: int | Sequence[int],
    *,
    pixel_pitch: float | Sequence[float],
    wavelength: float,
    distance: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the transfer kernel used for a single free-space propagation.

    The mathematical form follows the upstream `getH2` Rayleigh-Sommerfeld
    primitive, but is rewritten with pure torch instead of scipy/numpy glue.
    """
    height, width = normalize_hw(propagation_hw, name="propagation_hw")
    pitch_y, pitch_x = normalize_spacing(pixel_pitch, name="pixel_pitch")

    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be positive, got {wavelength}.")
    if distance <= 0.0:
        raise ValueError(f"distance must be positive, got {distance}.")
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Kernel dtype must be float32 or float64, got {dtype}.")

    y = torch.linspace(
        -(height / 2.0 - 0.5) * pitch_y,
        (height / 2.0 - 0.5) * pitch_y,
        steps=height,
        device=device,
        dtype=dtype,
    )
    x = torch.linspace(
        -(width / 2.0 - 0.5) * pitch_x,
        (width / 2.0 - 0.5) * pitch_x,
        steps=width,
        device=device,
        dtype=dtype,
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    radial_distance = torch.sqrt(distance * distance + xx.square() + yy.square())
    wavenumber = (2.0 * math.pi) / wavelength

    carrier = torch.polar(torch.ones_like(radial_distance), wavenumber * radial_distance)
    rs_real = 1.0 / (2.0 * math.pi * radial_distance)
    rs_imag = -torch.full_like(radial_distance, 1.0 / wavelength)
    rs_term = torch.complex(rs_real, rs_imag)
    geometry_term = distance / (distance * distance + xx.square() + yy.square())

    spatial_kernel = carrier * geometry_term * rs_term
    transfer_kernel = fft2_centered(spatial_kernel) * (pitch_y * pitch_x)
    return transfer_kernel


def propagation_inverse_scale(
    propagation_hw: int | Sequence[int],
    *,
    pixel_pitch: float | Sequence[float],
) -> float:
    """Match the discrete scaling used by the upstream propagation routine."""
    height, width = normalize_hw(propagation_hw, name="propagation_hw")
    pitch_y, pitch_x = normalize_spacing(pixel_pitch, name="pixel_pitch")

    physical_size_y = height * pitch_y
    physical_size_x = width * pitch_x
    return ((height + 1) * (width + 1)) / (physical_size_y * physical_size_x)


def propagate_with_transfer_kernel(
    field: torch.Tensor,
    *,
    transfer_kernel: torch.Tensor,
    pixel_pitch: float | Sequence[float],
) -> torch.Tensor:
    """Propagate a complex field with a precomputed transfer kernel."""
    ensure_complex_field(field, name="field")

    if transfer_kernel.ndim != 2:
        raise ValueError(
            f"transfer_kernel must be 2D [H, W], got shape {tuple(transfer_kernel.shape)}."
        )
    if tuple(transfer_kernel.shape) != tuple(field.shape[-2:]):
        raise ValueError(
            "transfer_kernel spatial shape must match the field grid, "
            f"got kernel={tuple(transfer_kernel.shape)} vs field={tuple(field.shape[-2:])}."
        )

    pitch_y, pitch_x = normalize_spacing(pixel_pitch, name="pixel_pitch")
    scaled_spectrum = fft2_centered(field) * (pitch_y * pitch_x)
    propagated_spectrum = scaled_spectrum * transfer_kernel.view(1, 1, *transfer_kernel.shape)
    propagated = ifft2_centered(propagated_spectrum)
    return propagated * propagation_inverse_scale(
        transfer_kernel.shape,
        pixel_pitch=(pitch_y, pitch_x),
    )


def apply_phase_modulation(field: torch.Tensor, phase_mask: torch.Tensor) -> torch.Tensor:
    """Apply a phase-only mask to a complex field."""
    ensure_complex_field(field, name="field")

    if phase_mask.ndim == 2:
        phase_mask = phase_mask.unsqueeze(0).unsqueeze(0)
    elif phase_mask.ndim != 4:
        raise ValueError(
            f"phase_mask must be 2D [H, W] or 4D [B, 1, H, W], got {tuple(phase_mask.shape)}."
        )
    if phase_mask.shape[-2:] != field.shape[-2:]:
        raise ValueError(
            "phase_mask spatial shape must match the field grid, "
            f"got mask={tuple(phase_mask.shape[-2:])} vs field={tuple(field.shape[-2:])}."
        )

    phase_mask = phase_mask.to(device=field.device, dtype=field.real.dtype)
    modulation_field = phase_to_field(
        phase_mask,
        phase_mapping="identity",
        complex_dtype=field.dtype,
    )
    return field * modulation_field


class PropagationOperator(nn.Module):
    """Single config-driven free-space propagation operator."""

    def __init__(
        self,
        propagation_hw: int | Sequence[int],
        *,
        pixel_pitch: float | Sequence[float],
        wavelength: float,
        distance: float,
        kernel_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.propagation_hw = normalize_hw(propagation_hw, name="propagation_hw")
        self.pixel_pitch = normalize_spacing(pixel_pitch, name="pixel_pitch")
        self.wavelength = float(wavelength)
        self.distance = float(distance)

        transfer_kernel = build_rayleigh_sommerfeld_transfer_kernel(
            self.propagation_hw,
            pixel_pitch=self.pixel_pitch,
            wavelength=self.wavelength,
            distance=self.distance,
            dtype=kernel_dtype,
        )
        self.register_buffer("transfer_kernel", transfer_kernel, persistent=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        ensure_complex_field(field, name="field", expected_hw=self.propagation_hw)
        kernel = self.transfer_kernel.to(device=field.device, dtype=field.dtype)
        return propagate_with_transfer_kernel(
            field,
            transfer_kernel=kernel,
            pixel_pitch=self.pixel_pitch,
        )


__all__ = [
    "PropagationOperator",
    "apply_phase_modulation",
    "build_rayleigh_sommerfeld_transfer_kernel",
    "fft2_centered",
    "ifft2_centered",
    "propagate_with_transfer_kernel",
]
