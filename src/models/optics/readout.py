"""Output-plane readout helpers for the Stage 3 optical decoder contract.

This module keeps readout concerns separate from propagation:

- `intensity_readout` converts a complex field into real intensity.
- `ReadoutConfig` owns the explicit `output_crop_hw` contract.
- `center_crop_2d` performs deterministic center cropping on the full grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from src.models.optics.phase_utils import ensure_complex_field, normalize_hw


@dataclass(frozen=True)
class ReadoutConfig:
    """Config for Stage 3 output-plane readout."""

    output_crop_hw: tuple[int, int]

    @classmethod
    def from_config(cls, config: "ReadoutConfig | Mapping[str, object]") -> "ReadoutConfig":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("readout_config must be a ReadoutConfig or a mapping.")
        return cls(
            output_crop_hw=normalize_hw(config["output_crop_hw"], name="output_crop_hw"),
        )

    def center_crop_slices(self, full_hw: tuple[int, int]) -> tuple[slice, slice]:
        """Build deterministic center-crop slices for a full propagation grid."""
        full_h, full_w = normalize_hw(full_hw, name="full_hw")
        crop_h, crop_w = self.output_crop_hw

        if crop_h > full_h or crop_w > full_w:
            raise ValueError(
                "output_crop_hw must fit inside the full propagation grid, "
                f"got crop={self.output_crop_hw}, full={(full_h, full_w)}."
            )

        top = (full_h - crop_h) // 2
        left = (full_w - crop_w) // 2
        return (slice(top, top + crop_h), slice(left, left + crop_w))


def intensity_readout(field: torch.Tensor) -> torch.Tensor:
    """Read output-plane intensity from a complex field.

    The returned tensor keeps the same `[B, 1, H, W]` layout, but is real and
    nonnegative by construction.
    """
    ensure_complex_field(field, name="field")
    return field.real.square() + field.imag.square()


def center_crop_2d(
    tensor: torch.Tensor,
    crop_hw: tuple[int, int] | ReadoutConfig,
) -> torch.Tensor:
    """Crop the center ROI from a tensor using its last two dimensions."""
    if tensor.ndim < 2:
        raise ValueError(f"tensor must have at least 2 dimensions, got shape {tuple(tensor.shape)}.")

    if isinstance(crop_hw, ReadoutConfig):
        crop_config = crop_hw
    else:
        crop_config = ReadoutConfig(output_crop_hw=normalize_hw(crop_hw, name="output_crop_hw"))

    crop_h, crop_w = crop_config.output_crop_hw
    full_h, full_w = tensor.shape[-2:]
    if crop_h > full_h or crop_w > full_w:
        raise ValueError(
            "output_crop_hw must fit inside the tensor spatial shape, "
            f"got crop={(crop_h, crop_w)}, full={(full_h, full_w)}."
        )

    crop_h_slice, crop_w_slice = crop_config.center_crop_slices((full_h, full_w))
    return tensor[..., crop_h_slice, crop_w_slice]


__all__ = [
    "ReadoutConfig",
    "center_crop_2d",
    "intensity_readout",
]
