#!/usr/bin/env python
"""Minimal reproducible validation for Stage 3 Issue 3 readout/crop."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Keep `python scripts/check_optical_readout.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optics.diffractive_decoder import DiffractiveDecoder


def build_decoder(output_crop_hw: tuple[int, int]) -> DiffractiveDecoder:
    return DiffractiveDecoder(
        num_diffractive_layers=3,
        wavelength=532e-9,
        pixel_pitch=8e-6,
        grid_config={
            "input_pattern_hw": (24, 24),
            "layer_hw": (32, 32),
            "propagation_hw": (48, 48),
        },
        distance_schedule={
            "input_to_first": 0.01,
            "inter_layer": [0.02, 0.02],
            "last_to_sensor": 0.03,
        },
        readout_config={
            "output_crop_hw": output_crop_hw,
        },
        phase_mask_config={
            "init_mode": "zeros",
            "phase_mapping": "tanh",
            "phase_range": (-math.pi, math.pi),
        },
    )


def assert_real_nonnegative(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.is_complex():
        raise AssertionError(f"{name} must be real-valued, got dtype {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} must contain only finite values.")
    if (tensor < 0).any():
        raise AssertionError(f"{name} must be nonnegative.")


def main() -> None:
    torch.manual_seed(7)

    phi_base = torch.zeros(1, 1, 24, 24, dtype=torch.float32)
    phi_perturbed = phi_base.clone()
    phi_perturbed[..., 12, 12] = 0.5

    decoder_crop_20 = build_decoder((20, 20))
    output_base = decoder_crop_20(phi_base)
    output_perturbed = decoder_crop_20(phi_perturbed)

    if not output_base["U_out_full"].is_complex():
        raise AssertionError("U_out_full must be complex-valued.")

    assert_real_nonnegative(output_base["I_out_full"], name="I_out_full")
    assert_real_nonnegative(output_base["I_out_roi"], name="I_out_roi")

    if tuple(output_base["I_out_full"].shape) != (1, 1, 48, 48):
        raise AssertionError(
            f"I_out_full shape mismatch: expected (1, 1, 48, 48), got {tuple(output_base['I_out_full'].shape)}."
        )
    if tuple(output_base["I_out_roi"].shape) != (1, 1, 20, 20):
        raise AssertionError(
            f"I_out_roi shape mismatch for crop (20, 20): got {tuple(output_base['I_out_roi'].shape)}."
        )

    decoder_crop_12 = build_decoder((12, 12))
    output_crop_12 = decoder_crop_12(phi_base)
    if tuple(output_crop_12["I_out_roi"].shape) != (1, 1, 12, 12):
        raise AssertionError(
            f"I_out_roi shape mismatch for crop (12, 12): got {tuple(output_crop_12['I_out_roi'].shape)}."
        )

    oversized_crop_error: str | None = None
    try:
        build_decoder((64, 64))(phi_base)
    except ValueError as exc:
        oversized_crop_error = str(exc)
    if oversized_crop_error is None:
        raise AssertionError("Oversized crop must raise an explicit ValueError.")

    full_delta = torch.mean(
        torch.abs(output_base["I_out_full"] - output_perturbed["I_out_full"])
    ).item()
    roi_delta = torch.mean(
        torch.abs(output_base["I_out_roi"] - output_perturbed["I_out_roi"])
    ).item()
    if full_delta <= 1e-6:
        raise AssertionError(f"I_out_full should change when input phase changes, got {full_delta}.")
    if roi_delta <= 1e-6:
        raise AssertionError(f"I_out_roi should change when input phase changes, got {roi_delta}.")

    print("Stage 3 Issue 3 readout/crop validation passed.")
    print("check_1: U_out_full is complex")
    print("check_2: I_out_full and I_out_roi are real and nonnegative")
    print("check_3: output_crop_hw changes ROI shape correctly")
    print("check_4: oversized crop raises an explicit error")
    print("check_5: changing input phase changes I_out_full and I_out_roi")
    print(f"full_shape={tuple(output_base['I_out_full'].shape)}")
    print(f"roi_shape_20={tuple(output_base['I_out_roi'].shape)}")
    print(f"roi_shape_12={tuple(output_crop_12['I_out_roi'].shape)}")
    print(f"oversized_crop_error={oversized_crop_error}")
    print(f"full_delta={full_delta:.6e}")
    print(f"roi_delta={roi_delta:.6e}")


if __name__ == "__main__":
    main()
