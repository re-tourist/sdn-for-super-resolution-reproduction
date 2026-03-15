#!/usr/bin/env python
"""Minimal reproducible forward-sanity validation for Stage 3 Issue 4."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Keep `python scripts/check_optical_forward_depths.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optics.diffractive_decoder import DiffractiveDecoder


OUTPUT_CROP_HW = (20, 20)
INPUT_PATTERN_HW = (24, 24)
LAYER_HW = (32, 32)
PROPAGATION_HW = (48, 48)


def build_distance_schedule(num_diffractive_layers: int) -> dict[str, object]:
    return {
        "input_to_first": 0.01,
        "inter_layer": [0.02] * max(0, num_diffractive_layers - 1),
        "last_to_sensor": 0.03,
    }


def build_decoder(num_diffractive_layers: int) -> DiffractiveDecoder:
    return DiffractiveDecoder(
        num_diffractive_layers=num_diffractive_layers,
        wavelength=532e-9,
        pixel_pitch=8e-6,
        grid_config={
            "input_pattern_hw": INPUT_PATTERN_HW,
            "layer_hw": LAYER_HW,
            "propagation_hw": PROPAGATION_HW,
        },
        distance_schedule=build_distance_schedule(num_diffractive_layers),
        readout_config={
            "output_crop_hw": OUTPUT_CROP_HW,
        },
        phase_mask_config={
            "init_mode": "zeros",
            "phase_mapping": "tanh",
            "phase_range": (-math.pi, math.pi),
        },
    )


def make_controlled_phase() -> torch.Tensor:
    y = torch.linspace(-0.75, 0.75, INPUT_PATTERN_HW[0], dtype=torch.float32)
    x = torch.linspace(-1.00, 1.00, INPUT_PATTERN_HW[1], dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 0.6 * xx + 0.4 * yy
    phase[INPUT_PATTERN_HW[0] // 2, INPUT_PATTERN_HW[1] // 2] += 0.5
    return phase.unsqueeze(0).unsqueeze(0)


def assert_real_nonnegative(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.is_complex():
        raise AssertionError(f"{name} must be real-valued, got dtype {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} must contain only finite values.")
    if (tensor < 0).any():
        raise AssertionError(f"{name} must be nonnegative.")


def assert_output_contract(
    output: dict[str, torch.Tensor | list[torch.Tensor]],
    *,
    expected_crop_hw: tuple[int, int],
) -> None:
    required_keys = {"U0", "U_out_full", "I_out_full", "I_out_roi"}
    missing_keys = required_keys.difference(output.keys())
    if missing_keys:
        raise AssertionError(f"Missing required output keys: {sorted(missing_keys)}.")

    U_out_full = output["U_out_full"]
    I_out_full = output["I_out_full"]
    I_out_roi = output["I_out_roi"]
    if not isinstance(U_out_full, torch.Tensor):
        raise AssertionError("U_out_full must be a tensor.")
    if not isinstance(I_out_full, torch.Tensor):
        raise AssertionError("I_out_full must be a tensor.")
    if not isinstance(I_out_roi, torch.Tensor):
        raise AssertionError("I_out_roi must be a tensor.")

    if not U_out_full.is_complex():
        raise AssertionError(f"U_out_full must be complex-valued, got dtype {U_out_full.dtype}.")

    assert_real_nonnegative(I_out_full, name="I_out_full")
    assert_real_nonnegative(I_out_roi, name="I_out_roi")

    if tuple(I_out_full.shape) != (1, 1, *PROPAGATION_HW):
        raise AssertionError(
            "I_out_full shape mismatch: "
            f"expected {(1, 1, *PROPAGATION_HW)}, got {tuple(I_out_full.shape)}."
        )
    if tuple(I_out_roi.shape) != (1, 1, *expected_crop_hw):
        raise AssertionError(
            f"I_out_roi shape mismatch: expected {(1, 1, *expected_crop_hw)}, got {tuple(I_out_roi.shape)}."
        )


def main() -> None:
    torch.manual_seed(7)
    phase = make_controlled_phase()

    outputs: dict[int, dict[str, torch.Tensor | list[torch.Tensor]]] = {}
    output_key_signature: set[str] | None = None
    for num_diffractive_layers in (1, 3, 5):
        decoder = build_decoder(num_diffractive_layers)
        output = decoder(phase)
        assert_output_contract(output, expected_crop_hw=OUTPUT_CROP_HW)

        current_keys = set(output.keys())
        if output_key_signature is None:
            output_key_signature = current_keys
        elif current_keys != output_key_signature:
            raise AssertionError(
                "Output interface mismatch across depths: "
                f"expected {sorted(output_key_signature)}, got {sorted(current_keys)} for L={num_diffractive_layers}."
            )

        outputs[num_diffractive_layers] = output

    if output_key_signature is None:
        raise AssertionError("No decoder outputs were produced.")

    pairwise_full_deltas: dict[tuple[int, int], float] = {}
    pairwise_roi_deltas: dict[tuple[int, int], float] = {}
    for left_depth, right_depth in ((1, 3), (1, 5), (3, 5)):
        left_output = outputs[left_depth]
        right_output = outputs[right_depth]

        left_full = left_output["I_out_full"]
        right_full = right_output["I_out_full"]
        left_roi = left_output["I_out_roi"]
        right_roi = right_output["I_out_roi"]
        if not isinstance(left_full, torch.Tensor) or not isinstance(right_full, torch.Tensor):
            raise AssertionError("I_out_full tensors are missing for pairwise comparison.")
        if not isinstance(left_roi, torch.Tensor) or not isinstance(right_roi, torch.Tensor):
            raise AssertionError("I_out_roi tensors are missing for pairwise comparison.")

        full_delta = torch.mean(torch.abs(left_full - right_full)).item()
        roi_delta = torch.mean(torch.abs(left_roi - right_roi)).item()
        if full_delta <= 1e-6:
            raise AssertionError(
                f"I_out_full for L={left_depth} and L={right_depth} should not be trivially identical."
            )
        if roi_delta <= 1e-6:
            raise AssertionError(
                f"I_out_roi for L={left_depth} and L={right_depth} should not be trivially identical."
            )

        pairwise_full_deltas[(left_depth, right_depth)] = full_delta
        pairwise_roi_deltas[(left_depth, right_depth)] = roi_delta

    invalid_schedule_error: str | None = None
    try:
        DiffractiveDecoder(
            num_diffractive_layers=5,
            wavelength=532e-9,
            pixel_pitch=8e-6,
            grid_config={
                "input_pattern_hw": INPUT_PATTERN_HW,
                "layer_hw": LAYER_HW,
                "propagation_hw": PROPAGATION_HW,
            },
            distance_schedule={
                "input_to_first": 0.01,
                "inter_layer": [0.02, 0.02],
                "last_to_sensor": 0.03,
            },
            readout_config={
                "output_crop_hw": OUTPUT_CROP_HW,
            },
            phase_mask_config={
                "init_mode": "zeros",
                "phase_mapping": "tanh",
                "phase_range": (-math.pi, math.pi),
            },
        )
    except ValueError as exc:
        invalid_schedule_error = str(exc)

    if invalid_schedule_error is None:
        raise AssertionError(
            "Invalid distance_schedule length must raise a ValueError for explicit layer semantics."
        )

    print("Stage 3 Issue 4 forward sanity validation passed.")
    print("depths_checked=(1, 3, 5)")
    print(f"output_keys={sorted(output_key_signature)}")
    print("check_1: forward pass runs successfully for L=1/3/5")
    print("check_2: outputs include U_out_full, I_out_full, I_out_roi")
    print("check_3: U_out_full remains complex")
    print("check_4: I_out_full and I_out_roi remain real and nonnegative")
    print("check_5: ROI shape matches configured crop size")
    print("check_6: output interfaces are consistent across depths")
    print("check_7: outputs for different L are not trivially identical")
    print("check_8: invalid distance schedule is rejected explicitly")
    for depth in (1, 3, 5):
        depth_output = outputs[depth]
        I_out_full = depth_output["I_out_full"]
        I_out_roi = depth_output["I_out_roi"]
        if not isinstance(I_out_full, torch.Tensor) or not isinstance(I_out_roi, torch.Tensor):
            raise AssertionError("Depth output is missing intensity tensors.")
        print(
            f"L={depth} full_shape={tuple(I_out_full.shape)} "
            f"roi_shape={tuple(I_out_roi.shape)}"
        )
    for pair, delta in pairwise_full_deltas.items():
        print(f"full_delta_L{pair[0]}_L{pair[1]}={delta:.6e}")
    for pair, delta in pairwise_roi_deltas.items():
        print(f"roi_delta_L{pair[0]}_L{pair[1]}={delta:.6e}")
    print(f"invalid_schedule_error={invalid_schedule_error}")


if __name__ == "__main__":
    main()
