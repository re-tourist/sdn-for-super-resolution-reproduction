#!/usr/bin/env python
"""Stage 5 optics config sanity for paper-aligned L=1/3/5 settings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optics.diffractive_decoder import DiffractiveDecoder
from src.models.optics.readout import ReadoutConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage5/stage5_optics_paper_aligned.yaml",
        help="Path to the Stage 5 optics YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional override for the summary output directory.",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the Stage 5 optics config.") from exc

    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping at YAML root, got {type(payload)!r}.")
    return payload


def assert_real_nonnegative(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.is_complex():
        raise AssertionError(f"{name} must be real-valued, got dtype {tensor.dtype}.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} must contain only finite values.")
    if (tensor < 0).any():
        raise AssertionError(f"{name} must be nonnegative.")


def make_controlled_phase(input_pattern_hw: tuple[int, int]) -> torch.Tensor:
    height, width = input_pattern_hw
    y = torch.linspace(-0.8, 0.8, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 0.55 * xx + 0.35 * yy
    phase[height // 2, width // 2] += 0.75
    phase += 0.1 * torch.sin(xx * math.pi) * torch.cos(yy * math.pi)
    return phase.unsqueeze(0).unsqueeze(0)


def build_decoder(
    stage5_config: dict[str, Any],
    *,
    depth_key: str,
) -> DiffractiveDecoder:
    units = stage5_config["units"]
    grid = stage5_config["grid"]
    readout = stage5_config["readout"]
    depth_payload = stage5_config["depths"][depth_key]
    phase_mask_config = stage5_config.get("phase_mask_config")

    decoder_kwargs: dict[str, Any] = {
        "num_diffractive_layers": int(depth_payload["num_diffractive_layers"]),
        "wavelength": float(units["wavelength"]),
        "pixel_pitch": float(units["pixel_pitch_lambda"]),
        "grid_config": {
            "input_pattern_hw": tuple(int(v) for v in grid["input_pattern_hw"]),
            "layer_hw": tuple(int(v) for v in grid["layer_hw"]),
            "propagation_hw": tuple(int(v) for v in grid["propagation_hw"]),
        },
        "distance_schedule": depth_payload["distance_schedule"],
        "readout_config": {
            "output_crop_hw": tuple(int(v) for v in readout["output_crop_hw"]),
        },
    }
    if phase_mask_config is not None:
        decoder_kwargs["phase_mask_config"] = phase_mask_config

    return DiffractiveDecoder(**decoder_kwargs)


def summarize_depth(
    stage5_config: dict[str, Any],
    *,
    depth_key: str,
) -> dict[str, Any]:
    decoder = build_decoder(stage5_config, depth_key=depth_key)
    grid = decoder.grid_config
    readout = decoder.readout_config

    phase = make_controlled_phase(grid.input_pattern_hw)
    output = decoder(phase)

    U0 = output["U0"]
    U_out_full = output["U_out_full"]
    I_out_full = output["I_out_full"]
    I_out_roi = output["I_out_roi"]
    if not isinstance(U0, torch.Tensor) or not U0.is_complex():
        raise AssertionError("U0 must be a complex tensor.")
    if not isinstance(U_out_full, torch.Tensor) or not U_out_full.is_complex():
        raise AssertionError("U_out_full must be a complex tensor.")
    if not isinstance(I_out_full, torch.Tensor):
        raise AssertionError("I_out_full must be a tensor.")
    if not isinstance(I_out_roi, torch.Tensor):
        raise AssertionError("I_out_roi must be a tensor.")

    assert_real_nonnegative(I_out_full, name="I_out_full")
    assert_real_nonnegative(I_out_roi, name="I_out_roi")

    full_hw = tuple(int(v) for v in grid.propagation_hw)
    crop_hw = tuple(int(v) for v in readout.output_crop_hw)
    crop_slices = ReadoutConfig(output_crop_hw=crop_hw).center_crop_slices(full_hw)
    top = int(crop_slices[0].start)
    left = int(crop_slices[1].start)

    if tuple(U_out_full.shape) != (1, 1, *full_hw):
        raise AssertionError(
            f"U_out_full shape mismatch for L={depth_key}: {tuple(U_out_full.shape)} vs {(1, 1, *full_hw)}."
        )
    if tuple(I_out_full.shape) != (1, 1, *full_hw):
        raise AssertionError(
            f"I_out_full shape mismatch for L={depth_key}: {tuple(I_out_full.shape)} vs {(1, 1, *full_hw)}."
        )
    if tuple(I_out_roi.shape) != (1, 1, *crop_hw):
        raise AssertionError(
            f"I_out_roi shape mismatch for L={depth_key}: {tuple(I_out_roi.shape)} vs {(1, 1, *crop_hw)}."
        )

    depth_payload = stage5_config["depths"][depth_key]
    return {
        "num_diffractive_layers": int(depth_payload["num_diffractive_layers"]),
        "grid": {
            "input_pattern_hw": list(grid.input_pattern_hw),
            "layer_hw": list(grid.layer_hw),
            "propagation_hw": list(grid.propagation_hw),
        },
        "readout": {
            "output_crop_hw": list(crop_hw),
            "crop_policy": stage5_config["readout"]["crop_policy"],
            "center_offset_hw": [top, left],
        },
        "distance_mapping": depth_payload["distance_mapping"],
        "distance_schedule": {
            "input_to_first": float(decoder.distance_schedule.input_to_first),
            "inter_layer": [float(v) for v in decoder.distance_schedule.inter_layer],
            "last_to_sensor": float(decoder.distance_schedule.last_to_sensor),
        },
        "output_shapes": {
            "U0": list(U0.shape),
            "U_out_full": list(U_out_full.shape),
            "I_out_full": list(I_out_full.shape),
            "I_out_roi": list(I_out_roi.shape),
        },
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    stage5_config = config["optics"]["stage5_paper_aligned"]

    torch.manual_seed(int(stage5_config["sanity"]["seed"]))
    output_root = Path(args.output_root or stage5_config["sanity"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    depth_keys = [str(depth) for depth in stage5_config["sanity"]["depths"]]
    depth_summaries = {depth_key: summarize_depth(stage5_config, depth_key=depth_key) for depth_key in depth_keys}

    summary = {
        "config_path": str(Path(args.config).as_posix()),
        "paper_units": stage5_config["units"],
        "grid": stage5_config["grid"],
        "readout": stage5_config["readout"],
        "depths": depth_summaries,
    }
    summary_path = output_root / "stage5_optics_forward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 5 optics forward sanity passed.")
    print(f"config={args.config}")
    print(f"summary={summary_path.as_posix()}")
    for depth_key in depth_keys:
        depth_summary = depth_summaries[depth_key]
        shapes = depth_summary["output_shapes"]
        schedule = depth_summary["distance_schedule"]
        crop = depth_summary["readout"]["output_crop_hw"]
        offset = depth_summary["readout"]["center_offset_hw"]
        print(
            f"L={depth_key} layer_hw={tuple(depth_summary['grid']['layer_hw'])} "
            f"propagation_hw={tuple(depth_summary['grid']['propagation_hw'])} "
            f"crop_hw={tuple(crop)} crop_offset={tuple(offset)} "
            f"distances=(input_to_first={schedule['input_to_first']}, "
            f"inter_layer={schedule['inter_layer']}, last_to_sensor={schedule['last_to_sensor']}) "
            f"shapes=(U_out_full={tuple(shapes['U_out_full'])}, "
            f"I_out_full={tuple(shapes['I_out_full'])}, I_out_roi={tuple(shapes['I_out_roi'])})"
        )


if __name__ == "__main__":
    main()
