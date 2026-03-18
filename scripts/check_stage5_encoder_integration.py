#!/usr/bin/env python
"""Stage 5 encoder-to-decoder sanity for the paper-aligned phase path."""

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

from src.models.encoders import PaperPhaseEncoder
from src.models.optics.diffractive_decoder import DiffractiveDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage5/stage5_paper_encoder.yaml",
        help="Path to the Stage 5 paper encoder YAML config.",
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
        raise RuntimeError("PyYAML is required to load the Stage 5 configs.") from exc

    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping at YAML root, got {type(payload)!r}.")
    return payload


def make_controlled_hr(input_hw: tuple[int, int]) -> torch.Tensor:
    height, width = input_hw
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    image = 0.35 * torch.sin(math.pi * xx)
    image += 0.25 * torch.cos(2.0 * math.pi * yy)
    image += 0.20 * torch.exp(-4.5 * (xx.square() + yy.square()))
    image[height // 4 : height // 4 + 12, width // 3 : width // 3 + 18] += 0.35
    image = image - image.min()
    image = image / image.max().clamp_min(1e-6)
    return image.unsqueeze(0).unsqueeze(0)


def build_encoder(encoder_config: dict[str, Any]) -> PaperPhaseEncoder:
    return PaperPhaseEncoder(
        in_channels=int(encoder_config["in_channels"]),
        input_hw=tuple(int(v) for v in encoder_config["input_hw"]),
        target_hw=tuple(int(v) for v in encoder_config["target_hw"]),
        base_channels=int(encoder_config["base_channels"]),
        hidden_channels=int(encoder_config["hidden_channels"]),
        interpolation_mode=str(encoder_config["interpolation_mode"]),
        align_corners=bool(encoder_config["align_corners"]),
        phase_mapping=str(encoder_config["phase_mapping"]),
        phase_range=tuple(float(v) for v in encoder_config["phase_range"]),
    )


def build_decoder(optics_config: dict[str, Any], *, depth_key: str) -> DiffractiveDecoder:
    stage5_optics = optics_config["optics"]["stage5_paper_aligned"]
    units = stage5_optics["units"]
    grid = stage5_optics["grid"]
    readout = stage5_optics["readout"]
    depth_payload = stage5_optics["depths"][depth_key]
    phase_mask_config = stage5_optics.get("phase_mask_config")

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


def assert_phase_range(phi_lr: torch.Tensor, *, phase_range: tuple[float, float]) -> tuple[float, float]:
    if phi_lr.is_complex():
        raise AssertionError(f"phi_lr must be real-valued, got dtype {phi_lr.dtype}.")
    detached = phi_lr.detach()
    observed_min = float(detached.min())
    observed_max = float(detached.max())
    phase_min, phase_max = phase_range
    if observed_min < phase_min - 1e-6 or observed_max > phase_max + 1e-6:
        raise AssertionError(
            f"phi_lr observed range {(observed_min, observed_max)} exceeds configured range {phase_range}."
        )
    return observed_min, observed_max


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    encoder_config = config["encoder"]["stage5_paper_aligned"]
    optics_config = load_yaml_config(encoder_config["sanity"]["optics_config_path"])
    stage5_optics = optics_config["optics"]["stage5_paper_aligned"]

    torch.manual_seed(int(encoder_config["sanity"]["seed"]))

    encoder = build_encoder(encoder_config)
    optics_input_pattern_hw = tuple(int(v) for v in stage5_optics["grid"]["input_pattern_hw"])
    if encoder.target_hw != optics_input_pattern_hw:
        raise AssertionError(
            "Encoder target_hw must match the Stage 5 optics input_pattern_hw, "
            f"got {encoder.target_hw} vs {optics_input_pattern_hw}."
        )

    x_hr = make_controlled_hr(encoder.input_hw)
    raw_phase = encoder.encode_raw_phase(x_hr)
    phi_lr = encoder.map_raw_phase(raw_phase)
    observed_min, observed_max = assert_phase_range(phi_lr, phase_range=encoder.phase_range)

    depth_keys = [str(depth) for depth in encoder_config["sanity"]["depths"]]
    downstream_summaries: dict[str, Any] = {}
    for depth_key in depth_keys:
        decoder = build_decoder(optics_config, depth_key=depth_key)
        output = decoder.forward_from_phase(phi_lr)
        U0 = output["U0"]
        U_out_full = output["U_out_full"]
        I_out_full = output["I_out_full"]
        I_out_roi = output["I_out_roi"]
        if not isinstance(U0, torch.Tensor) or not U0.is_complex():
            raise AssertionError(f"L={depth_key} U0 must be a complex tensor.")
        if not isinstance(U_out_full, torch.Tensor) or not U_out_full.is_complex():
            raise AssertionError(f"L={depth_key} U_out_full must be a complex tensor.")
        if not isinstance(I_out_full, torch.Tensor) or I_out_full.is_complex():
            raise AssertionError(f"L={depth_key} I_out_full must be a real tensor.")
        if not isinstance(I_out_roi, torch.Tensor) or I_out_roi.is_complex():
            raise AssertionError(f"L={depth_key} I_out_roi must be a real tensor.")

        downstream_summaries[depth_key] = {
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

    output_root = Path(args.output_root or encoder_config["sanity"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "encoder_config_path": str(Path(args.config).as_posix()),
        "optics_config_path": str(Path(encoder_config["sanity"]["optics_config_path"]).as_posix()),
        "encoder": {
            "input_hw": list(encoder.input_hw),
            "target_hw": list(encoder.target_hw),
            "phase_mapping": encoder.phase_mapping,
            "phase_range": [float(encoder.phase_range[0]), float(encoder.phase_range[1])],
            "alignment_note": encoder_config["note"],
        },
        "observed": {
            "encoder_input_shape": list(x_hr.shape),
            "raw_phase_shape": list(raw_phase.shape),
            "phi_lr_shape": list(phi_lr.shape),
            "phi_lr_min": observed_min,
            "phi_lr_max": observed_max,
        },
        "decoder_depths": downstream_summaries,
    }
    summary_path = output_root / "stage5_encoder_integration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 5 encoder integration sanity passed.")
    print(f"config={args.config}")
    print(f"summary={summary_path.as_posix()}")
    print(
        f"encoder_input_shape={tuple(x_hr.shape)} raw_phase_shape={tuple(raw_phase.shape)} "
        f"phi_lr_shape={tuple(phi_lr.shape)}"
    )
    print(
        f"phase_mapping={encoder.phase_mapping} "
        f"phase_range={encoder.phase_range} "
        f"observed_phi_range=({observed_min:.6f}, {observed_max:.6f})"
    )
    for depth_key in depth_keys:
        shapes = downstream_summaries[depth_key]["output_shapes"]
        print(
            f"L={depth_key} U0={tuple(shapes['U0'])} U_out_full={tuple(shapes['U_out_full'])} "
            f"I_out_full={tuple(shapes['I_out_full'])} I_out_roi={tuple(shapes['I_out_roi'])}"
        )


if __name__ == "__main__":
    main()
