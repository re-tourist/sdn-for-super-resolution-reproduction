#!/usr/bin/env python
"""Stage 5 loss sanity for normalized MAE plus efficiency penalty."""

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

from src.losses import Stage5SuperResolutionLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage5/stage5_sr_loss.yaml",
        help="Path to the Stage 5 loss YAML config.",
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
        raise RuntimeError("PyYAML is required to load the Stage 5 loss config.") from exc

    with Path(path).open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping at YAML root, got {type(payload)!r}.")
    return payload


def make_fake_batch(batch_size: int, roi_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = roi_hw
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    target_samples: list[torch.Tensor] = []
    prediction_samples: list[torch.Tensor] = []
    for sample_index in range(batch_size):
        shift = 0.15 * sample_index
        target = 0.03 + 0.02 * torch.exp(-((xx + 0.25 - shift) ** 2 + (yy - 0.10) ** 2) / 0.10)
        target += 0.015 * torch.exp(-((xx - 0.30) ** 2 + (yy + 0.20 - shift) ** 2) / 0.08)
        target += 0.010 * ((yy - 0.55 * xx).abs() < 0.10).to(torch.float32)

        prediction = 0.82 * target
        prediction += 0.006 * torch.cos((sample_index + 1) * math.pi * xx)
        prediction += 0.004 * torch.sin((sample_index + 2) * math.pi * yy)
        prediction = prediction.clamp_min(1e-4)

        target_samples.append(target.unsqueeze(0))
        prediction_samples.append(prediction.unsqueeze(0))

    target_roi = torch.stack(target_samples, dim=0)
    prediction_roi = torch.stack(prediction_samples, dim=0)
    return prediction_roi, target_roi


def assert_finite(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        raise AssertionError(f"{name} must not be None.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def build_loss(loss_config: dict[str, Any], *, enable_efficiency_term: bool) -> Stage5SuperResolutionLoss:
    efficiency = loss_config["efficiency_term"]
    return Stage5SuperResolutionLoss(
        epsilon=float(loss_config["epsilon"]),
        sigma_mode=str(loss_config["sigma_mode"]),
        gamma_by_depth=efficiency["gamma_by_depth"],
        enable_efficiency_term=enable_efficiency_term,
        eta_scale=float(efficiency["eta_scale"]),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    loss_config = config["loss"]["stage5_sr"]

    torch.manual_seed(int(loss_config["sanity"]["seed"]))
    roi_hw = tuple(int(v) for v in loss_config["sanity"]["roi_hw"])
    batch_size = int(loss_config["sanity"]["batch_size"])
    depths = [int(depth) for depth in loss_config["sanity"]["depths"]]

    prediction_roi, target_roi = make_fake_batch(batch_size=batch_size, roi_hw=roi_hw)
    input_power = torch.tensor([50000.0, 52000.0], dtype=torch.float32)
    if batch_size != input_power.shape[0]:
        raise AssertionError("Sanity input_power must match configured batch size.")

    loss_enabled = build_loss(loss_config, enable_efficiency_term=True)
    loss_disabled = build_loss(loss_config, enable_efficiency_term=False)

    per_depth_summary: dict[str, Any] = {}
    for depth in depths:
        enabled = loss_enabled(
            prediction_roi,
            target_roi,
            depth=depth,
            input_power=input_power,
        )
        disabled = loss_disabled(
            prediction_roi,
            target_roi,
            depth=depth,
            input_power=None,
        )

        assert_finite("enabled.loss", enabled["loss"])
        assert_finite("enabled.mae_term", enabled["mae_term"])
        assert_finite("enabled.sigma", enabled["sigma"])
        assert_finite("disabled.loss", disabled["loss"])
        assert_finite("disabled.mae_term", disabled["mae_term"])
        assert_finite("disabled.sigma", disabled["sigma"])
        assert_finite("enabled.eta", enabled["eta"])

        enabled_loss = float(enabled["loss"].item())
        disabled_loss = float(disabled["loss"].item())
        efficiency_term = float(enabled["efficiency_term"].item())
        gamma = float(enabled["gamma"].item())
        eta = enabled["eta"]
        assert isinstance(eta, torch.Tensor)

        if depth in (1, 3):
            if efficiency_term <= 0.0:
                raise AssertionError(f"Expected positive efficiency term for L={depth}, got {efficiency_term}.")
            if abs(enabled_loss - disabled_loss) <= 1e-9:
                raise AssertionError(f"Enabled and disabled losses should differ for L={depth}.")
        if depth == 5:
            if gamma != 0.0:
                raise AssertionError(f"Stage 5 L=5 gamma must be explicit and currently 0.0, got {gamma}.")
            if efficiency_term != 0.0:
                raise AssertionError(f"Expected zero efficiency term for L=5 with gamma=0, got {efficiency_term}.")

        per_depth_summary[str(depth)] = {
            "gamma": gamma,
            "enabled_loss": enabled_loss,
            "disabled_loss": disabled_loss,
            "mae_term": float(enabled["mae_term"].item()),
            "efficiency_term": efficiency_term,
            "sigma_mean": float(enabled["sigma"].mean().item()),
            "eta_mean": float(eta.mean().item()),
            "toggle_changes_total_loss": abs(enabled_loss - disabled_loss) > 1e-9,
        }

    output_root = Path(args.output_root or loss_config["sanity"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_path": str(Path(args.config).as_posix()),
        "loss": {
            "epsilon": float(loss_config["epsilon"]),
            "sigma_mode": str(loss_config["sigma_mode"]),
            "aggregation": str(loss_config["aggregation"]),
            "efficiency_term_enabled": bool(loss_config["efficiency_term"]["enabled"]),
            "eta_scale": float(loss_config["efficiency_term"]["eta_scale"]),
            "output_power_source": str(loss_config["efficiency_term"]["output_power_source"]),
            "input_power_source": str(loss_config["efficiency_term"]["input_power_source"]),
            "gamma_by_depth": {
                str(key): float(value)
                for key, value in loss_config["efficiency_term"]["gamma_by_depth"].items()
            },
        },
        "observed": {
            "prediction_roi_shape": list(prediction_roi.shape),
            "target_roi_shape": list(target_roi.shape),
            "input_power": [float(v) for v in input_power.tolist()],
        },
        "depths": per_depth_summary,
    }
    summary_path = output_root / "stage5_sr_loss_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Stage 5 SR loss sanity passed.")
    print(f"config={args.config}")
    print(f"summary={summary_path.as_posix()}")
    print(
        f"prediction_roi_shape={tuple(prediction_roi.shape)} "
        f"target_roi_shape={tuple(target_roi.shape)}"
    )
    for depth in depths:
        depth_summary = per_depth_summary[str(depth)]
        print(
            f"L={depth} gamma={depth_summary['gamma']:.6f} "
            f"enabled_loss={depth_summary['enabled_loss']:.6f} "
            f"disabled_loss={depth_summary['disabled_loss']:.6f} "
            f"efficiency_term={depth_summary['efficiency_term']:.6f} "
            f"sigma_mean={depth_summary['sigma_mean']:.6f} "
            f"eta_mean={depth_summary['eta_mean']:.6f}"
        )


if __name__ == "__main__":
    main()
