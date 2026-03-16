#!/usr/bin/env python
"""Minimal decoder-only small-subset fitting for Stage 3 Issue 6."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Keep `python scripts/train_decoder_only_small_subset.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optics.diffractive_decoder import DiffractiveDecoder


INPUT_PATTERN_HW = (24, 24)
LAYER_HW = (32, 32)
PROPAGATION_HW = (48, 48)
OUTPUT_CROP_HW = (20, 20)

TARGET_LIBRARY: tuple[dict[str, object], ...] = (
    {"center_a": (-0.40, -0.15), "center_b": (0.28, 0.30), "bar": "vertical", "slope": 0.55},
    {"center_a": (-0.18, 0.28), "center_b": (0.36, -0.34), "bar": "horizontal", "slope": -0.60},
    {"center_a": (0.22, -0.28), "center_b": (-0.32, 0.18), "bar": "cross", "slope": 0.00},
    {"center_a": (0.00, 0.00), "center_b": (-0.42, -0.34), "bar": "vertical", "slope": -0.35},
    {"center_a": (0.34, 0.10), "center_b": (-0.28, -0.22), "bar": "horizontal", "slope": 0.80},
    {"center_a": (-0.30, 0.04), "center_b": (0.16, -0.36), "bar": "cross", "slope": -0.72},
    {"center_a": (0.12, 0.34), "center_b": (-0.36, -0.08), "bar": "vertical", "slope": 0.28},
    {"center_a": (-0.08, -0.38), "center_b": (0.40, 0.22), "bar": "horizontal", "slope": -0.18},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 decoder-only small-subset optical capacity sanity."
    )
    parser.add_argument("--depth", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr-phase", type=float, default=0.15)
    parser.add_argument("--lr-decoder", type=float, default=0.03)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/optics/decoder_only_small_subset",
    )
    return parser.parse_args()


def normalize_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    return torch.device(device_arg)


def build_distance_schedule(num_diffractive_layers: int) -> dict[str, object]:
    return {
        "input_to_first": 0.01,
        "inter_layer": [0.02] * max(0, num_diffractive_layers - 1),
        "last_to_sensor": 0.03,
    }


def build_decoder(num_diffractive_layers: int, device: torch.device) -> DiffractiveDecoder:
    decoder = DiffractiveDecoder(
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
    return decoder.to(device)


def build_synthetic_target(sample_index: int, device: torch.device) -> torch.Tensor:
    """Create a deterministic target image for the requested subset index."""
    if sample_index >= len(TARGET_LIBRARY):
        raise ValueError(
            f"sample_index={sample_index} exceeds target library size {len(TARGET_LIBRARY)}."
        )

    cfg = TARGET_LIBRARY[sample_index]
    height, width = OUTPUT_CROP_HW
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    center_a_x, center_a_y = cfg["center_a"]  # type: ignore[misc]
    center_b_x, center_b_y = cfg["center_b"]  # type: ignore[misc]
    gaussian_a = torch.exp(-((xx - center_a_x) ** 2 + (yy - center_a_y) ** 2) / 0.08)
    gaussian_b = 0.85 * torch.exp(-((xx - center_b_x) ** 2 + (yy - center_b_y) ** 2) / 0.05)

    bar = str(cfg["bar"])
    if bar == "vertical":
        bar_component = ((xx.abs() < 0.10) & (yy > -0.8) & (yy < 0.6)).to(torch.float32) * 0.32
    elif bar == "horizontal":
        bar_component = ((yy.abs() < 0.10) & (xx > -0.7) & (xx < 0.75)).to(torch.float32) * 0.32
    elif bar == "cross":
        vertical = ((xx.abs() < 0.08) & (yy > -0.8) & (yy < 0.7)).to(torch.float32) * 0.18
        horizontal = ((yy.abs() < 0.08) & (xx > -0.8) & (xx < 0.8)).to(torch.float32) * 0.18
        bar_component = vertical + horizontal
    else:
        raise ValueError(f"Unsupported bar pattern: {bar}.")

    slope = float(cfg["slope"])
    diagonal_band = ((yy - slope * xx).abs() < 0.12).to(torch.float32) * 0.22

    target = gaussian_a + gaussian_b + bar_component + diagonal_band
    target = target / target.max().clamp_min(1e-6)
    return target.unsqueeze(0).unsqueeze(0)


def constrain_input_phase(phase_raw: torch.Tensor) -> torch.Tensor:
    return math.pi * torch.tanh(phase_raw)


def normalized_mae_roi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must match, got {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )

    reduce_dims = tuple(range(2, prediction.ndim))
    sigma = target.sum(dim=reduce_dims, keepdim=True) / (
        prediction.sum(dim=reduce_dims, keepdim=True) + epsilon
    )
    prediction_rescaled = sigma * prediction
    per_sample_mae = torch.mean(torch.abs(target - prediction_rescaled), dim=reduce_dims)
    loss = per_sample_mae.mean()
    return loss, sigma


def save_roi_triptych(
    output_path: Path,
    *,
    target: torch.Tensor,
    initial_roi: torch.Tensor,
    best_roi: torch.Tensor,
    final_roi: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    items = (
        ("Target ROI", target),
        ("Initial ROI", initial_roi),
        ("Best ROI", best_roi),
        ("Final ROI", final_roi),
    )
    for ax, (title, tensor) in zip(axes, items):
        image = tensor.detach().cpu().squeeze().numpy()
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(output_path: Path, loss_history: list[float], *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_single_target(
    *,
    sample_index: int,
    depth: int,
    steps: int,
    seed: int,
    device: torch.device,
    lr_phase: float,
    lr_decoder: float,
    freeze_decoder: bool,
    sample_dir: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed + sample_index)

    decoder = build_decoder(depth, device=device)
    decoder.train()
    target_roi = build_synthetic_target(sample_index, device=device)
    input_phase_raw = nn.Parameter(
        0.05 * torch.randn(1, 1, *INPUT_PATTERN_HW, dtype=torch.float32, device=device)
    )

    optim_groups: list[dict[str, Any]] = [
        {"params": [input_phase_raw], "lr": lr_phase},
    ]
    if freeze_decoder:
        decoder.requires_grad_(False)
    else:
        optim_groups.append({"params": decoder.parameters(), "lr": lr_decoder})
    optimizer = torch.optim.Adam(optim_groups)

    with torch.no_grad():
        initial_phase = constrain_input_phase(input_phase_raw)
        initial_output = decoder(initial_phase)
        initial_loss_tensor, initial_sigma = normalized_mae_roi(
            initial_output["I_out_roi"],
            target_roi,
        )
        initial_loss = float(initial_loss_tensor.item())
        initial_roi = initial_output["I_out_roi"].detach().cpu()

    best_loss = initial_loss
    best_step = -1
    best_roi = initial_roi
    best_phase = initial_phase.detach().cpu()
    best_sigma = float(initial_sigma.mean().item())
    loss_history: list[float] = []

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        phase_input = constrain_input_phase(input_phase_raw)
        output = decoder(phase_input)
        loss, sigma = normalized_mae_roi(output["I_out_roi"], target_roi)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step
            best_roi = output["I_out_roi"].detach().cpu()
            best_phase = phase_input.detach().cpu()
            best_sigma = float(sigma.mean().item())

    with torch.no_grad():
        final_phase = constrain_input_phase(input_phase_raw)
        final_output = decoder(final_phase)
        final_loss_tensor, final_sigma = normalized_mae_roi(
            final_output["I_out_roi"],
            target_roi,
        )
        final_loss = float(final_loss_tensor.item())
        final_roi = final_output["I_out_roi"].detach().cpu()

    roi_change = float(torch.mean(torch.abs(final_roi - initial_roi)).item())
    best_roi_std = float(best_roi.std(unbiased=False).item())
    success = (best_loss < initial_loss) and (roi_change > 1e-5) and (best_roi_std > 1e-6)

    target_roi_cpu = target_roi.detach().cpu()
    sample_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_roi": target_roi_cpu,
            "initial_roi": initial_roi,
            "best_roi": best_roi,
            "final_roi": final_roi,
            "best_phase": best_phase,
            "final_phase": final_phase.detach().cpu(),
            "loss_history": torch.tensor(loss_history, dtype=torch.float32),
        },
        sample_dir / "artifacts.pt",
    )
    save_roi_triptych(
        sample_dir / "roi_triptych.png",
        target=target_roi_cpu,
        initial_roi=initial_roi,
        best_roi=best_roi,
        final_roi=final_roi,
    )
    save_loss_curve(
        sample_dir / "loss_curve.png",
        loss_history=loss_history,
        title=f"Sample {sample_index} Loss",
    )

    sample_summary = {
        "sample_index": sample_index,
        "depth": depth,
        "steps": steps,
        "freeze_decoder": freeze_decoder,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "best_step": best_step,
        "best_sigma": best_sigma,
        "final_sigma": float(final_sigma.mean().item()),
        "loss_decrease": initial_loss - best_loss,
        "roi_change_mean_abs": roi_change,
        "best_roi_std": best_roi_std,
        "success": success,
        "artifact_file": str((sample_dir / "artifacts.pt").resolve()),
        "roi_triptych": str((sample_dir / "roi_triptych.png").resolve()),
        "loss_curve": str((sample_dir / "loss_curve.png").resolve()),
    }
    with (sample_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(sample_summary, file, indent=2, ensure_ascii=False)

    return sample_summary


def run_subset_validation(
    *,
    depth: int,
    subset_size: int,
    steps: int,
    seed: int,
    device: torch.device,
    lr_phase: float,
    lr_decoder: float,
    freeze_decoder: bool,
    output_dir: Path,
) -> dict[str, Any]:
    if subset_size < 1:
        raise ValueError("--subset-size must be >= 1.")
    if subset_size > len(TARGET_LIBRARY):
        raise ValueError(
            f"--subset-size ({subset_size}) exceeds target library size {len(TARGET_LIBRARY)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_summaries: list[dict[str, Any]] = []
    for sample_index in range(subset_size):
        sample_dir = output_dir / f"sample_{sample_index:03d}"
        sample_summary = fit_single_target(
            sample_index=sample_index,
            depth=depth,
            steps=steps,
            seed=seed,
            device=device,
            lr_phase=lr_phase,
            lr_decoder=lr_decoder,
            freeze_decoder=freeze_decoder,
            sample_dir=sample_dir,
        )
        sample_summaries.append(sample_summary)

    success_count = sum(1 for sample in sample_summaries if sample["success"])
    best_losses = [float(sample["best_loss"]) for sample in sample_summaries]
    initial_losses = [float(sample["initial_loss"]) for sample in sample_summaries]
    mean_initial_loss = sum(initial_losses) / len(initial_losses)
    mean_best_loss = sum(best_losses) / len(best_losses)
    total_count = len(sample_summaries)
    success_rate = success_count / total_count

    if success_count < subset_size:
        failed_samples = [sample["sample_index"] for sample in sample_summaries if not sample["success"]]
        raise AssertionError(
            f"Not all subset samples fit successfully. Failed sample indices: {failed_samples}."
        )

    subset_summary = {
        "depth": depth,
        "subset_size": subset_size,
        "steps": steps,
        "seed": seed,
        "device": str(device),
        "freeze_decoder": freeze_decoder,
        "input_pattern_hw": list(INPUT_PATTERN_HW),
        "output_crop_hw": list(OUTPUT_CROP_HW),
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "mean_initial_loss": mean_initial_loss,
        "mean_best_loss": mean_best_loss,
        "mean_loss_decrease": mean_initial_loss - mean_best_loss,
        "protocol": {
            "depth": depth,
            "subset_size": subset_size,
            "steps": steps,
            "seed": seed,
            "device": str(device),
            "lr_phase": lr_phase,
            "lr_decoder": lr_decoder,
            "freeze_decoder": freeze_decoder,
            "loss_name": "normalized_mae_roi",
            "input_pattern_hw": list(INPUT_PATTERN_HW),
            "layer_hw": list(LAYER_HW),
            "propagation_hw": list(PROPAGATION_HW),
            "output_crop_hw": list(OUTPUT_CROP_HW),
        },
        "sample_summaries": sample_summaries,
    }
    with (output_dir / "subset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(subset_summary, file, indent=2, ensure_ascii=False)

    return subset_summary


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)
    output_dir = Path(args.output_dir)

    subset_summary = run_subset_validation(
        depth=args.depth,
        subset_size=args.subset_size,
        steps=args.steps,
        seed=args.seed,
        device=device,
        lr_phase=args.lr_phase,
        lr_decoder=args.lr_decoder,
        freeze_decoder=args.freeze_decoder,
        output_dir=output_dir,
    )

    print("Stage 3 Issue 6 decoder-only small-subset validation completed.")
    print(
        f"depth={subset_summary['depth']}, subset_size={subset_summary['subset_size']}, "
        f"steps={subset_summary['steps']}, freeze_decoder={subset_summary['freeze_decoder']}"
    )
    print(f"success_count={subset_summary['success_count']}/{subset_summary['total_count']}")
    print(f"mean_initial_loss={subset_summary['mean_initial_loss']:.6f}")
    print(f"mean_best_loss={subset_summary['mean_best_loss']:.6f}")
    print(f"subset_summary={output_dir / 'subset_summary.json'}")


if __name__ == "__main__":
    main()
