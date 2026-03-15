#!/usr/bin/env python
"""Minimal decoder-only single-sample fitting for Stage 3 Issue 5."""

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

# Keep `python scripts/train_decoder_only_single_sample.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.optics.diffractive_decoder import DiffractiveDecoder


INPUT_PATTERN_HW = (24, 24)
LAYER_HW = (32, 32)
PROPAGATION_HW = (48, 48)
OUTPUT_CROP_HW = (20, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 decoder-only single-sample fitting for optical capacity sanity."
    )
    parser.add_argument("--depth", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr-phase", type=float, default=0.15)
    parser.add_argument("--lr-decoder", type=float, default=0.03)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/optics/decoder_only_single_sample",
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


def build_synthetic_target(device: torch.device) -> torch.Tensor:
    """Create a deterministic nontrivial ROI target image."""
    height, width = OUTPUT_CROP_HW
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    gaussian_a = torch.exp(-((xx + 0.35) ** 2 + (yy - 0.15) ** 2) / 0.09)
    gaussian_b = 0.85 * torch.exp(-((xx - 0.30) ** 2 + (yy + 0.30) ** 2) / 0.05)
    vertical_bar = ((xx.abs() < 0.10) & (yy > -0.75) & (yy < 0.55)).to(torch.float32) * 0.35
    diagonal_band = ((yy - 0.55 * xx).abs() < 0.12).to(torch.float32) * 0.25

    target = gaussian_a + gaussian_b + vertical_bar + diagonal_band
    target = target / target.max().clamp_min(1e-6)
    return target.unsqueeze(0).unsqueeze(0)


def constrain_input_phase(phase_raw: torch.Tensor) -> torch.Tensor:
    """Keep the learnable input phase in a physical-looking range."""
    return math.pi * torch.tanh(phase_raw)


def normalized_mae_roi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 3 engineering default normalized MAE on ROI."""
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


def save_image_triptych(
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


def save_loss_curve(output_path: Path, loss_history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, linewidth=1.5)
    ax.set_title("Decoder-Only Single-Sample Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    decoder = build_decoder(args.depth, device=device)
    decoder.train()

    target_roi = build_synthetic_target(device)
    input_phase_raw = nn.Parameter(
        0.05 * torch.randn(1, 1, *INPUT_PATTERN_HW, dtype=torch.float32, device=device)
    )

    optim_groups: list[dict[str, Any]] = [
        {"params": [input_phase_raw], "lr": args.lr_phase},
    ]
    if args.freeze_decoder:
        decoder.requires_grad_(False)
    else:
        optim_groups.append({"params": decoder.parameters(), "lr": args.lr_decoder})

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

    loss_history: list[float] = []
    best_loss = initial_loss
    best_step = -1
    best_roi = initial_output["I_out_roi"].detach().cpu()
    best_phase = initial_phase.detach().cpu()
    best_sigma = float(initial_sigma.mean().item())

    for step in range(args.steps):
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
    final_roi_std = float(final_roi.std(unbiased=False).item())
    best_roi_std = float(best_roi.std(unbiased=False).item())
    target_roi_cpu = target_roi.detach().cpu()

    if best_loss >= initial_loss:
        raise AssertionError(
            f"Loss did not decrease: initial={initial_loss:.6f}, best={best_loss:.6f}."
        )
    if roi_change <= 1e-5:
        raise AssertionError(f"I_out_roi change is too small: {roi_change:.6e}.")
    if best_roi_std <= 1e-6:
        raise AssertionError(f"Best ROI output looks trivially constant: std={best_roi_std:.6e}.")

    artifact_payload = {
        "target_roi": target_roi_cpu,
        "initial_roi": initial_roi,
        "best_roi": best_roi,
        "final_roi": final_roi,
        "best_phase": best_phase,
        "final_phase": final_phase.detach().cpu(),
        "loss_history": torch.tensor(loss_history, dtype=torch.float32),
    }
    torch.save(artifact_payload, output_dir / "artifacts.pt")

    save_image_triptych(
        output_dir / "roi_triptych.png",
        target=target_roi_cpu,
        initial_roi=initial_roi,
        best_roi=best_roi,
        final_roi=final_roi,
    )
    save_loss_curve(output_dir / "loss_curve.png", loss_history=loss_history)

    summary = {
        "depth": args.depth,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "freeze_decoder": args.freeze_decoder,
        "input_pattern_hw": list(INPUT_PATTERN_HW),
        "output_crop_hw": list(OUTPUT_CROP_HW),
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "best_step": best_step,
        "best_sigma": best_sigma,
        "final_sigma": float(final_sigma.mean().item()),
        "loss_decrease": initial_loss - best_loss,
        "roi_change_mean_abs": roi_change,
        "best_roi_std": best_roi_std,
        "final_roi_std": final_roi_std,
        "artifact_file": str((output_dir / "artifacts.pt").resolve()),
        "roi_triptych": str((output_dir / "roi_triptych.png").resolve()),
        "loss_curve": str((output_dir / "loss_curve.png").resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Stage 3 Issue 5 decoder-only single-sample fitting completed.")
    print(f"depth={args.depth}, freeze_decoder={args.freeze_decoder}, steps={args.steps}")
    print(f"initial_loss={initial_loss:.6f}")
    print(f"best_loss={best_loss:.6f} at step={best_step}")
    print(f"final_loss={final_loss:.6f}")
    print(f"roi_change_mean_abs={roi_change:.6e}")
    print(f"best_roi_std={best_roi_std:.6e}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
