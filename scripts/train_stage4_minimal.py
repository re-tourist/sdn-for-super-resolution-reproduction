#!/usr/bin/env python
"""Train the Stage 4 minimal closed-loop hybrid system.

This script is intentionally small and explicitly non-paper-final. Its job is
to answer one engineering question under the frozen Stage 3 optical contract:

Can `x_hr -> encoder -> phi_lr -> optical decoder -> I_out_roi` learn on a
real minimal data path once we supervise it with `target_roi`?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

# Keep `python scripts/train_stage4_minimal.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import Stage4RoiTargetAdapter, build_stage4_train_val_datasets
from src.eval.metrics import compute_psnr, compute_ssim
from src.models.encoders import MinimalPhaseEncoder
from src.models.hybrid import MinimalHybridWrapper
from src.models.optics.diffractive_decoder import DiffractiveDecoder


TRAIN_DEFAULTS: dict[str, Any] = {
    "output_root": "outputs/stage4/minimal_trainer",
    "steps": 100,
    "subset_size": 4,
    "batch_size": None,
    "seed": 42,
    "device": "auto",
    "num_workers": 0,
    "download": False,
    "dataset_root": "data/raw/emnist",
    "emnist_split": "letters",
    "val_ratio": 0.25,
    "hr_size": 96,
    "depth": 1,
    "input_pattern_hw": 24,
    "layer_hw": 32,
    "propagation_hw": 48,
    "output_crop_hw": 20,
    "wavelength": 532e-9,
    "pixel_pitch": 8e-6,
    "lr_encoder": 5e-4,
    "lr_optics": 1e-3,
    "encoder_base_channels": 16,
    "encoder_hidden_channels": None,
    "freeze_encoder": False,
    "freeze_optics": False,
    "val_every": 10,
    "preview_limit": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 minimal trainer for encoder + optical decoder closed-loop learning."
    )
    parser.add_argument("--output-dir", type=str, default=TRAIN_DEFAULTS["output_root"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--single-sample", action="store_true")
    parser.add_argument("--subset-size", type=int, default=TRAIN_DEFAULTS["subset_size"])
    parser.add_argument("--steps", type=int, default=TRAIN_DEFAULTS["steps"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=TRAIN_DEFAULTS["seed"])
    parser.add_argument("--device", type=str, default=TRAIN_DEFAULTS["device"])
    parser.add_argument("--num-workers", type=int, default=TRAIN_DEFAULTS["num_workers"])
    parser.add_argument("--dataset-root", type=str, default=TRAIN_DEFAULTS["dataset_root"])
    parser.add_argument("--download", action="store_true", default=TRAIN_DEFAULTS["download"])
    parser.add_argument("--emnist-split", type=str, default=TRAIN_DEFAULTS["emnist_split"])
    parser.add_argument("--val-ratio", type=float, default=TRAIN_DEFAULTS["val_ratio"])
    parser.add_argument("--hr-size", type=int, default=TRAIN_DEFAULTS["hr_size"])
    parser.add_argument("--depth", type=int, choices=(1, 3, 5), default=TRAIN_DEFAULTS["depth"])
    parser.add_argument(
        "--input-pattern-hw", type=int, default=TRAIN_DEFAULTS["input_pattern_hw"]
    )
    parser.add_argument("--layer-hw", type=int, default=TRAIN_DEFAULTS["layer_hw"])
    parser.add_argument(
        "--propagation-hw", type=int, default=TRAIN_DEFAULTS["propagation_hw"]
    )
    parser.add_argument("--output-crop-hw", type=int, default=TRAIN_DEFAULTS["output_crop_hw"])
    parser.add_argument("--wavelength", type=float, default=TRAIN_DEFAULTS["wavelength"])
    parser.add_argument("--pixel-pitch", type=float, default=TRAIN_DEFAULTS["pixel_pitch"])
    parser.add_argument("--lr-encoder", type=float, default=TRAIN_DEFAULTS["lr_encoder"])
    parser.add_argument("--lr-optics", type=float, default=TRAIN_DEFAULTS["lr_optics"])
    parser.add_argument(
        "--encoder-base-channels",
        type=int,
        default=TRAIN_DEFAULTS["encoder_base_channels"],
    )
    parser.add_argument(
        "--encoder-hidden-channels",
        type=int,
        default=TRAIN_DEFAULTS["encoder_hidden_channels"],
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--freeze-optics", action="store_true")
    parser.add_argument("--val-every", type=int, default=TRAIN_DEFAULTS["val_every"])
    parser.add_argument("--preview-limit", type=int, default=TRAIN_DEFAULTS["preview_limit"])
    return parser.parse_args()


def normalize_device(device_arg: str) -> torch.device:
    normalized = device_arg.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported --device value: {device_arg}.")
    return torch.device(normalized)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.hr_size < 1:
        raise ValueError("--hr-size must be >= 1.")
    if args.subset_size < 1:
        raise ValueError("--subset-size must be >= 1.")
    if not args.single_sample and args.subset_size < 2:
        raise ValueError("Small-subset mode needs --subset-size >= 2.")
    if args.batch_size is None:
        args.batch_size = 1 if args.single_sample else min(args.subset_size, 4)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.preview_limit < 1:
        raise ValueError("--preview-limit must be >= 1.")
    if args.val_every < 1:
        raise ValueError("--val-every must be >= 1.")
    if args.freeze_encoder and args.freeze_optics:
        raise ValueError("At least one of encoder/optics must stay trainable.")
    if args.lr_encoder <= 0.0:
        raise ValueError("--lr-encoder must be positive.")
    if args.lr_optics <= 0.0:
        raise ValueError("--lr-optics must be positive.")
    if args.wavelength <= 0.0:
        raise ValueError("--wavelength must be positive.")
    if args.pixel_pitch <= 0.0:
        raise ValueError("--pixel-pitch must be positive.")

    args.mode = "single_sample_overfit" if args.single_sample else "small_subset_sanity"
    args.dataset_num_samples = 1 if args.single_sample else args.subset_size
    return args


def build_distance_schedule(num_diffractive_layers: int) -> dict[str, object]:
    return {
        "input_to_first": 0.01,
        "inter_layer": [0.02] * max(0, num_diffractive_layers - 1),
        "last_to_sensor": 0.03,
    }


def build_stage4_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MinimalHybridWrapper, Stage4RoiTargetAdapter]:
    input_pattern_hw = (args.input_pattern_hw, args.input_pattern_hw)
    layer_hw = (args.layer_hw, args.layer_hw)
    propagation_hw = (args.propagation_hw, args.propagation_hw)
    output_crop_hw = (args.output_crop_hw, args.output_crop_hw)

    encoder = MinimalPhaseEncoder(
        in_channels=1,
        target_hw=input_pattern_hw,
        base_channels=args.encoder_base_channels,
        hidden_channels=args.encoder_hidden_channels,
        phase_range=(-math.pi, math.pi),
    )
    decoder = DiffractiveDecoder(
        num_diffractive_layers=args.depth,
        wavelength=args.wavelength,
        pixel_pitch=args.pixel_pitch,
        grid_config={
            "input_pattern_hw": input_pattern_hw,
            "layer_hw": layer_hw,
            "propagation_hw": propagation_hw,
        },
        distance_schedule=build_distance_schedule(args.depth),
        readout_config={"output_crop_hw": output_crop_hw},
        phase_mask_config={
            "init_mode": "zeros",
            "phase_mapping": "tanh",
            "phase_range": (-math.pi, math.pi),
        },
    )
    wrapper = MinimalHybridWrapper(
        encoder=encoder.to(device),
        decoder=decoder.to(device),
    )
    if args.freeze_encoder:
        wrapper.encoder.requires_grad_(False)
    if args.freeze_optics:
        wrapper.decoder.requires_grad_(False)

    target_adapter = Stage4RoiTargetAdapter(output_crop_hw=wrapper.decoder.readout_config)
    return wrapper, target_adapter


def build_optimizer(model: MinimalHybridWrapper, args: argparse.Namespace) -> Adam:
    parameter_groups: list[dict[str, Any]] = []

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": args.lr_encoder})

    optics_params = [p for p in model.decoder.parameters() if p.requires_grad]
    if optics_params:
        parameter_groups.append({"params": optics_params, "lr": args.lr_optics})

    if not parameter_groups:
        raise RuntimeError("No trainable parameters found for the Stage 4 trainer.")

    return Adam(parameter_groups)


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


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 2:
        return tensor.numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].numpy()
    raise ValueError(f"Expected grayscale tensor, got shape {tuple(tensor.shape)}.")


def collect_module_grad_stats(module: torch.nn.Module) -> dict[str, Any]:
    trainable_tensor_count = 0
    grad_tensor_count = 0
    nonzero_grad_tensor_count = 0
    mean_abs_values: list[float] = []
    max_abs_value = 0.0
    grad_l2_sq = 0.0

    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensor_count += 1
        grad = parameter.grad
        if grad is None:
            continue
        grad_tensor_count += 1
        grad_detached = grad.detach()
        abs_grad = grad_detached.abs()
        mean_abs = float(abs_grad.mean().item())
        max_abs = float(abs_grad.max().item())
        mean_abs_values.append(mean_abs)
        max_abs_value = max(max_abs_value, max_abs)
        grad_l2_sq += float(torch.sum(grad_detached.square()).item())
        if max_abs > 0.0:
            nonzero_grad_tensor_count += 1

    return {
        "trainable_tensor_count": trainable_tensor_count,
        "grad_tensor_count": grad_tensor_count,
        "nonzero_grad_tensor_count": nonzero_grad_tensor_count,
        "has_any_grad": grad_tensor_count > 0,
        "has_nonzero_grad": nonzero_grad_tensor_count > 0,
        "mean_abs_grad": (
            sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else 0.0
        ),
        "max_abs_grad": max_abs_value,
        "l2_grad_norm": math.sqrt(grad_l2_sq),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def run_stage4_batch(
    model: MinimalHybridWrapper,
    batch: dict[str, Any],
    *,
    target_adapter: Stage4RoiTargetAdapter,
    device: torch.device,
    return_encoder_debug: bool = False,
) -> dict[str, Any]:
    moved_batch = move_batch_to_device(batch, device)
    x_hr = moved_batch["x_hr"]
    target_hr = moved_batch["target_hr"]
    if not isinstance(x_hr, torch.Tensor) or not isinstance(target_hr, torch.Tensor):
        raise TypeError("Batch must contain tensor fields 'x_hr' and 'target_hr'.")

    target_roi = target_adapter(target_hr)
    output = model(
        x_hr,
        return_intermediates=False,
        return_encoder_debug=return_encoder_debug,
    )
    pred_roi = output["I_out_roi"]
    phi_lr = output["phi_lr"]
    if not isinstance(pred_roi, torch.Tensor) or not isinstance(phi_lr, torch.Tensor):
        raise TypeError("Wrapper output is missing tensor fields 'I_out_roi' or 'phi_lr'.")

    loss, sigma = normalized_mae_roi(pred_roi, target_roi)
    _assert_finite("x_hr", x_hr)
    _assert_finite("target_roi", target_roi)
    _assert_finite("phi_lr", phi_lr)
    _assert_finite("I_out_roi", pred_roi)
    _assert_finite("loss", loss.unsqueeze(0))
    return {
        "x_hr": x_hr,
        "target_hr": target_hr,
        "target_roi": target_roi,
        "output": output,
        "pred_roi": pred_roi,
        "loss": loss,
        "sigma": sigma,
    }


@torch.no_grad()
def evaluate_loader(
    model: MinimalHybridWrapper,
    loader: DataLoader,
    *,
    target_adapter: Stage4RoiTargetAdapter,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_list: list[torch.Tensor] = []
    target_list: list[torch.Tensor] = []

    for batch in loader:
        batch_output = run_stage4_batch(
            model,
            batch,
            target_adapter=target_adapter,
            device=device,
            return_encoder_debug=False,
        )
        batch_size = batch_output["x_hr"].shape[0]
        total_loss += float(batch_output["loss"].item()) * batch_size
        total_count += batch_size
        pred_list.append(batch_output["pred_roi"].detach().cpu())
        target_list.append(batch_output["target_roi"].detach().cpu())

    if total_count == 0:
        raise RuntimeError("Validation loader is empty.")

    preds = torch.cat(pred_list, dim=0)
    targets = torch.cat(target_list, dim=0)
    return {
        "val_loss": total_loss / total_count,
        "val_psnr": compute_psnr(preds, targets),
        "val_ssim": compute_ssim(preds, targets),
    }


def snapshot_from_batch_output(
    batch_output: dict[str, Any],
    *,
    preview_limit: int,
) -> dict[str, torch.Tensor]:
    output = batch_output["output"]
    raw_phase = output.get("raw_phase")
    snapshot = {
        "x_hr": batch_output["x_hr"][:preview_limit].detach().cpu(),
        "target_roi": batch_output["target_roi"][:preview_limit].detach().cpu(),
        "pred_roi": batch_output["pred_roi"][:preview_limit].detach().cpu(),
        "phi_lr": output["phi_lr"][:1].detach().cpu(),
    }
    if isinstance(raw_phase, torch.Tensor):
        snapshot["raw_phase"] = raw_phase[:1].detach().cpu()
    return snapshot


def save_preview_grid(output_path: Path, snapshot: dict[str, torch.Tensor]) -> None:
    x_hr = snapshot["x_hr"]
    target_roi = snapshot["target_roi"]
    pred_roi = snapshot["pred_roi"]
    row_count = int(x_hr.shape[0])
    if row_count < 1:
        raise ValueError("Need at least one sample to save a preview grid.")

    target_up = torch.nn.functional.interpolate(
        target_roi,
        size=tuple(x_hr.shape[-2:]),
        mode="nearest",
    )
    pred_up = torch.nn.functional.interpolate(
        pred_roi,
        size=tuple(x_hr.shape[-2:]),
        mode="nearest",
    )
    diff_up = torch.abs(pred_up - target_up)

    fig, axes = plt.subplots(row_count, 4, figsize=(12, max(3.2, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ("x_hr", "target_roi", "pred I_out_roi", "|pred-target|")
    for row_idx in range(row_count):
        panels = (
            _to_gray_numpy(x_hr[row_idx]),
            _to_gray_numpy(target_up[row_idx]),
            _to_gray_numpy(pred_up[row_idx]),
            _to_gray_numpy(diff_up[row_idx]),
        )
        ranges = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, max(1e-6, float(diff_up[row_idx].max()))),
        )
        cmaps = ("gray", "gray", "gray", "inferno")
        for col_idx, ax in enumerate(axes[row_idx]):
            vmin, vmax = ranges[col_idx]
            ax.imshow(panels[col_idx], cmap=cmaps[col_idx], vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(titles[col_idx])
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(f"sample {row_idx + 1}", rotation=90, labelpad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_phase_preview(output_path: Path, snapshot: dict[str, torch.Tensor]) -> None:
    phi_lr = snapshot["phi_lr"][0]
    raw_phase = snapshot.get("raw_phase")

    column_count = 2 if raw_phase is not None else 1
    fig, axes = plt.subplots(1, column_count, figsize=(4.5 * column_count, 4))
    if column_count == 1:
        axes = [axes]

    phi_np = _to_gray_numpy(phi_lr)
    axes[0].imshow(phi_np, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[0].set_title(
        "phi_lr\n"
        f"min={float(phi_lr.min()):.3f}, max={float(phi_lr.max()):.3f}"
    )
    axes[0].axis("off")

    if raw_phase is not None:
        raw_phase_tensor = raw_phase[0]
        raw_np = _to_gray_numpy(raw_phase_tensor)
        raw_abs_max = max(1e-6, float(raw_phase_tensor.abs().max()))
        axes[1].imshow(raw_np, cmap="coolwarm", vmin=-raw_abs_max, vmax=raw_abs_max)
        axes[1].set_title(
            "raw_phase\n"
            f"min={float(raw_phase_tensor.min()):.3f}, max={float(raw_phase_tensor.max()):.3f}"
        )
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(output_path: Path, history: list[dict[str, Any]]) -> None:
    steps = [int(entry["step"]) for entry in history]
    train_losses = [float(entry["train_loss"]) for entry in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, train_losses, linewidth=1.5, label="train_loss")

    val_points = [
        (int(entry["step"]), float(entry["val_loss"]))
        for entry in history
        if entry.get("val_loss") is not None
    ]
    if val_points:
        val_steps = [point[0] for point in val_points]
        val_losses = [point[1] for point in val_points]
        ax.plot(val_steps, val_losses, linewidth=1.2, marker="o", markersize=3, label="val_loss")

    ax.set_title("Stage 4 Minimal Loss Curve")
    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized MAE")
    ax.grid(True, alpha=0.3)
    if val_points:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_json(output_path: Path, payload: Any) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def prepare_output_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"run_{timestamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = normalize_device(args.device)
    run_dir = prepare_output_dir(args)

    train_ds, val_ds, dataset_sizes = build_stage4_train_val_datasets(
        dataset_root=args.dataset_root,
        emnist_split=args.emnist_split,
        hr_hw=args.hr_size,
        num_samples=args.dataset_num_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=args.download,
        overfit_single_sample=args.single_sample,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=not args.single_sample,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model, target_adapter = build_stage4_model(args, device=device)
    optimizer = build_optimizer(model, args)

    config_snapshot = {
        "task": "stage4_minimal_closed_loop_trainer",
        "mode": args.mode,
        "single_sample": args.single_sample,
        "subset_size": args.subset_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "dataset": {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "emnist_split": args.emnist_split,
            "hr_size": args.hr_size,
            "selected_samples": args.dataset_num_samples,
            "train_samples": dataset_sizes["train"],
            "val_samples": dataset_sizes["val"],
            "val_ratio": args.val_ratio,
            "single_sample_protocol": args.single_sample,
        },
        "model": {
            "encoder": {
                "target_hw": [args.input_pattern_hw, args.input_pattern_hw],
                "base_channels": args.encoder_base_channels,
                "hidden_channels": args.encoder_hidden_channels,
                "phase_range": [-math.pi, math.pi],
            },
            "optics": {
                "depth": args.depth,
                "wavelength": args.wavelength,
                "pixel_pitch": args.pixel_pitch,
                "input_pattern_hw": [args.input_pattern_hw, args.input_pattern_hw],
                "layer_hw": [args.layer_hw, args.layer_hw],
                "propagation_hw": [args.propagation_hw, args.propagation_hw],
                "output_crop_hw": [args.output_crop_hw, args.output_crop_hw],
                "distance_schedule": build_distance_schedule(args.depth),
            },
            "freeze_encoder": args.freeze_encoder,
            "freeze_optics": args.freeze_optics,
        },
        "optimizer": {
            "name": "Adam",
            "lr_encoder": args.lr_encoder,
            "lr_optics": args.lr_optics,
        },
        "protocol_note": (
            "Stage 4 minimal closed-loop learnability check using real EMNIST HR input, "
            "explicit target_hr -> target_roi adaptation, and the current minimal optics grid. "
            "This is not the paper-final training protocol."
        ),
    }
    save_json(run_dir / "config_snapshot.json", config_snapshot)

    preview_batch = next(iter(train_loader))
    model.eval()
    with torch.no_grad():
        initial_preview_output = run_stage4_batch(
            model,
            preview_batch,
            target_adapter=target_adapter,
            device=device,
            return_encoder_debug=True,
        )
    initial_snapshot = snapshot_from_batch_output(
        initial_preview_output,
        preview_limit=args.preview_limit,
    )
    save_preview_grid(run_dir / "preview_step0.png", initial_snapshot)
    save_phase_preview(run_dir / "phi_preview_step0.png", initial_snapshot)
    initial_loss = float(initial_preview_output["loss"].item())

    initial_eval = evaluate_loader(
        model,
        val_loader,
        target_adapter=target_adapter,
        device=device,
    )

    history: list[dict[str, Any]] = [
        {
            "step": 0,
            "epoch": 0,
            "train_loss": initial_loss,
            "val_loss": float(initial_eval["val_loss"]),
            "val_psnr": float(initial_eval["val_psnr"]),
            "val_ssim": float(initial_eval["val_ssim"]),
        }
    ]
    grad_stats_history: list[dict[str, Any]] = []

    best_loss = initial_loss
    best_step = 0
    best_snapshot = initial_snapshot
    best_checkpoint_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    latest_checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    torch.save(
        {
            "step": 0,
            "epoch": 0,
            "best_loss": best_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_snapshot": config_snapshot,
        },
        best_checkpoint_path,
    )

    train_iterator = iter(train_loader)
    steps_per_epoch = max(1, len(train_loader))

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        epoch = ((step - 1) // steps_per_epoch) + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        batch_output = run_stage4_batch(
            model,
            batch,
            target_adapter=target_adapter,
            device=device,
            return_encoder_debug=False,
        )
        loss = batch_output["loss"]
        loss.backward()

        encoder_grad_stats = collect_module_grad_stats(model.encoder)
        optics_grad_stats = collect_module_grad_stats(model.decoder)
        grad_stats_entry = {
            "step": step,
            "epoch": epoch,
            "encoder": encoder_grad_stats,
            "optics": optics_grad_stats,
        }
        grad_stats_history.append(grad_stats_entry)

        optimizer.step()

        train_loss = float(loss.item())
        history_entry: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_sigma_mean": float(batch_output["sigma"].mean().item()),
            "val_loss": None,
            "val_psnr": None,
            "val_ssim": None,
        }

        if (step % args.val_every == 0) or (step == args.steps):
            val_metrics = evaluate_loader(
                model,
                val_loader,
                target_adapter=target_adapter,
                device=device,
            )
            history_entry.update(
                {
                    "val_loss": float(val_metrics["val_loss"]),
                    "val_psnr": float(val_metrics["val_psnr"]),
                    "val_ssim": float(val_metrics["val_ssim"]),
                }
            )

        history.append(history_entry)

        if train_loss < best_loss:
            best_loss = train_loss
            best_step = step
            model.eval()
            with torch.no_grad():
                best_preview_output = run_stage4_batch(
                    model,
                    preview_batch,
                    target_adapter=target_adapter,
                    device=device,
                    return_encoder_debug=True,
                )
            best_snapshot = snapshot_from_batch_output(
                best_preview_output,
                preview_limit=args.preview_limit,
            )
            torch.save(
                {
                    "step": step,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config_snapshot": config_snapshot,
                },
                best_checkpoint_path,
            )

        if step == 1 or step == args.steps or step % max(1, args.val_every) == 0:
            print(
                f"[step {step:04d}/{args.steps}] "
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"encoder_grad_nonzero={encoder_grad_stats['has_nonzero_grad']} "
                f"optics_grad_nonzero={optics_grad_stats['has_nonzero_grad']}"
            )

    model.eval()
    with torch.no_grad():
        final_preview_output = run_stage4_batch(
            model,
            preview_batch,
            target_adapter=target_adapter,
            device=device,
            return_encoder_debug=True,
        )
    final_snapshot = snapshot_from_batch_output(
        final_preview_output,
        preview_limit=args.preview_limit,
    )
    final_loss = float(final_preview_output["loss"].item())
    final_eval = evaluate_loader(
        model,
        val_loader,
        target_adapter=target_adapter,
        device=device,
    )

    save_preview_grid(run_dir / "preview_best.png", best_snapshot)
    save_preview_grid(run_dir / "preview_final.png", final_snapshot)
    save_phase_preview(run_dir / "phi_preview_best.png", best_snapshot)
    save_phase_preview(run_dir / "phi_preview_final.png", final_snapshot)
    save_loss_curve(run_dir / "loss_curve.png", history)

    torch.save(
        {
            "step": args.steps,
            "best_step": best_step,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_snapshot": config_snapshot,
        },
        latest_checkpoint_path,
    )

    save_json(run_dir / "history.json", history)
    save_json(run_dir / "grad_stats.json", grad_stats_history)

    encoder_grad_observed = any(
        entry["encoder"]["has_nonzero_grad"] for entry in grad_stats_history
    )
    optics_grad_observed = any(
        entry["optics"]["has_nonzero_grad"] for entry in grad_stats_history
    )

    run_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "stage4_minimal_closed_loop_trainer",
        "mode": args.mode,
        "single_sample": args.single_sample,
        "steps": args.steps,
        "device": str(device),
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": final_loss,
        "loss_decreased": best_loss < initial_loss,
        "dataset": dataset_sizes,
        "freeze_encoder": args.freeze_encoder,
        "freeze_optics": args.freeze_optics,
        "gradient_observability": {
            "encoder_grad_observed": encoder_grad_observed,
            "optics_grad_observed": optics_grad_observed,
            "optics_expected": not args.freeze_optics,
            "encoder_expected": not args.freeze_encoder,
        },
        "nan_inf_check": {
            "initial_preview_finite": True,
            "final_preview_finite": True,
        },
        "final_val_metrics": final_eval,
        "artifacts": {
            "config_snapshot": str((run_dir / "config_snapshot.json").resolve()),
            "history": str((run_dir / "history.json").resolve()),
            "run_summary": str((run_dir / "run_summary.json").resolve()),
            "loss_curve": str((run_dir / "loss_curve.png").resolve()),
            "preview_step0": str((run_dir / "preview_step0.png").resolve()),
            "preview_best": str((run_dir / "preview_best.png").resolve()),
            "preview_final": str((run_dir / "preview_final.png").resolve()),
            "phi_preview_step0": str((run_dir / "phi_preview_step0.png").resolve()),
            "phi_preview_best": str((run_dir / "phi_preview_best.png").resolve()),
            "phi_preview_final": str((run_dir / "phi_preview_final.png").resolve()),
            "checkpoint_best": str(best_checkpoint_path.resolve()),
            "checkpoint_latest": str(latest_checkpoint_path.resolve()),
            "grad_stats": str((run_dir / "grad_stats.json").resolve()),
        },
        "protocol_note": config_snapshot["protocol_note"],
    }
    save_json(run_dir / "run_summary.json", run_summary)

    return {
        "run_dir": run_dir,
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "final_loss": final_loss,
        "encoder_grad_observed": encoder_grad_observed,
        "optics_grad_observed": optics_grad_observed,
    }


def main() -> None:
    args = parse_args()
    args = resolve_runtime_args(args)
    result = train(args)
    print("Stage 4 minimal trainer run completed.")
    print(f"run_dir={result['run_dir']}")
    print(f"initial_loss={result['initial_loss']:.6f}")
    print(f"best_loss={result['best_loss']:.6f} at step={result['best_step']}")
    print(f"final_loss={result['final_loss']:.6f}")
    print(f"encoder_grad_observed={result['encoder_grad_observed']}")
    print(f"optics_grad_observed={result['optics_grad_observed']}")


if __name__ == "__main__":
    main()
