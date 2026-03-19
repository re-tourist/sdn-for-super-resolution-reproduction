#!/usr/bin/env python
"""Train the Stage 5 paper-aligned encoder + diffractive decoder pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import Stage4RoiTargetAdapter, build_stage5_emnist_display_dataset
from src.losses import Stage5SuperResolutionLoss
from src.models.encoders import PaperPhaseEncoder
from src.models.optics.diffractive_decoder import DiffractiveDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 5 paper-aligned super-resolution pipeline."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--preview-limit", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must load to a dict, got {type(data)!r}.")
    return data


def require_mapping(root: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = root
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing config path: {'/'.join(keys)}")
        current = current[key]
    if not isinstance(current, dict):
        raise TypeError(f"Config path {'/'.join(keys)} must be a mapping.")
    return current


def normalize_device(device_arg: str) -> torch.device:
    normalized = device_arg.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but it is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device value: {device_arg}.")
    return torch.device(normalized)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def format_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_optional_float(value: float | None, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def get_cuda_memory_summary(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return (
        f"cuda_alloc_mb={allocated_mb:.1f} "
        f"cuda_reserved_mb={reserved_mb:.1f} "
        f"cuda_max_alloc_mb={max_allocated_mb:.1f}"
    )


def compact_history_artifact(
    *,
    artifact_name: str,
    entries: list[dict[str, Any]] | None,
    entry_count: int,
    jsonl_path: Path,
    last_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    if entries is not None:
        return {
            "artifact_name": artifact_name,
            "storage_mode": "in_memory_full_list",
            "entry_count": entry_count,
            "entries": entries,
        }
    return {
        "artifact_name": artifact_name,
        "storage_mode": "jsonl_external",
        "entry_count": entry_count,
        "jsonl_path": str(jsonl_path.resolve()),
        "last_entry": last_entry,
        "note": (
            "Full per-step records are stored incrementally in JSONL to avoid "
            "unbounded checkpoint growth during long Stage 5 runs."
        ),
    }


def build_status_payload(
    *,
    phase: str,
    mode: str,
    run_name: str,
    depth: int,
    device: torch.device,
    current_step: int,
    total_steps: int,
    current_epoch: int | None,
    steps_per_epoch: int | None,
    validate_every: int | None,
    log_every: int | None,
    checkpoint_every: int | None,
    resume_from: str | None,
    started_at_utc: str,
    output_root: Path,
    train_log_path: Path,
    history_jsonl_path: Path,
    grad_jsonl_path: Path,
    best_val_loss: float | None,
    best_step: int | None,
    last_train_metrics: dict[str, Any] | None,
    last_val_metrics: dict[str, Any] | None,
    note: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": phase,
        "mode": mode,
        "run_name": run_name,
        "depth": depth,
        "device": str(device),
        "resume_from": resume_from,
        "runtime": {
            "current_step": current_step,
            "total_steps": total_steps,
            "current_epoch": current_epoch,
            "steps_per_epoch": steps_per_epoch,
            "validate_every": validate_every,
            "log_every": log_every,
            "checkpoint_every": checkpoint_every,
        },
        "best": {
            "best_val_loss": best_val_loss,
            "best_step": best_step,
        },
        "last_metrics": {
            "train": last_train_metrics,
            "val": last_val_metrics,
        },
        "artifacts": {
            "output_root": str(output_root.resolve()),
            "train_log": str(train_log_path.resolve()),
            "history_jsonl": str(history_jsonl_path.resolve()),
            "grad_stats_jsonl": str(grad_jsonl_path.resolve()),
        },
        "timestamps": {
            "started_at_utc": started_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    if note is not None:
        payload["note"] = note
    if error is not None:
        payload["error"] = error
    return payload


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def _non_batch_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(1, tensor.ndim))


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf.")


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


def build_dataset(
    dataset_cfg: dict[str, Any],
    *,
    split: str,
    download: bool,
) -> Any:
    return build_stage5_emnist_display_dataset(
        split=split,
        dataset_root=dataset_cfg["dataset_root"],
        emnist_split=dataset_cfg["emnist_split"],
        sample_count=int(dataset_cfg["sample_counts"][split]),
        letter_count_choices=list(dataset_cfg["letter_count_choices"][split]),
        seed=int(dataset_cfg["seed"]),
        canvas_hw=tuple(dataset_cfg["canvas_hw"]),
        cell_hw=tuple(dataset_cfg["cell_hw"]),
        grid_shape=tuple(dataset_cfg["grid_shape"]),
        augmentation=dataset_cfg["augmentation"][split],
        fix_emnist_orientation=bool(dataset_cfg.get("fix_emnist_orientation", True)),
        download=download,
    )


def maybe_subset(dataset: Any, subset_size: int | None) -> Any:
    if subset_size is None:
        return dataset
    subset_size = int(subset_size)
    if subset_size < 1:
        raise ValueError("Subset size must be >= 1.")
    if subset_size > len(dataset):
        raise ValueError(f"Subset size {subset_size} exceeds dataset length {len(dataset)}.")
    return Subset(dataset, list(range(subset_size)))


def build_encoder_decoder(
    encoder_cfg: dict[str, Any],
    optics_cfg: dict[str, Any],
    *,
    depth: int,
    device: torch.device,
) -> tuple[PaperPhaseEncoder, DiffractiveDecoder]:
    input_hw = tuple(encoder_cfg["target_hw"])
    optics_input_hw = tuple(optics_cfg["grid"]["input_pattern_hw"])
    if input_hw != optics_input_hw:
        raise ValueError(
            "Stage 5 encoder target_hw must match optics input_pattern_hw, "
            f"got {input_hw} vs {optics_input_hw}."
        )

    encoder = PaperPhaseEncoder(
        in_channels=int(encoder_cfg["in_channels"]),
        input_hw=tuple(encoder_cfg["input_hw"]),
        target_hw=input_hw,
        base_channels=int(encoder_cfg["base_channels"]),
        hidden_channels=int(encoder_cfg["hidden_channels"]),
        interpolation_mode=str(encoder_cfg["interpolation_mode"]),
        align_corners=bool(encoder_cfg["align_corners"]),
        phase_mapping=str(encoder_cfg["phase_mapping"]),
        phase_range=tuple(float(value) for value in encoder_cfg["phase_range"]),
    ).to(device)

    depth_cfg = optics_cfg["depths"][str(depth)]
    decoder = DiffractiveDecoder(
        num_diffractive_layers=int(depth_cfg["num_diffractive_layers"]),
        wavelength=float(optics_cfg["units"]["wavelength"]),
        pixel_pitch=float(optics_cfg["units"]["pixel_pitch_lambda"]),
        grid_config=optics_cfg["grid"],
        distance_schedule=depth_cfg["distance_schedule"],
        readout_config={"output_crop_hw": optics_cfg["readout"]["output_crop_hw"]},
        phase_mask_config=dict(optics_cfg.get("phase_mask_config", {})),
    ).to(device)
    return encoder, decoder


def build_loss(loss_cfg: dict[str, Any], *, device: torch.device) -> Stage5SuperResolutionLoss:
    efficiency_cfg = loss_cfg["efficiency_term"]
    return Stage5SuperResolutionLoss(
        epsilon=float(loss_cfg["epsilon"]),
        sigma_mode=str(loss_cfg["sigma_mode"]),
        gamma_by_depth=efficiency_cfg["gamma_by_depth"],
        enable_efficiency_term=bool(efficiency_cfg["enabled"]),
        eta_scale=float(efficiency_cfg["eta_scale"]),
    ).to(device)


def build_optimizer(
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    trainer_cfg: dict[str, Any],
) -> Adam:
    encoder_params = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    decoder_params = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    if not encoder_params or not decoder_params:
        raise RuntimeError("Stage 5 trainer expects both encoder and decoder to be trainable.")
    return Adam(
        [
            {"params": encoder_params, "lr": float(trainer_cfg["optimizer"]["lr_encoder"])},
            {"params": decoder_params, "lr": float(trainer_cfg["optimizer"]["lr_decoder"])},
        ]
    )


def build_train_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


def compute_input_power_from_u0(U0: torch.Tensor) -> torch.Tensor:
    intensity = U0.real.square() + U0.imag.square()
    return intensity.sum(dim=_non_batch_dims(intensity))


def run_stage5_batch(
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    loss_fn: Stage5SuperResolutionLoss,
    batch: dict[str, Any],
    *,
    depth: int,
    target_adapter: Stage4RoiTargetAdapter,
    device: torch.device,
) -> dict[str, Any]:
    moved_batch = move_batch_to_device(batch, device)
    x_hr = moved_batch["x_hr"]
    target_hr = moved_batch["target_hr"]
    if not isinstance(x_hr, torch.Tensor) or not isinstance(target_hr, torch.Tensor):
        raise TypeError("Batch must contain tensor fields 'x_hr' and 'target_hr'.")

    raw_phase = encoder.encode_raw_phase(x_hr)
    phi_lr = encoder.map_raw_phase(raw_phase)
    optical_output = decoder.forward_from_phase(phi_lr, return_intermediates=False)
    target_roi = target_adapter(target_hr)
    prediction_roi = optical_output["I_out_roi"]
    U0 = optical_output["U0"]
    if not isinstance(prediction_roi, torch.Tensor) or not isinstance(U0, torch.Tensor):
        raise TypeError("Decoder output is missing tensor fields 'I_out_roi' or 'U0'.")

    input_power = compute_input_power_from_u0(U0)
    loss_dict = loss_fn(
        prediction_roi,
        target_roi,
        depth=depth,
        input_power=input_power,
    )

    _assert_finite("x_hr", x_hr)
    _assert_finite("target_roi", target_roi)
    _assert_finite("raw_phase", raw_phase)
    _assert_finite("phi_lr", phi_lr)
    _assert_finite("I_out_roi", prediction_roi)
    _assert_finite("loss", loss_dict["loss"].reshape(1))

    return {
        "x_hr": x_hr,
        "target_hr": target_hr,
        "target_roi": target_roi,
        "raw_phase": raw_phase,
        "phi_lr": phi_lr,
        "output": optical_output,
        "pred_roi": prediction_roi,
        "loss_dict": loss_dict,
        "input_power": input_power,
    }


def summarize_batch_metrics(batch_output: dict[str, Any]) -> dict[str, float | None]:
    loss_dict = batch_output["loss_dict"]
    eta = loss_dict["eta"]
    output_power = loss_dict["output_power"]
    return {
        "loss": float(loss_dict["loss"].item()),
        "mae_term": float(loss_dict["mae_term"].item()),
        "efficiency_term": float(loss_dict["efficiency_term"].item()),
        "sigma_mean": float(loss_dict["sigma"].mean().item()),
        "eta_mean": None if eta is None else float(eta.mean().item()),
        "input_power_mean": float(batch_output["input_power"].mean().item()),
        "output_power_mean": (
            None if output_power is None else float(output_power.mean().item())
        ),
    }


@torch.no_grad()
def evaluate_loader(
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    loss_fn: Stage5SuperResolutionLoss,
    loader: DataLoader,
    *,
    depth: int,
    target_adapter: Stage4RoiTargetAdapter,
    device: torch.device,
) -> dict[str, float | None]:
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_efficiency = 0.0
    total_sigma = 0.0
    total_eta = 0.0
    total_input_power = 0.0
    total_output_power = 0.0
    eta_count = 0
    output_power_count = 0
    sample_count = 0

    for batch in loader:
        batch_output = run_stage5_batch(
            encoder,
            decoder,
            loss_fn,
            batch,
            depth=depth,
            target_adapter=target_adapter,
            device=device,
        )
        metrics = summarize_batch_metrics(batch_output)
        batch_size = batch_output["x_hr"].shape[0]
        total_loss += metrics["loss"] * batch_size
        total_mae += metrics["mae_term"] * batch_size
        total_efficiency += metrics["efficiency_term"] * batch_size
        total_sigma += metrics["sigma_mean"] * batch_size
        total_input_power += metrics["input_power_mean"] * batch_size
        if metrics["eta_mean"] is not None:
            total_eta += float(metrics["eta_mean"]) * batch_size
            eta_count += batch_size
        if metrics["output_power_mean"] is not None:
            total_output_power += float(metrics["output_power_mean"]) * batch_size
            output_power_count += batch_size
        sample_count += batch_size

    if sample_count < 1:
        raise RuntimeError("Validation loader is empty.")

    return {
        "val_loss": total_loss / sample_count,
        "val_mae_term": total_mae / sample_count,
        "val_efficiency_term": total_efficiency / sample_count,
        "val_sigma_mean": total_sigma / sample_count,
        "val_eta_mean": None if eta_count == 0 else total_eta / eta_count,
        "val_input_power_mean": total_input_power / sample_count,
        "val_output_power_mean": (
            None if output_power_count == 0 else total_output_power / output_power_count
        ),
    }


def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 2:
        return tensor.numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].numpy()
    raise ValueError(f"Expected grayscale tensor, got shape {tuple(tensor.shape)}.")


def save_preview_grid(output_path: Path, batch_output: dict[str, Any], *, preview_limit: int) -> None:
    row_count = min(preview_limit, batch_output["x_hr"].shape[0])
    x_hr = batch_output["x_hr"][:row_count].detach().cpu()
    target_roi = batch_output["target_roi"][:row_count].detach().cpu()
    pred_roi = batch_output["pred_roi"][:row_count].detach().cpu()
    abs_error = torch.abs(target_roi - pred_roi)

    fig, axes = plt.subplots(row_count, 4, figsize=(12, max(3.0, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ("x_hr", "target_roi", "pred_roi", "abs_error")
    for row_index in range(row_count):
        panels = (
            _to_gray_numpy(x_hr[row_index]),
            _to_gray_numpy(target_roi[row_index]),
            _to_gray_numpy(pred_roi[row_index]),
            _to_gray_numpy(abs_error[row_index]),
        )
        ranges = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, float(abs_error.max().item()) + 1e-8),
        )
        for col_index, axis in enumerate(axes[row_index]):
            vmin, vmax = ranges[col_index]
            axis.imshow(panels[col_index], cmap="gray", vmin=vmin, vmax=vmax)
            if row_index == 0:
                axis.set_title(titles[col_index])
            axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_phase_preview(output_path: Path, batch_output: dict[str, Any], *, preview_limit: int) -> None:
    row_count = min(preview_limit, batch_output["phi_lr"].shape[0])
    raw_phase = batch_output["raw_phase"][:row_count].detach().cpu()
    phi_lr = batch_output["phi_lr"][:row_count].detach().cpu()

    fig, axes = plt.subplots(row_count, 2, figsize=(6, max(3.0, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index in range(row_count):
        panels = (
            _to_gray_numpy(raw_phase[row_index]),
            _to_gray_numpy(phi_lr[row_index]),
        )
        ranges = (
            (float(raw_phase.min().item()), float(raw_phase.max().item())),
            (float(phi_lr.min().item()), float(phi_lr.max().item())),
        )
        for col_index, axis in enumerate(axes[row_index]):
            vmin, vmax = ranges[col_index]
            axis.imshow(panels[col_index], cmap="viridis", vmin=vmin, vmax=vmax)
            if row_index == 0:
                axis.set_title(("raw_phase", "phi_lr")[col_index])
            axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(output_path: Path, history: list[dict[str, Any]]) -> None:
    train_steps = [entry["step"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_points = [
        (entry["step"], entry["val_loss"]) for entry in history if entry["val_loss"] is not None
    ]

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(train_steps, train_loss, label="train_loss", color="tab:blue", marker="o")
    if val_points:
        axis.plot(
            [item[0] for item in val_points],
            [item[1] for item in val_points],
            label="val_loss",
            color="tab:orange",
            marker="s",
        )
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.set_title("Stage 5 Trainer Loss History")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    steps_per_epoch: int,
    best_val_loss: float,
    best_step: int | None,
    history: list[dict[str, Any]] | None,
    grad_history: list[dict[str, Any]] | None,
    history_entry_count: int,
    grad_history_entry_count: int,
    last_history_entry: dict[str, Any] | None,
    last_grad_history_entry: dict[str, Any] | None,
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    optimizer: Adam,
    config_snapshot: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "steps_per_epoch": steps_per_epoch,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "history_entry_count": history_entry_count,
        "grad_history_entry_count": grad_history_entry_count,
        "last_history_entry": last_history_entry,
        "last_grad_history_entry": last_grad_history_entry,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config_snapshot": config_snapshot,
    }
    if history is not None:
        payload["history"] = history
    if grad_history is not None:
        payload["grad_stats_history"] = grad_history
    torch.save(payload, path)


def split_history_entry_metrics(
    history_entry: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if history_entry is None:
        return None, None
    train_metrics = {
        key: value for key, value in history_entry.items() if key.startswith("train_")
    }
    val_metrics = {
        key: value
        for key, value in history_entry.items()
        if key.startswith("val_") and value is not None
    }
    return train_metrics or None, val_metrics or None


def main() -> None:
    args = parse_args()
    trainer_root = load_yaml(args.config)
    trainer_cfg = require_mapping(trainer_root, "trainer", "stage5_paper")
    runtime_cfg = dict(trainer_cfg["runtime"])
    total_steps = int(args.steps if args.steps is not None else runtime_cfg["steps"])
    if total_steps < 1:
        raise ValueError("Total steps must be >= 1.")
    log_every = int(args.log_every if args.log_every is not None else runtime_cfg.get("log_every", 1))
    if log_every < 1:
        raise ValueError("log_every must be >= 1.")
    checkpoint_every = int(runtime_cfg.get("checkpoint_every", 1))
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1.")
    keep_history_in_memory = bool(runtime_cfg.get("keep_history_in_memory", True))
    preview_limit = int(
        args.preview_limit if args.preview_limit is not None else runtime_cfg["preview_limit"]
    )
    if preview_limit < 1:
        raise ValueError("preview_limit must be >= 1.")

    device = normalize_device(str(args.device or runtime_cfg["device"]))
    seed = int(runtime_cfg["seed"])
    set_seed(seed)
    depth = int(trainer_cfg["model"]["depth"])

    output_root = Path(trainer_cfg["output_root"])
    if args.resume:
        checkpoint_path = Path(args.resume)
        run_dir = checkpoint_path.parent.parent
        run_name = run_dir.name
    else:
        run_name = args.run_name or trainer_cfg.get("run_name") or datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        run_dir = output_root / run_name

    download = bool(args.download or runtime_cfg.get("download", False))
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = run_dir / "train.log"
    history_jsonl_path = run_dir / "history.jsonl"
    grad_jsonl_path = run_dir / "grad_stats.jsonl"
    status_path = run_dir / "status.json"
    failure_path = run_dir / "failure.json"
    loss_curve_points_path = run_dir / "loss_curve_points.json"
    latest_checkpoint_path = run_dir / "checkpoints" / "checkpoint_latest.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "checkpoint_best.pt"
    started_at_utc = datetime.now(timezone.utc).isoformat()
    logger = RunLogger(train_log_path)
    pin_memory = device.type == "cuda"
    resume_from = None if not args.resume else str(Path(args.resume).resolve())

    dataset_cfg: dict[str, Any] | None = None
    optics_cfg: dict[str, Any] | None = None
    encoder_cfg: dict[str, Any] | None = None
    loss_cfg: dict[str, Any] | None = None
    encoder: PaperPhaseEncoder | None = None
    decoder: DiffractiveDecoder | None = None
    loss_fn: Stage5SuperResolutionLoss | None = None
    optimizer: Adam | None = None
    target_adapter: Stage4RoiTargetAdapter | None = None
    preview_batch: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = [] if keep_history_in_memory else None
    grad_history: list[dict[str, Any]] | None = [] if keep_history_in_memory else None
    curve_history: list[dict[str, Any]] = []
    last_history_entry: dict[str, Any] | None = None
    last_grad_history_entry: dict[str, Any] | None = None
    last_train_metrics: dict[str, Any] | None = None
    last_val_metrics: dict[str, Any] | None = None
    best_val_loss = math.inf
    best_step: int | None = None
    history_entry_count = 0
    grad_history_entry_count = 0
    start_step = 0
    current_epoch = 0
    steps_per_epoch: int | None = None
    validate_every: int | None = None
    global_step = 0
    config_snapshot: dict[str, Any] | None = None

    def write_status(phase: str, note: str, error: dict[str, Any] | None = None) -> None:
        save_json(
            status_path,
            build_status_payload(
                phase=phase,
                mode=trainer_cfg["mode"],
                run_name=run_name,
                depth=depth,
                device=device,
                current_step=global_step,
                total_steps=total_steps,
                current_epoch=current_epoch if steps_per_epoch is not None else None,
                steps_per_epoch=steps_per_epoch,
                validate_every=validate_every,
                log_every=log_every,
                checkpoint_every=checkpoint_every,
                resume_from=resume_from,
                started_at_utc=started_at_utc,
                output_root=run_dir,
                train_log_path=train_log_path,
                history_jsonl_path=history_jsonl_path,
                grad_jsonl_path=grad_jsonl_path,
                best_val_loss=None if best_step is None else best_val_loss,
                best_step=best_step,
                last_train_metrics=last_train_metrics,
                last_val_metrics=last_val_metrics,
                note=note,
                error=error,
            ),
        )

    try:
        logger.log("Stage 5 paper trainer bootstrapping.")
        logger.log(
            f"Config={Path(args.config).resolve()} mode={trainer_cfg['mode']} "
            f"run_name={run_name} depth={depth} device={device} resume_from={resume_from or 'none'}"
        )
        logger.log(
            f"Runtime budget: steps={total_steps} batch_size={runtime_cfg['batch_size']} "
            f"validate_every={runtime_cfg['validate_every']} log_every={log_every} "
            f"checkpoint_every={checkpoint_every} preview_limit={preview_limit} seed={seed}"
        )
        logger.log(f"Output root={output_root.resolve()} run_dir={run_dir.resolve()}")
        if download:
            logger.log(
                "Dataset download is enabled. If EMNIST is missing, torchvision may spend "
                "time downloading/extracting before the next trainer log."
            )
        write_status("startup", "Trainer started; dataset/model construction has not finished yet.")

        dataset_cfg = require_mapping(
            load_yaml(trainer_cfg["config_paths"]["dataset"]), "data", "stage5_display"
        )
        optics_cfg = require_mapping(
            load_yaml(trainer_cfg["config_paths"]["optics"]), "optics", "stage5_paper_aligned"
        )
        encoder_cfg = require_mapping(
            load_yaml(trainer_cfg["config_paths"]["encoder"]), "encoder", "stage5_paper_aligned"
        )
        loss_cfg = require_mapping(
            load_yaml(trainer_cfg["config_paths"]["loss"]), "loss", "stage5_sr"
        )

        logger.log(
            f"Building train dataset split from root={dataset_cfg['dataset_root']} download={download}."
        )
        write_status("building_train_dataset", "Building Stage 5 train dataset.")
        train_base = build_dataset(dataset_cfg, split="train", download=download)
        logger.log(f"Train dataset ready: base_len={len(train_base)}")

        logger.log("Building val dataset split.")
        write_status("building_val_dataset", "Building Stage 5 val dataset.")
        val_base = build_dataset(dataset_cfg, split="val", download=download)
        logger.log(f"Val dataset ready: base_len={len(val_base)}")

        train_dataset = maybe_subset(
            train_base, trainer_cfg.get("dataset_selection", {}).get("train_subset_size")
        )
        val_dataset = maybe_subset(
            val_base, trainer_cfg.get("dataset_selection", {}).get("val_subset_size")
        )
        logger.log(
            f"Dataset selection resolved: train_len={len(train_dataset)} val_len={len(val_dataset)} "
            f"train_subset={trainer_cfg.get('dataset_selection', {}).get('train_subset_size')} "
            f"val_subset={trainer_cfg.get('dataset_selection', {}).get('val_subset_size')}"
        )

        encoder, decoder = build_encoder_decoder(
            encoder_cfg,
            optics_cfg,
            depth=depth,
            device=device,
        )
        target_adapter = Stage4RoiTargetAdapter(output_crop_hw=decoder.readout_config)
        loss_fn = build_loss(loss_cfg, device=device)
        optimizer = build_optimizer(encoder, decoder, trainer_cfg)

        batch_size = int(runtime_cfg["batch_size"])
        num_workers = int(runtime_cfg["num_workers"])
        train_shuffle = bool(runtime_cfg["train_shuffle"])
        steps_per_epoch = math.ceil(len(train_dataset) / batch_size)
        validate_every = int(runtime_cfg["validate_every"])
        logger.log(
            f"Model+loss ready: encoder_target_hw={tuple(encoder_cfg['target_hw'])} "
            f"optics_input_pattern_hw={tuple(optics_cfg['grid']['input_pattern_hw'])} "
            f"output_crop_hw={tuple(optics_cfg['readout']['output_crop_hw'])}"
        )
        logger.log(
            f"Optimizer groups ready: encoder_lr={float(trainer_cfg['optimizer']['lr_encoder'])} "
            f"decoder_lr={float(trainer_cfg['optimizer']['lr_decoder'])} "
            f"pin_memory={pin_memory} num_workers={num_workers}"
        )

        config_snapshot = {
            "trainer_config_path": str(Path(args.config).resolve()),
            "mode": trainer_cfg["mode"],
            "protocol_note": trainer_cfg["protocol_note"],
            "run_name": run_name,
            "depth": depth,
            "device": str(device),
            "runtime": {
                **runtime_cfg,
                "steps": total_steps,
                "preview_limit": preview_limit,
                "download": download,
                "log_every": log_every,
                "checkpoint_every": checkpoint_every,
                "keep_history_in_memory": keep_history_in_memory,
            },
            "optimizer": trainer_cfg["optimizer"],
            "optimizer_parameter_groups": [
                {"name": "encoder", "lr": float(trainer_cfg["optimizer"]["lr_encoder"] )},
                {"name": "decoder", "lr": float(trainer_cfg["optimizer"]["lr_decoder"] )},
            ],
            "loss_wiring": {
                **trainer_cfg["loss_wiring"],
                "implemented_formula": "input_power = sum(|U0|^2) over full propagation grid per sample",
                "efficiency_enabled": bool(loss_cfg["efficiency_term"]["enabled"]),
            },
            "dataset_protocol": {
                "train": train_base.get_protocol_summary(),
                "val": val_base.get_protocol_summary(),
                "train_subset_size": len(train_dataset),
                "val_subset_size": len(val_dataset),
                "subset_note": (
                    "Subset sizes are trainer budget choices only; "
                    "they do not redefine the Stage 5 dataset protocol."
                ),
            },
            "artifact_policy": {
                "history_storage_mode": (
                    "in_memory_full_list" if keep_history_in_memory else "jsonl_incremental"
                ),
                "history_jsonl": str(history_jsonl_path.resolve()),
                "grad_stats_jsonl": str(grad_jsonl_path.resolve()),
                "status_json": str(status_path.resolve()),
                "train_log": str(train_log_path.resolve()),
                "loss_curve_points": str(loss_curve_points_path.resolve()),
            },
            "optics_config": optics_cfg,
            "encoder_config": encoder_cfg,
            "loss_config": loss_cfg,
            "resume_from": resume_from,
            "created_at_utc": started_at_utc,
        }

        if args.resume:
            logger.log(f"Loading checkpoint from {resume_from}.")
            checkpoint = torch.load(args.resume, map_location=device)
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            decoder.load_state_dict(checkpoint["decoder_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = int(checkpoint["step"])
            global_step = start_step
            best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
            best_step = checkpoint.get("best_step")
            if history is not None and "history" in checkpoint:
                history = list(checkpoint["history"])
            if grad_history is not None and "grad_stats_history" in checkpoint:
                grad_history = list(checkpoint["grad_stats_history"])
            history_entry_count = int(
                checkpoint.get("history_entry_count", 0 if history is None else len(history))
            )
            grad_history_entry_count = int(
                checkpoint.get(
                    "grad_history_entry_count", 0 if grad_history is None else len(grad_history)
                )
            )
            last_history_entry = checkpoint.get("last_history_entry")
            last_grad_history_entry = checkpoint.get("last_grad_history_entry")
            last_train_metrics, last_val_metrics = split_history_entry_metrics(last_history_entry)
            if loss_curve_points_path.exists():
                curve_history = list(load_json(loss_curve_points_path))
            logger.log(
                f"Resume state restored: start_step={start_step} best_step={best_step} "
                f"best_val_loss={format_optional_float(None if best_step is None else best_val_loss)} "
                f"history_entry_count={history_entry_count} "
                f"grad_history_entry_count={grad_history_entry_count}"
            )

        save_json(run_dir / "config_snapshot.json", config_snapshot)
        logger.log(f"Config snapshot saved to {(run_dir / 'config_snapshot.json').resolve()}.")

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        preview_batch = next(iter(val_loader))
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            preview_output = run_stage5_batch(
                encoder,
                decoder,
                loss_fn,
                preview_batch,
                depth=depth,
                target_adapter=target_adapter,
                device=device,
            )
        if args.resume:
            save_preview_grid(
                run_dir / "preview_resume_start.png",
                preview_output,
                preview_limit=preview_limit,
            )
            save_phase_preview(
                run_dir / "phi_preview_resume_start.png",
                preview_output,
                preview_limit=preview_limit,
            )
        else:
            save_preview_grid(run_dir / "preview_step0.png", preview_output, preview_limit=preview_limit)
            save_phase_preview(
                run_dir / "phi_preview_step0.png",
                preview_output,
                preview_limit=preview_limit,
            )
        logger.log("Initial preview artifacts saved.")

        current_epoch = start_step // steps_per_epoch
        step_in_epoch = start_step % steps_per_epoch
        train_loader = build_train_loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=train_shuffle,
            seed=seed,
            epoch=current_epoch,
        )
        train_iterator = iter(train_loader)
        for _ in range(step_in_epoch):
            next(train_iterator)

        write_status("training", "Training loop entered.")
        logger.log(
            f"Training loop entered: start_step={start_step} steps_per_epoch={steps_per_epoch} "
            f"validate_every={validate_every}"
        )

        train_loop_start = time.perf_counter()
        while global_step < total_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                current_epoch += 1
                train_loader = build_train_loader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=train_shuffle,
                    seed=seed,
                    epoch=current_epoch,
                )
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            step_start = time.perf_counter()
            encoder.train()
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            batch_output = run_stage5_batch(
                encoder,
                decoder,
                loss_fn,
                batch,
                depth=depth,
                target_adapter=target_adapter,
                device=device,
            )
            loss = batch_output["loss_dict"]["loss"]
            loss.backward()

            encoder_grad = collect_module_grad_stats(encoder)
            decoder_grad = collect_module_grad_stats(decoder)
            optimizer.step()
            global_step += 1

            train_metrics = summarize_batch_metrics(batch_output)
            history_entry = {
                "step": global_step,
                "epoch": current_epoch,
                "train_loss": train_metrics["loss"],
                "train_mae_term": train_metrics["mae_term"],
                "train_efficiency_term": train_metrics["efficiency_term"],
                "train_sigma_mean": train_metrics["sigma_mean"],
                "train_eta_mean": train_metrics["eta_mean"],
                "train_input_power_mean": train_metrics["input_power_mean"],
                "train_output_power_mean": train_metrics["output_power_mean"],
                "val_loss": None,
                "val_mae_term": None,
                "val_efficiency_term": None,
                "val_sigma_mean": None,
                "val_eta_mean": None,
            }
            grad_entry = {
                "step": global_step,
                "encoder": encoder_grad,
                "decoder": decoder_grad,
            }
            if history is not None:
                history.append(history_entry)
            if grad_history is not None:
                grad_history.append(grad_entry)
            history_entry_count += 1
            grad_history_entry_count += 1
            last_history_entry = dict(history_entry)
            last_grad_history_entry = grad_entry
            last_train_metrics, _ = split_history_entry_metrics(last_history_entry)
            append_jsonl(grad_jsonl_path, grad_entry)

            should_validate = global_step % validate_every == 0 or global_step == total_steps
            if should_validate:
                logger.log(
                    f"[validate:start] step={global_step}/{total_steps} val_samples={len(val_dataset)}"
                )
                write_status("validating", "Running validation.")
                val_metrics = evaluate_loader(
                    encoder,
                    decoder,
                    loss_fn,
                    val_loader,
                    depth=depth,
                    target_adapter=target_adapter,
                    device=device,
                )
                history_entry.update(val_metrics)
                last_history_entry = dict(history_entry)
                last_train_metrics, last_val_metrics = split_history_entry_metrics(last_history_entry)
                logger.log(
                    f"[validate:done] step={global_step}/{total_steps} "
                    f"val_loss={float(val_metrics['val_loss']):.6f} "
                    f"val_mae={float(val_metrics['val_mae_term']):.6f} "
                    f"val_eff={float(val_metrics['val_efficiency_term']):.6f} "
                    f"val_sigma={float(val_metrics['val_sigma_mean']):.6f} "
                    f"val_eta={format_optional_float(val_metrics['val_eta_mean'])}"
                )
                if float(val_metrics["val_loss"]) < best_val_loss:
                    best_val_loss = float(val_metrics["val_loss"])
                    best_step = global_step
                    with torch.no_grad():
                        best_preview = run_stage5_batch(
                            encoder,
                            decoder,
                            loss_fn,
                            preview_batch,
                            depth=depth,
                            target_adapter=target_adapter,
                            device=device,
                        )
                    save_preview_grid(
                        run_dir / "preview_best.png",
                        best_preview,
                        preview_limit=preview_limit,
                    )
                    save_phase_preview(
                        run_dir / "phi_preview_best.png",
                        best_preview,
                        preview_limit=preview_limit,
                    )
                    save_checkpoint(
                        best_checkpoint_path,
                        step=global_step,
                        steps_per_epoch=steps_per_epoch,
                        best_val_loss=best_val_loss,
                        best_step=best_step,
                        history=history,
                        grad_history=grad_history,
                        history_entry_count=history_entry_count,
                        grad_history_entry_count=grad_history_entry_count,
                        last_history_entry=last_history_entry,
                        last_grad_history_entry=last_grad_history_entry,
                        encoder=encoder,
                        decoder=decoder,
                        optimizer=optimizer,
                        config_snapshot=config_snapshot,
                    )
                    logger.log(
                        f"[checkpoint:best] step={global_step}/{total_steps} "
                        f"best_val_loss={best_val_loss:.6f} path={best_checkpoint_path.resolve()}"
                    )

            append_jsonl(history_jsonl_path, history_entry)

            step_elapsed = time.perf_counter() - step_start
            run_elapsed = time.perf_counter() - train_loop_start
            completed_run_steps = max(1, global_step - start_step)
            steps_per_second = completed_run_steps / run_elapsed if run_elapsed > 0 else None
            eta_seconds = (
                None
                if steps_per_second is None or steps_per_second <= 0
                else (total_steps - global_step) / steps_per_second
            )
            should_log = global_step == 1 or global_step % log_every == 0 or should_validate
            should_save_latest = (
                global_step % checkpoint_every == 0 or should_validate or global_step == total_steps
            )

            if should_log:
                curve_history.append(dict(history_entry))
                save_json(loss_curve_points_path, curve_history)

            if should_save_latest:
                save_checkpoint(
                    latest_checkpoint_path,
                    step=global_step,
                    steps_per_epoch=steps_per_epoch,
                    best_val_loss=best_val_loss,
                    best_step=best_step,
                    history=history,
                    grad_history=grad_history,
                    history_entry_count=history_entry_count,
                    grad_history_entry_count=grad_history_entry_count,
                    last_history_entry=last_history_entry,
                    last_grad_history_entry=last_grad_history_entry,
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    config_snapshot=config_snapshot,
                )
                if should_log:
                    logger.log(
                        f"[checkpoint:latest] step={global_step}/{total_steps} "
                        f"path={latest_checkpoint_path.resolve()}"
                    )

            if should_log:
                logger.log(
                    f"[train] step={global_step}/{total_steps} epoch={current_epoch} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_mae={train_metrics['mae_term']:.6f} "
                    f"train_eff={train_metrics['efficiency_term']:.6f} "
                    f"last_val_loss={format_optional_float(None if last_val_metrics is None else last_val_metrics.get('val_loss'))} "
                    f"best_val_loss={format_optional_float(None if best_step is None else best_val_loss)} "
                    f"encoder_grad_l2={encoder_grad['l2_grad_norm']:.6f} "
                    f"decoder_grad_l2={decoder_grad['l2_grad_norm']:.6f} "
                    f"step_time_s={step_elapsed:.3f} "
                    f"steps_per_sec={format_optional_float(steps_per_second, precision=3)} "
                    f"eta={format_seconds(eta_seconds)} {get_cuda_memory_summary(device)}"
                )
                write_status("training", "Run in progress.")

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            final_preview = run_stage5_batch(
                encoder,
                decoder,
                loss_fn,
                preview_batch,
                depth=depth,
                target_adapter=target_adapter,
                device=device,
            )
        save_preview_grid(run_dir / "preview_final.png", final_preview, preview_limit=preview_limit)
        save_phase_preview(
            run_dir / "phi_preview_final.png",
            final_preview,
            preview_limit=preview_limit,
        )

        if history is not None:
            save_json(run_dir / "history.json", history)
        else:
            save_json(
                run_dir / "history.json",
                compact_history_artifact(
                    artifact_name="history",
                    entries=None,
                    entry_count=history_entry_count,
                    jsonl_path=history_jsonl_path,
                    last_entry=last_history_entry,
                ),
            )
        if grad_history is not None:
            save_json(run_dir / "grad_stats.json", grad_history)
        else:
            save_json(
                run_dir / "grad_stats.json",
                compact_history_artifact(
                    artifact_name="grad_stats",
                    entries=None,
                    entry_count=grad_history_entry_count,
                    jsonl_path=grad_jsonl_path,
                    last_entry=last_grad_history_entry,
                ),
            )
        save_loss_curve(run_dir / "loss_curve.png", curve_history)

        run_summary = {
            "mode": trainer_cfg["mode"],
            "run_name": run_name,
            "completed_steps": global_step,
            "steps_per_epoch": steps_per_epoch,
            "depth": depth,
            "device": str(device),
            "optimizer_parameter_groups": config_snapshot["optimizer_parameter_groups"],
            "input_power_source": config_snapshot["loss_wiring"]["input_power_source"],
            "input_power_formula": config_snapshot["loss_wiring"]["implemented_formula"],
            "efficiency_term_enabled": config_snapshot["loss_wiring"]["efficiency_enabled"],
            "history_storage_mode": config_snapshot["artifact_policy"]["history_storage_mode"],
            "best_val_loss": None if best_step is None else best_val_loss,
            "best_step": best_step,
            "final_train_loss": None if last_history_entry is None else last_history_entry["train_loss"],
            "final_val_loss": None if last_history_entry is None else last_history_entry["val_loss"],
            "history_entry_count": history_entry_count,
            "grad_history_entry_count": grad_history_entry_count,
            "output_shapes": {
                "encoder_input": list(final_preview["x_hr"].shape),
                "phi_lr": list(final_preview["phi_lr"].shape),
                "U0": list(final_preview["output"]["U0"].shape),
                "I_out_full": list(final_preview["output"]["I_out_full"].shape),
                "I_out_roi": list(final_preview["pred_roi"].shape),
            },
            "artifacts": {
                "config_snapshot": str((run_dir / "config_snapshot.json").resolve()),
                "train_log": str(train_log_path.resolve()),
                "status": str(status_path.resolve()),
                "history": str((run_dir / "history.json").resolve()),
                "history_jsonl": str(history_jsonl_path.resolve()),
                "grad_stats": str((run_dir / "grad_stats.json").resolve()),
                "grad_stats_jsonl": str(grad_jsonl_path.resolve()),
                "loss_curve": str((run_dir / "loss_curve.png").resolve()),
                "loss_curve_points": str(loss_curve_points_path.resolve()),
                "preview_best": str((run_dir / "preview_best.png").resolve()),
                "preview_final": str((run_dir / "preview_final.png").resolve()),
                "phi_preview_best": str((run_dir / "phi_preview_best.png").resolve()),
                "phi_preview_final": str((run_dir / "phi_preview_final.png").resolve()),
                "checkpoint_latest": str(latest_checkpoint_path.resolve()),
                "checkpoint_best": str(best_checkpoint_path.resolve()),
            },
        }
        if (run_dir / "preview_step0.png").exists():
            run_summary["artifacts"]["preview_step0"] = str((run_dir / "preview_step0.png").resolve())
            run_summary["artifacts"]["phi_preview_step0"] = str(
                (run_dir / "phi_preview_step0.png").resolve()
            )
        if (run_dir / "preview_resume_start.png").exists():
            run_summary["artifacts"]["preview_resume_start"] = str(
                (run_dir / "preview_resume_start.png").resolve()
            )
            run_summary["artifacts"]["phi_preview_resume_start"] = str(
                (run_dir / "phi_preview_resume_start.png").resolve()
            )

        save_json(run_dir / "run_summary.json", run_summary)
        write_status("completed", "Run completed successfully.")
        logger.log(f"Run completed successfully. Summary={str((run_dir / 'run_summary.json').resolve())}")
        print(json.dumps(run_summary, indent=2))
    except KeyboardInterrupt:
        interruption_error = {
            "type": "KeyboardInterrupt",
            "message": "Training interrupted by user.",
        }
        save_json(failure_path, interruption_error)
        write_status("interrupted", "Run interrupted before completion.", interruption_error)
        logger.log("Run interrupted by user. status.json and failure.json were updated.")
        raise
    except Exception as exc:
        failure_payload = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_step": global_step,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(failure_path, failure_payload)
        write_status(
            "failed",
            "Run failed before completion.",
            {
                "type": failure_payload["type"],
                "message": failure_payload["message"],
                "failure_json": str(failure_path.resolve()),
            },
        )
        logger.log(f"Run failed with {type(exc).__name__}: {exc}")
        logger.log(traceback.format_exc())
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
