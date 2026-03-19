#!/usr/bin/env python
"""Evaluate the Stage 5 paper-aligned checkpoint with a bicubic baseline."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import Stage4RoiTargetAdapter, build_stage5_emnist_display_dataset
from src.eval.evaluator import evaluate_batch
from src.models.encoders import PaperPhaseEncoder
from src.models.optics.diffractive_decoder import DiffractiveDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 5 paper-aligned checkpoint with bicubic baseline."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, choices=("val", "test"))
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 2:
        return tensor.numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].numpy()
    raise ValueError(f"Expected grayscale tensor, got shape {tuple(tensor.shape)}.")


def build_dataset(
    dataset_cfg: dict[str, Any],
    *,
    split: str,
    subset_size: int | None,
    download: bool,
) -> Any:
    dataset = build_stage5_emnist_display_dataset(
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
    if subset_size is None:
        return dataset
    subset_size = int(subset_size)
    if subset_size < 1:
        raise ValueError("subset_size must be >= 1.")
    if subset_size > len(dataset):
        raise ValueError(f"subset_size {subset_size} exceeds dataset length {len(dataset)}.")
    return Subset(dataset, list(range(subset_size)))


def build_encoder_decoder(
    encoder_cfg: dict[str, Any],
    optics_cfg: dict[str, Any],
    *,
    depth: int,
    device: torch.device,
) -> tuple[PaperPhaseEncoder, DiffractiveDecoder]:
    target_hw = tuple(encoder_cfg["target_hw"])
    optics_input_hw = tuple(optics_cfg["grid"]["input_pattern_hw"])
    if target_hw != optics_input_hw:
        raise ValueError(
            "Stage 5 encoder target_hw must match optics input_pattern_hw, "
            f"got {target_hw} vs {optics_input_hw}."
        )

    encoder = PaperPhaseEncoder(
        in_channels=int(encoder_cfg["in_channels"]),
        input_hw=tuple(encoder_cfg["input_hw"]),
        target_hw=target_hw,
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


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[PaperPhaseEncoder, DiffractiveDecoder, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_snapshot = checkpoint.get("config_snapshot")
    if not isinstance(config_snapshot, dict):
        raise KeyError("Checkpoint is missing config_snapshot required for Stage 5 eval.")

    encoder_cfg = config_snapshot.get("encoder_config")
    optics_cfg = config_snapshot.get("optics_config")
    depth = config_snapshot.get("depth")
    if not isinstance(encoder_cfg, dict) or not isinstance(optics_cfg, dict) or depth is None:
        raise KeyError(
            "Checkpoint config_snapshot must contain encoder_config, optics_config, and depth."
        )

    encoder, decoder = build_encoder_decoder(
        encoder_cfg,
        optics_cfg,
        depth=int(depth),
        device=device,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, config_snapshot, int(depth)


def interpolate_single_channel(
    tensor: torch.Tensor,
    *,
    size: tuple[int, int],
    mode: str,
    align_corners: bool,
    antialias: bool,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "size": size,
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = align_corners
    if antialias and mode in {"bilinear", "bicubic"}:
        kwargs["antialias"] = True
    return F.interpolate(tensor.float(), **kwargs)


def build_bicubic_baseline(target_roi: torch.Tensor, baseline_cfg: dict[str, Any]) -> torch.Tensor:
    lr_hw = tuple(int(value) for value in baseline_cfg["input_lr_hw"])
    downsample = interpolate_single_channel(
        target_roi,
        size=lr_hw,
        mode=str(baseline_cfg["downsample_mode"]),
        align_corners=bool(baseline_cfg["align_corners"]),
        antialias=bool(baseline_cfg["antialias_downsample"]),
    )
    return interpolate_single_channel(
        downsample,
        size=tuple(int(value) for value in target_roi.shape[-2:]),
        mode=str(baseline_cfg["upsample_mode"]),
        align_corners=bool(baseline_cfg["align_corners"]),
        antialias=bool(baseline_cfg["antialias_upsample"]),
    )


@torch.no_grad()
def run_model_batch(
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    batch: dict[str, Any],
    *,
    target_adapter: Stage4RoiTargetAdapter,
    device: torch.device,
) -> dict[str, Any]:
    moved_batch = move_batch_to_device(batch, device)
    x_hr = moved_batch["x_hr"]
    target_hr = moved_batch["target_hr"]
    if not isinstance(x_hr, torch.Tensor) or not isinstance(target_hr, torch.Tensor):
        raise TypeError("Batch must contain tensor fields 'x_hr' and 'target_hr'.")

    phi_lr = encoder(x_hr)
    output = decoder.forward_from_phase(phi_lr, return_intermediates=False)
    prediction_roi = output["I_out_roi"]
    target_roi = target_adapter(target_hr)
    if not isinstance(prediction_roi, torch.Tensor):
        raise TypeError("Decoder output is missing tensor field 'I_out_roi'.")

    return {
        "x_hr": x_hr,
        "target_hr": target_hr,
        "target_roi": target_roi,
        "phi_lr": phi_lr,
        "prediction_roi": prediction_roi,
        "output": output,
        "dataset_index": moved_batch.get("dataset_index"),
        "letter_count": moved_batch.get("letter_count"),
    }


def weighted_average(metric_summaries: list[dict[str, float | int]]) -> dict[str, float]:
    total_count = sum(int(item["num_samples"]) for item in metric_summaries)
    if total_count < 1:
        raise RuntimeError("No metric summaries were collected.")
    return {
        "psnr_mean": sum(float(item["psnr_mean"]) * int(item["num_samples"]) for item in metric_summaries)
        / total_count,
        "ssim_mean": sum(float(item["ssim_mean"]) * int(item["num_samples"]) for item in metric_summaries)
        / total_count,
        "num_samples": float(total_count),
    }


def save_eval_preview(
    output_path: Path,
    batch_output: dict[str, Any],
    bicubic_prediction: torch.Tensor,
    *,
    preview_limit: int,
) -> None:
    row_count = min(preview_limit, batch_output["x_hr"].shape[0])
    x_hr = batch_output["x_hr"][:row_count].detach().cpu()
    target_roi = batch_output["target_roi"][:row_count].detach().cpu()
    model_prediction = batch_output["prediction_roi"][:row_count].detach().cpu()
    bicubic_prediction = bicubic_prediction[:row_count].detach().cpu()
    model_abs_error = torch.abs(target_roi - model_prediction)
    bicubic_abs_error = torch.abs(target_roi - bicubic_prediction)

    fig, axes = plt.subplots(row_count, 6, figsize=(18, max(3.0, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ("input", "target_roi", "model", "bicubic", "|model-target|", "|bicubic-target|")
    for row_index in range(row_count):
        panels = (
            _to_gray_numpy(x_hr[row_index]),
            _to_gray_numpy(target_roi[row_index]),
            _to_gray_numpy(model_prediction[row_index]),
            _to_gray_numpy(bicubic_prediction[row_index]),
            _to_gray_numpy(model_abs_error[row_index]),
            _to_gray_numpy(bicubic_abs_error[row_index]),
        )
        error_max = float(max(model_abs_error.max().item(), bicubic_abs_error.max().item())) + 1e-8
        ranges = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, error_max),
            (0.0, error_max),
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


def main() -> None:
    args = parse_args()
    eval_root = load_yaml(args.config)
    eval_cfg = require_mapping(eval_root, "eval", "stage5_paper")
    dataset_cfg = require_mapping(load_yaml(eval_cfg["dataset_config_path"]), "data", "stage5_display")

    runtime_cfg = dict(eval_cfg["runtime"])
    split = str(args.split or runtime_cfg["split"])
    subset_size = args.subset_size if args.subset_size is not None else runtime_cfg["subset_size"]
    preview_limit = int(
        args.preview_limit if args.preview_limit is not None else runtime_cfg["preview_limit"]
    )
    if preview_limit < 1:
        raise ValueError("preview_limit must be >= 1.")

    device = normalize_device(str(args.device or runtime_cfg["device"]))
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(eval_cfg["output_root"])
    run_name = args.run_name or eval_cfg.get("run_name") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder, checkpoint_snapshot, depth = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
    )
    dataset = build_dataset(
        dataset_cfg,
        split=split,
        subset_size=subset_size,
        download=bool(args.download or runtime_cfg.get("download", False)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(runtime_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(runtime_cfg["num_workers"]),
    )
    target_adapter = Stage4RoiTargetAdapter(output_crop_hw=decoder.readout_config)
    baseline_cfg = dict(eval_cfg["bicubic_baseline"])

    config_snapshot = {
        "eval_config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "run_name": run_name,
        "split": split,
        "device": str(device),
        "runtime": {
            **runtime_cfg,
            "subset_size": subset_size,
            "preview_limit": preview_limit,
            "download": bool(args.download or runtime_cfg.get("download", False)),
        },
        "bicubic_baseline": baseline_cfg,
        "checkpoint_snapshot_excerpt": {
            "depth": depth,
            "optics_readout": checkpoint_snapshot["optics_config"]["readout"],
            "encoder_target_hw": checkpoint_snapshot["encoder_config"]["target_hw"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_dir / "config_snapshot.json", config_snapshot)

    model_metric_summaries: list[dict[str, float | int]] = []
    bicubic_metric_summaries: list[dict[str, float | int]] = []
    preview_batch_output: dict[str, Any] | None = None
    preview_bicubic: torch.Tensor | None = None

    for batch in dataloader:
        batch_output = run_model_batch(
            encoder,
            decoder,
            batch,
            target_adapter=target_adapter,
            device=device,
        )
        bicubic_prediction = build_bicubic_baseline(batch_output["target_roi"], baseline_cfg)

        model_metric_summaries.append(
            evaluate_batch(batch_output["prediction_roi"], batch_output["target_roi"])
        )
        bicubic_metric_summaries.append(
            evaluate_batch(bicubic_prediction, batch_output["target_roi"])
        )

        if preview_batch_output is None:
            preview_batch_output = batch_output
            preview_bicubic = bicubic_prediction.detach().cpu()

    if preview_batch_output is None or preview_bicubic is None:
        raise RuntimeError("Evaluation dataloader is empty.")

    model_metrics = weighted_average(model_metric_summaries)
    bicubic_metrics = weighted_average(bicubic_metric_summaries)
    save_eval_preview(
        run_dir / "preview_comparison.png",
        preview_batch_output,
        preview_bicubic,
        preview_limit=preview_limit,
    )

    readout_cfg = checkpoint_snapshot["optics_config"]["readout"]
    crop_hw = list(readout_cfg["output_crop_hw"])
    preview_dataset_index = preview_batch_output["dataset_index"]
    preview_letter_count = preview_batch_output["letter_count"]
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "evaluated_split": split,
        "evaluated_num_samples": int(model_metrics["num_samples"]),
        "model_metrics": {
            "psnr_mean": float(model_metrics["psnr_mean"]),
            "ssim_mean": float(model_metrics["ssim_mean"]),
        },
        "bicubic_baseline_metrics": {
            "psnr_mean": float(bicubic_metrics["psnr_mean"]),
            "ssim_mean": float(bicubic_metrics["ssim_mean"]),
        },
        "scored_tensor_pair": {
            "prediction": "I_out_roi",
            "target": "target_roi",
        },
        "crop_assumptions": {
            "output_crop_hw": crop_hw,
            "crop_policy": readout_cfg.get("crop_policy", "center_crop"),
            "center_offset_hw": readout_cfg.get("center_offset_hw"),
            "fov_alignment_note": readout_cfg.get("fov_alignment_note"),
            "target_adapter": {
                "resize_mode": target_adapter.resize_mode,
                "align_corners": target_adapter.align_corners,
                "clamp_range": list(target_adapter.clamp_range),
            },
        },
        "normalization_assumptions": {
            "metrics_helper": "src.eval.evaluator.evaluate_batch",
            "metric_value_range_policy": (
                "src.eval.metrics._normalize_to_unit normalizes to [0,1], "
                "including /255 for uint-like inputs and clipping otherwise."
            ),
        },
        "bicubic_baseline_assumptions": {
            "source_tensor": baseline_cfg["source_tensor"],
            "implementation": baseline_cfg["implementation"],
            "input_lr_hw": list(baseline_cfg["input_lr_hw"]),
            "scale_factor": baseline_cfg["scale_factor"],
            "downsample_mode": baseline_cfg["downsample_mode"],
            "upsample_mode": baseline_cfg["upsample_mode"],
            "align_corners": bool(baseline_cfg["align_corners"]),
            "antialias_downsample": bool(baseline_cfg["antialias_downsample"]),
            "antialias_upsample": bool(baseline_cfg["antialias_upsample"]),
            "note": baseline_cfg["note"],
        },
        "output_shapes": {
            "x_hr": list(preview_batch_output["x_hr"].shape),
            "target_roi": list(preview_batch_output["target_roi"].shape),
            "prediction_roi": list(preview_batch_output["prediction_roi"].shape),
            "phi_lr": list(preview_batch_output["phi_lr"].shape),
        },
        "preview_batch_metadata": {
            "dataset_index": (
                None
                if preview_dataset_index is None
                else [int(value) for value in preview_dataset_index.tolist()]
            ),
            "letter_count": (
                None
                if preview_letter_count is None
                else [int(value) for value in preview_letter_count.tolist()]
            ),
        },
        "artifacts": {
            "config_snapshot": str((run_dir / "config_snapshot.json").resolve()),
            "summary": str((run_dir / "summary.json").resolve()),
            "preview_comparison": str((run_dir / "preview_comparison.png").resolve()),
        },
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
