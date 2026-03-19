#!/usr/bin/env python
"""Run Stage 5 blind line-pair evaluation on a paper-path checkpoint."""

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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import Stage4RoiTargetAdapter
from src.eval.stage5_blind_targets import build_stage5_blind_line_pair_targets
from src.models.encoders import PaperPhaseEncoder
from src.models.optics.diffractive_decoder import DiffractiveDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 5 blind line-pair hook on a paper-aligned checkpoint."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preview-limit", type=int, default=None)
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


def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 2:
        return tensor.numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].numpy()
    raise ValueError(f"Expected grayscale tensor, got shape {tuple(tensor.shape)}.")


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
        raise KeyError("Checkpoint is missing config_snapshot required for Stage 5 blind eval.")

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


def save_image_grid(
    output_path: Path,
    images: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    title_key: str,
    max_items: int | None = None,
) -> None:
    item_count = images.shape[0] if max_items is None else min(images.shape[0], max_items)
    if item_count < 1:
        raise ValueError("save_image_grid requires at least one image.")
    num_cols = min(4, item_count)
    num_rows = int(np.ceil(item_count / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.0 * num_cols, 4.0 * num_rows))
    axes_array = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for flat_index in range(num_rows * num_cols):
        axis = axes_array.flat[flat_index]
        if flat_index >= item_count:
            axis.axis("off")
            continue
        axis.imshow(_to_gray_numpy(images[flat_index]), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(str(metadata[flat_index][title_key]))
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison_grid(
    output_path: Path,
    blind_targets: torch.Tensor,
    target_roi: torch.Tensor,
    prediction_roi: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    preview_limit: int,
) -> None:
    row_count = min(preview_limit, blind_targets.shape[0])
    if row_count < 1:
        raise ValueError("preview_limit must allow at least one comparison row.")

    blind_targets = blind_targets[:row_count].detach().cpu()
    target_roi = target_roi[:row_count].detach().cpu()
    prediction_roi = prediction_roi[:row_count].detach().cpu()
    abs_error = torch.abs(target_roi - prediction_roi)

    fig, axes = plt.subplots(row_count, 4, figsize=(12.0, max(3.0, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ("blind_target_hr", "target_roi", "model I_out_roi", "|target-model|")
    for row_index in range(row_count):
        panels = (
            _to_gray_numpy(blind_targets[row_index]),
            _to_gray_numpy(target_roi[row_index]),
            _to_gray_numpy(prediction_roi[row_index]),
            _to_gray_numpy(abs_error[row_index]),
        )
        error_max = float(abs_error[row_index].max().item()) + 1e-8
        ranges = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, error_max))
        for col_index, axis in enumerate(axes[row_index]):
            vmin, vmax = ranges[col_index]
            axis.imshow(panels[col_index], cmap="gray", vmin=vmin, vmax=vmax)
            if row_index == 0:
                axis.set_title(titles[col_index])
            axis.axis("off")
        axes[row_index, 0].set_ylabel(str(metadata[row_index]["target_id"]), rotation=90, labelpad=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_blind_forward(
    encoder: PaperPhaseEncoder,
    decoder: DiffractiveDecoder,
    blind_targets: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    prediction_parts: list[torch.Tensor] = []
    phi_parts: list[torch.Tensor] = []
    shape_summary: dict[str, list[int]] | None = None

    for batch_start in range(0, blind_targets.shape[0], batch_size):
        batch = blind_targets[batch_start : batch_start + batch_size].to(device)
        phi_lr = encoder(batch)
        output = decoder.forward_from_phase(phi_lr, return_intermediates=False)
        prediction_roi = output["I_out_roi"]
        prediction_parts.append(prediction_roi.detach().cpu())
        phi_parts.append(phi_lr.detach().cpu())
        if shape_summary is None:
            shape_summary = {
                "U0": list(output["U0"].shape),
                "U_out_full": list(output["U_out_full"].shape),
                "I_out_full": list(output["I_out_full"].shape),
                "I_out_roi": list(output["I_out_roi"].shape),
            }

    if shape_summary is None:
        raise RuntimeError("Blind target bank is empty.")
    return torch.cat(prediction_parts, dim=0), torch.cat(phi_parts, dim=0), shape_summary


def build_per_target_records(
    metadata: list[dict[str, Any]],
    target_roi: torch.Tensor,
    prediction_roi: torch.Tensor,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item, target, prediction in zip(metadata, target_roi, prediction_roi, strict=True):
        records.append(
            {
                **item,
                "target_roi_mean": float(target.mean().item()),
                "target_roi_sum": float(target.sum().item()),
                "prediction_min": float(prediction.min().item()),
                "prediction_max": float(prediction.max().item()),
                "prediction_mean": float(prediction.mean().item()),
                "prediction_sum": float(prediction.sum().item()),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    blind_root = load_yaml(args.config)
    blind_cfg = require_mapping(blind_root, "eval", "stage5_blind_linepair")
    runtime_cfg = dict(blind_cfg["runtime"])
    target_generation_cfg = dict(blind_cfg["target_generation"])

    preview_limit = int(
        args.preview_limit if args.preview_limit is not None else runtime_cfg["preview_limit"]
    )
    if preview_limit < 1:
        raise ValueError("preview_limit must be >= 1.")

    device = normalize_device(str(args.device or runtime_cfg["device"]))
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(blind_cfg["output_root"])
    run_name = args.run_name or blind_cfg.get("run_name") or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder, checkpoint_snapshot, depth = load_model_from_checkpoint(
        checkpoint_path,
        device=device,
    )

    blind_targets, metadata = build_stage5_blind_line_pair_targets(target_generation_cfg)
    target_adapter = Stage4RoiTargetAdapter(output_crop_hw=decoder.readout_config)
    target_roi = target_adapter(blind_targets)
    prediction_roi, phi_lr, sample_forward_shapes = run_blind_forward(
        encoder,
        decoder,
        blind_targets,
        batch_size=int(runtime_cfg["batch_size"]),
        device=device,
    )

    save_image_grid(run_dir / "blind_targets_grid.png", blind_targets, metadata, title_key="target_id")
    save_image_grid(run_dir / "blind_outputs_grid.png", prediction_roi, metadata, title_key="target_id")
    save_comparison_grid(
        run_dir / "blind_comparison_grid.png",
        blind_targets,
        target_roi,
        prediction_roi,
        metadata,
        preview_limit=preview_limit,
    )

    config_snapshot = {
        "blind_eval_config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "run_name": run_name,
        "device": str(device),
        "runtime": runtime_cfg,
        "target_generation": target_generation_cfg,
        "checkpoint_snapshot_excerpt": {
            "depth": depth,
            "optics_readout": checkpoint_snapshot["optics_config"]["readout"],
            "encoder_target_hw": checkpoint_snapshot["encoder_config"]["target_hw"],
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(run_dir / "config_snapshot.json", config_snapshot)

    readout_cfg = checkpoint_snapshot["optics_config"]["readout"]
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "blind_target_family": target_generation_cfg["target_family"],
        "blind_target_count": int(blind_targets.shape[0]),
        "blind_target_generation": {
            **target_generation_cfg,
            "binary_or_smoothed": (
                "binary" if float(target_generation_cfg.get("blur_sigma", 0.0)) == 0.0 else "smoothed"
            ),
        },
        "forward_hook": {
            "model_input": "generated_blind_target_hr",
            "prediction_tensor": "I_out_roi",
            "target_tensor_for_visual_reference": "target_roi",
            "hook_note": (
                "Issue 5.8 reuses the Stage 5 checkpoint-loading path and the same "
                "encoder -> decoder -> I_out_roi forward contract as regular eval."
            ),
        },
        "crop_assumptions": {
            "output_crop_hw": list(readout_cfg["output_crop_hw"]),
            "crop_policy": readout_cfg.get("crop_policy", "center_crop"),
            "center_offset_hw": readout_cfg.get("center_offset_hw"),
            "fov_alignment_note": readout_cfg.get("fov_alignment_note"),
            "target_adapter": {
                "resize_mode": target_adapter.resize_mode,
                "align_corners": target_adapter.align_corners,
                "clamp_range": list(target_adapter.clamp_range),
            },
        },
        "scalar_metric_policy": {
            "paper_final_blind_metric_claimed": False,
            "note": (
                "Issue 5.8 saves reproducible blind-test artifacts and per-target metadata "
                "for visual inspection; it does not introduce a new paper-final scalar blind score."
            ),
        },
        "output_shapes": {
            "blind_target_hr": list(blind_targets.shape),
            "target_roi": list(target_roi.shape),
            "phi_lr": list(phi_lr.shape),
            "sample_forward": sample_forward_shapes,
        },
        "per_target_records": build_per_target_records(metadata, target_roi, prediction_roi),
        "artifacts": {
            "config_snapshot": str((run_dir / "config_snapshot.json").resolve()),
            "summary": str((run_dir / "summary.json").resolve()),
            "blind_targets_grid": str((run_dir / "blind_targets_grid.png").resolve()),
            "blind_outputs_grid": str((run_dir / "blind_outputs_grid.png").resolve()),
            "blind_comparison_grid": str((run_dir / "blind_comparison_grid.png").resolve()),
        },
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
