#!/usr/bin/env python
"""Run interpolation SR baseline (bilinear / bicubic) with unified evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import EMNIST

# Keep `python scripts/run_interpolation_baseline.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate_batch


INTERP_MODES = ("bilinear", "bicubic")
INTERP_DEFAULTS: Dict[str, Any] = {
    "split": "train",
    "emnist_split": "letters",
    "num_samples": 100,
    "seed": 42,
    "output_dir": "outputs/interpolation/train",
    "mode": "bicubic",
    "hr_size": 96,
    "scale": 4,
    "val_ratio": 0.1,
    "save_grid": True,
    "save_individual": True,
    "download": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolation baseline for Stage 1/2 SR sanity-check."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="train")
    parser.add_argument("--emnist-split", type=str, default="letters")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/interpolation/train")
    parser.add_argument("--mode", type=str, choices=INTERP_MODES, default="bicubic")
    parser.add_argument("--hr-size", type=int, default=96)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--save-grid", dest="save_grid", action="store_true", default=True
    )
    parser.add_argument("--no-save-grid", dest="save_grid", action="store_false")
    parser.add_argument(
        "--save-individual", dest="save_individual", action="store_true", default=True
    )
    parser.add_argument("--no-save-individual", dest="save_individual", action="store_false")
    parser.add_argument("--download", dest="download", action="store_true", default=False)
    parser.add_argument("--no-download", dest="download", action="store_false")
    return parser.parse_args()


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def get_nested(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_runtime_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    # CLI has priority; apply config only when argument is still parser default.
    seed_cfg = get_nested(cfg, ("seed",), None)
    if args.seed == INTERP_DEFAULTS["seed"] and seed_cfg is not None:
        args.seed = int(seed_cfg)

    split_cfg = get_nested(cfg, ("data", "split"), None)
    if args.split == INTERP_DEFAULTS["split"] and split_cfg is not None:
        args.split = str(split_cfg)

    emnist_split_cfg = get_nested(cfg, ("data", "emnist_split"), None)
    if args.emnist_split == INTERP_DEFAULTS["emnist_split"] and emnist_split_cfg is not None:
        args.emnist_split = str(emnist_split_cfg)

    num_samples_cfg = get_nested(cfg, ("data", "num_samples"), None)
    if args.num_samples == INTERP_DEFAULTS["num_samples"] and num_samples_cfg is not None:
        args.num_samples = int(num_samples_cfg)

    output_dir_cfg = get_nested(cfg, ("output", "dir"), None)
    if args.output_dir == INTERP_DEFAULTS["output_dir"] and output_dir_cfg is not None:
        args.output_dir = str(output_dir_cfg)

    mode_cfg = get_nested(cfg, ("model", "mode"), None)
    if args.mode == INTERP_DEFAULTS["mode"] and mode_cfg is not None:
        args.mode = str(mode_cfg)

    hr_size_cfg = get_nested(cfg, ("data", "hr_size"), None)
    if args.hr_size == INTERP_DEFAULTS["hr_size"] and hr_size_cfg is not None:
        args.hr_size = int(hr_size_cfg)

    val_ratio_cfg = get_nested(cfg, ("data", "val_ratio"), None)
    if args.val_ratio == INTERP_DEFAULTS["val_ratio"] and val_ratio_cfg is not None:
        args.val_ratio = float(val_ratio_cfg)

    save_grid_cfg = get_nested(cfg, ("eval", "save_grid"), None)
    if args.save_grid == INTERP_DEFAULTS["save_grid"] and save_grid_cfg is not None:
        args.save_grid = bool(save_grid_cfg)

    save_individual_cfg = get_nested(cfg, ("eval", "save_individual"), None)
    if (
        args.save_individual == INTERP_DEFAULTS["save_individual"]
        and save_individual_cfg is not None
    ):
        args.save_individual = bool(save_individual_cfg)

    download_cfg = get_nested(cfg, ("runtime", "download"), None)
    if args.download == INTERP_DEFAULTS["download"] and download_cfg is not None:
        args.download = bool(download_cfg)

    if args.dataset_root is None:
        cfg_root = get_nested(cfg, ("data", "root"), None)
        fallback_root = Path("data/raw/emnist")
        if isinstance(cfg_root, str):
            cfg_root_path = Path(cfg_root)
            looks_like_emnist_root = (
                "emnist" in cfg_root.lower()
                or (cfg_root_path / "EMNIST").exists()
                or (cfg_root_path / "raw").exists()
            )
            args.dataset_root = cfg_root if looks_like_emnist_root else str(fallback_root)
        else:
            args.dataset_root = str(fallback_root)

    cfg_scale = get_nested(cfg, ("data", "scale"), None)
    if args.scale == 4 and cfg_scale is not None:
        args.scale = int(cfg_scale)
    return args


def build_emnist_dataset(args: argparse.Namespace) -> Tuple[Dataset[Any], List[int]]:
    to_tensor = transforms.ToTensor()
    train_flag = args.split in ("train", "val")
    base_dataset = EMNIST(
        root=args.dataset_root,
        split=args.emnist_split,
        train=train_flag,
        transform=to_tensor,
        download=args.download,
    )
    if args.split == "test":
        indices = list(range(len(base_dataset)))
        return base_dataset, indices

    all_indices = list(range(len(base_dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    val_count = max(1, int(len(all_indices) * args.val_ratio))
    split_indices = all_indices[:val_count] if args.split == "val" else all_indices[val_count:]
    if len(split_indices) == 0:
        raise ValueError(
            f"Split '{args.split}' is empty. Adjust --val-ratio (current: {args.val_ratio})."
        )
    return Subset(base_dataset, split_indices), split_indices


def as_chw_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
    elif isinstance(image, np.ndarray):
        t = torch.from_numpy(image).float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
        t = t.permute(2, 0, 1)
    if t.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {tuple(t.shape)}")
    return t


def interpolate_chw(image_chw: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    kwargs: Dict[str, Any] = {"size": size, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(image_chw.unsqueeze(0), **kwargs).squeeze(0)


def tensor_to_display(image_chw: torch.Tensor) -> Tuple[np.ndarray, bool]:
    c = image_chw.shape[0]
    if c == 1:
        return image_chw[0].numpy(), True
    if c in (3, 4):
        return image_chw.permute(1, 2, 0).numpy(), False
    return image_chw[0].numpy(), True


def minmax_text(t: torch.Tensor) -> str:
    return f"[{float(t.min()):.3f}, {float(t.max()):.3f}]"


def save_sample_figure(
    output_path: Path,
    sample_index: int,
    source_index: int,
    hr: torch.Tensor,
    lr: torch.Tensor,
    pred: torch.Tensor,
    mode: str,
) -> None:
    diff = torch.abs(pred - hr)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    items = [("HR", hr), ("LR", lr), (f"Interp({mode})", pred), ("|Pred-HR|", diff)]
    for ax, (name, tensor) in zip(axes, items):
        arr, is_gray = tensor_to_display(tensor)
        if name == "|Pred-HR|":
            vmax = max(float(diff.max()), 1e-6)
            ax.imshow(arr, cmap="inferno", vmin=0.0, vmax=vmax)
        elif is_gray:
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(arr, 0.0, 1.0))
        ax.set_title(f"{name}\nshape={tuple(tensor.shape)}\nmin/max={minmax_text(tensor)}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"sample_idx={sample_index}, source_idx={source_index}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_grid_figure(
    output_path: Path,
    records: List[Dict[str, Any]],
    triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    mode: str,
) -> None:
    rows = len(records)
    fig, axes = plt.subplots(rows, 4, figsize=(14, max(3.0 * rows, 4.0)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (rec, (hr, lr, pred)) in enumerate(zip(records, triplets)):
        diff = torch.abs(pred - hr)
        for c, (name, tensor) in enumerate(
            (("HR", hr), ("LR", lr), (f"Interp({mode})", pred), ("|Pred-HR|", diff))
        ):
            ax = axes[r, c]
            arr, is_gray = tensor_to_display(tensor)
            if name == "|Pred-HR|":
                vmax = max(float(diff.max()), 1e-6)
                ax.imshow(arr, cmap="inferno", vmin=0.0, vmax=vmax)
            elif is_gray:
                ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(np.clip(arr, 0.0, 1.0))
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(
                    f"idx={rec['dataset_index']}\nsrc={rec['source_index']}\n{rec['hr_shape']}",
                    fontsize=8,
                )
            ax.axis("off")

    fig.suptitle("Interpolation Baseline Grid", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.hr_size < 1:
        raise ValueError("--hr-size must be >= 1")
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.hr_size % args.scale != 0:
        raise ValueError(
            f"--hr-size ({args.hr_size}) must be divisible by --scale ({args.scale})."
        )

    dataset, source_indices = build_emnist_dataset(args)
    if len(dataset) < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples, but split '{args.split}' only has {len(dataset)}."
        )

    random.seed(args.seed)
    chosen = sorted(random.sample(range(len(dataset)), k=args.num_samples))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hr_list: List[torch.Tensor] = []
    pred_list: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    lr_size = args.hr_size // args.scale
    for rank, ds_idx in enumerate(chosen, start=1):
        sample = dataset[ds_idx]
        image = sample[0] if isinstance(sample, (tuple, list)) else sample
        label = int(sample[1]) if isinstance(sample, (tuple, list)) and len(sample) > 1 else None

        hr = as_chw_tensor(image)
        if hr.shape[-2:] != (args.hr_size, args.hr_size):
            hr = interpolate_chw(hr, (args.hr_size, args.hr_size), mode="bicubic")

        lr = interpolate_chw(hr, (lr_size, lr_size), mode="bicubic")
        pred = interpolate_chw(lr, (args.hr_size, args.hr_size), mode=args.mode)

        source_idx = int(source_indices[ds_idx])
        rec = {
            "sample_rank": rank,
            "dataset_index": int(ds_idx),
            "source_index": source_idx,
            "label": label,
            "hr_shape": list(hr.shape),
            "lr_shape": list(lr.shape),
            "pred_shape": list(pred.shape),
            "hr_min": float(hr.min()),
            "hr_max": float(hr.max()),
            "lr_min": float(lr.min()),
            "lr_max": float(lr.max()),
            "pred_min": float(pred.min()),
            "pred_max": float(pred.max()),
        }
        records.append(rec)
        hr_list.append(hr)
        pred_list.append(pred)
        triplets.append((hr, lr, pred))

        if args.save_individual:
            out_file = output_dir / f"sample_{rank:03d}_idx_{ds_idx:06d}.png"
            save_sample_figure(
                out_file,
                sample_index=int(ds_idx),
                source_index=source_idx,
                hr=hr,
                lr=lr,
                pred=pred,
                mode=args.mode,
            )

    preds = torch.stack(pred_list, dim=0)
    targets = torch.stack(hr_list, dim=0)
    metrics = evaluate_batch(preds, targets)

    grid_file = None
    if args.save_grid:
        grid_file = output_dir / "interpolation_grid.png"
        save_grid_figure(grid_file, records, triplets, mode=args.mode)

    metrics_payload: Dict[str, Any] = {
        "mode": args.mode,
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "hr_size": args.hr_size,
        "scale": args.scale,
        "lr_size": lr_size,
        "psnr_mean": float(metrics["psnr_mean"]),
        "ssim_mean": float(metrics["ssim_mean"]),
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    summary: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "emnist_split": args.emnist_split,
        "hr_size": args.hr_size,
        "scale": args.scale,
        "lr_size": lr_size,
        "download": args.download,
        "save_grid": args.save_grid,
        "save_individual": args.save_individual,
        "metrics": metrics_payload,
        "grid_file": str(grid_file.resolve()) if grid_file else None,
        "samples": records,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "metrics": metrics_payload,
        "grid_file": grid_file,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    args = resolve_runtime_args(args, cfg)
    result = run_baseline(args)

    metrics = result["metrics"]
    print("Interpolation baseline completed.")
    print(
        f"mode={metrics['mode']}, split={metrics['split']}, samples={metrics['num_samples']}, seed={metrics['seed']}"
    )
    print(f"PSNR: {metrics['psnr_mean']:.4f}")
    print(f"SSIM: {metrics['ssim_mean']:.6f}")
    print(f"metrics: {result['metrics_path']}")
    print(f"summary: {result['summary_path']}")
    if result["grid_file"] is not None:
        print(f"grid: {result['grid_file']}")


if __name__ == "__main__":
    main()
