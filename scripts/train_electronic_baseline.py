#!/usr/bin/env python
"""Train a pure electronic bottleneck-constrained SR baseline."""

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
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import EMNIST

# Keep `python scripts/train_electronic_baseline.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate_batch
from src.models.electronic_baseline import ElectronicBottleneckAutoencoder


TRAIN_DEFAULTS: Dict[str, Any] = {
    "output_dir": "outputs/electronic_baseline/default_run",
    "epochs": 10,
    "batch_size": 32,
    "lr": 1e-3,
    "seed": 42,
    "hr_size": 96,
    "sr_factor": 4,
    "latent_channels": 1,
    "num_samples": 2000,
    "val_ratio": 0.1,
    "num_workers": 0,
    "download": False,
    "save_samples": True,
    "emnist_split": "letters",
    "overfit_single_sample": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage1&2-4: train pure electronic bottleneck baseline."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hr-size", type=int, default=None)
    parser.add_argument("--sr-factor", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Total samples used from EMNIST train split before train/val split.",
    )
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--download", dest="download", action="store_true")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--save-samples", dest="save_samples", action="store_true")
    parser.add_argument("--no-save-samples", dest="save_samples", action="store_false")
    parser.add_argument("--emnist-split", type=str, default=None)
    parser.add_argument(
        "--overfit-single-sample",
        action="store_true",
        help="Use the same single sample for both train and val sanity checks.",
    )
    parser.set_defaults(download=None, save_samples=None)
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


def choose_runtime_value(
    cli_value: Any,
    cfg: Dict[str, Any],
    keys: Sequence[str],
    default: Any,
    cast: Any = None,
) -> Any:
    value = cli_value
    if value is None:
        value = get_nested(cfg, keys, default)
    if value is None:
        value = default
    if cast is None:
        return value
    return cast(value)


def resolve_runtime_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    args.seed = choose_runtime_value(args.seed, cfg, ("seed",), TRAIN_DEFAULTS["seed"], int)
    args.output_dir = choose_runtime_value(
        args.output_dir,
        cfg,
        ("output", "dir"),
        TRAIN_DEFAULTS["output_dir"],
        str,
    )
    args.epochs = choose_runtime_value(
        args.epochs,
        cfg,
        ("train", "epochs"),
        TRAIN_DEFAULTS["epochs"],
        int,
    )
    args.batch_size = choose_runtime_value(
        args.batch_size,
        cfg,
        ("train", "batch_size"),
        TRAIN_DEFAULTS["batch_size"],
        int,
    )
    args.lr = choose_runtime_value(args.lr, cfg, ("train", "lr"), TRAIN_DEFAULTS["lr"], float)
    args.hr_size = choose_runtime_value(
        args.hr_size,
        cfg,
        ("data", "hr_size"),
        TRAIN_DEFAULTS["hr_size"],
        int,
    )

    sr_factor_cfg = get_nested(cfg, ("data", "sr_factor"), None)
    if sr_factor_cfg is None:
        sr_factor_cfg = get_nested(cfg, ("data", "scale"), None)
    args.sr_factor = int(
        args.sr_factor
        if args.sr_factor is not None
        else sr_factor_cfg
        if sr_factor_cfg is not None
        else TRAIN_DEFAULTS["sr_factor"]
    )

    args.latent_channels = choose_runtime_value(
        args.latent_channels,
        cfg,
        ("model", "latent_channels"),
        TRAIN_DEFAULTS["latent_channels"],
        int,
    )
    args.num_samples = choose_runtime_value(
        args.num_samples,
        cfg,
        ("data", "num_samples"),
        TRAIN_DEFAULTS["num_samples"],
        int,
    )
    args.val_ratio = choose_runtime_value(
        args.val_ratio,
        cfg,
        ("data", "val_ratio"),
        TRAIN_DEFAULTS["val_ratio"],
        float,
    )
    args.num_workers = choose_runtime_value(
        args.num_workers,
        cfg,
        ("data", "num_workers"),
        TRAIN_DEFAULTS["num_workers"],
        int,
    )
    args.download = choose_runtime_value(
        args.download,
        cfg,
        ("runtime", "download"),
        TRAIN_DEFAULTS["download"],
        bool,
    )
    args.save_samples = choose_runtime_value(
        args.save_samples,
        cfg,
        ("eval", "save_samples"),
        TRAIN_DEFAULTS["save_samples"],
        bool,
    )
    args.emnist_split = choose_runtime_value(
        args.emnist_split,
        cfg,
        ("data", "emnist_split"),
        TRAIN_DEFAULTS["emnist_split"],
        str,
    )

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
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_val_datasets(args: argparse.Namespace) -> Tuple[Dataset[Any], Dataset[Any], Dict[str, int]]:
    resize_to_hr = transforms.Compose(
        [
            transforms.Resize(
                (args.hr_size, args.hr_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )
    base_dataset = EMNIST(
        root=args.dataset_root,
        split=args.emnist_split,
        train=True,
        transform=resize_to_hr,
        download=args.download,
    )
    all_indices = list(range(len(base_dataset)))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)

    if args.num_samples > 0:
        use_count = min(args.num_samples, len(all_indices))
        active_indices = all_indices[:use_count]
    else:
        active_indices = all_indices

    if len(active_indices) < 1:
        raise ValueError("Need at least 1 sample.")

    if args.overfit_single_sample:
        overfit_index = active_indices[0]
        train_ds = Subset(base_dataset, [overfit_index])
        val_ds = Subset(base_dataset, [overfit_index])
        sizes = {"train": 1, "val": 1, "total": 1}
        return train_ds, val_ds, sizes

    if len(active_indices) < 2:
        raise ValueError("Need at least 2 samples to split train/val.")

    val_count = max(1, int(len(active_indices) * args.val_ratio))
    if val_count >= len(active_indices):
        val_count = len(active_indices) - 1

    val_indices = active_indices[:val_count]
    train_indices = active_indices[val_count:]
    train_ds = Subset(base_dataset, train_indices)
    val_ds = Subset(base_dataset, val_indices)
    sizes = {"train": len(train_ds), "val": len(val_ds), "total": len(active_indices)}
    return train_ds, val_ds, sizes


def prepare_hr_batch(images: torch.Tensor, hr_size: int) -> torch.Tensor:
    x = images.float()
    if x.ndim != 4:
        raise ValueError(f"Expected batch [B,C,H,W], got shape {tuple(x.shape)}")
    if x.shape[-2:] != (hr_size, hr_size):
        raise ValueError(
            "Unexpected batch resolution after dataset transform: "
            f"got {tuple(x.shape[-2:])}, expected {(hr_size, hr_size)}"
        )
    return torch.clamp(x, 0.0, 1.0)


def _to_gray_numpy(x: torch.Tensor) -> np.ndarray:
    t = x.detach().cpu().float()
    if t.ndim == 3:
        if t.shape[0] == 1:
            return t[0].numpy()
        return t.mean(dim=0).numpy()
    if t.ndim == 2:
        return t.numpy()
    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


def save_reconstruction_grid(
    output_path: Path,
    rows: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    if not rows:
        raise ValueError("Need at least one preview row to save sample grid.")

    row_count = len(rows)
    fig, axes = plt.subplots(row_count, 4, figsize=(12, max(3.2, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (hr, latent, recon) in enumerate(rows):
        diff = torch.abs(recon - hr)
        latent_vis = torch.nn.functional.interpolate(
            latent.unsqueeze(0),
            size=hr.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        hr_np = _to_gray_numpy(hr)
        latent_np = _to_gray_numpy(latent_vis)
        recon_np = _to_gray_numpy(recon)
        diff_np = _to_gray_numpy(diff)
        diff_vmax = max(1e-6, float(diff_np.max()))

        panels = [hr_np, latent_np, recon_np, diff_np]
        cmaps = ["gray", "gray", "gray", "inferno"]
        vmn_vmx = [(0.0, 1.0), (None, None), (0.0, 1.0), (0.0, diff_vmax)]
        titles = ["HR", "Latent (upsampled)", "Recon", "|Recon-HR|"]
        for col_idx, ax in enumerate(axes[row_idx]):
            vmin, vmax = vmn_vmx[col_idx]
            ax.imshow(panels[col_idx], cmap=cmaps[col_idx], vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(titles[col_idx])
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(f"sample {row_idx + 1}", rotation=90, labelpad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_validation(
    model: ElectronicBottleneckAutoencoder,
    loader: DataLoader,
    device: torch.device,
    hr_size: int,
    loss_fn: nn.Module,
    preview_limit: int = 4,
) -> Tuple[float, Dict[str, float | int], List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    pred_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []
    preview_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for images, _labels in loader:
            hr = prepare_hr_batch(images, hr_size=hr_size).to(device)
            recon, latent = model(hr, return_latent=True)
            loss = loss_fn(recon, hr)
            bs = hr.shape[0]
            total_loss += float(loss.item()) * bs
            total_count += bs

            pred_list.append(recon.detach().cpu())
            target_list.append(hr.detach().cpu())

            remaining = preview_limit - len(preview_rows)
            if remaining > 0:
                take_count = min(remaining, hr.shape[0])
                for idx in range(take_count):
                    preview_rows.append(
                        (
                            hr[idx].detach().cpu(),
                            latent[idx].detach().cpu(),
                            recon[idx].detach().cpu(),
                        )
                    )

    if total_count == 0:
        raise RuntimeError("Validation loader is empty.")

    preds = torch.cat(pred_list, dim=0)
    targets = torch.cat(target_list, dim=0)
    metrics = evaluate_batch(preds, targets)
    mean_loss = total_loss / total_count
    return mean_loss, metrics, preview_rows


def train(args: argparse.Namespace) -> Dict[str, Any]:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.hr_size < 1:
        raise ValueError("--hr-size must be >= 1")
    if args.sr_factor != 4:
        raise ValueError("This baseline currently supports --sr-factor 4 only.")
    if args.hr_size % args.sr_factor != 0:
        raise ValueError("--hr-size must be divisible by --sr-factor.")
    if args.latent_channels < 1:
        raise ValueError("--latent-channels must be >= 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, data_sizes = build_train_val_datasets(args)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = ElectronicBottleneckAutoencoder(
        in_channels=1,
        base_channels=32,
        latent_channels=args.latent_channels,
        sr_factor=args.sr_factor,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()

    history: List[Dict[str, float | int]] = []
    best_psnr = float("-inf")
    best_epoch = -1
    best_ckpt_path = ckpt_dir / "best.pt"
    final_ckpt_path = ckpt_dir / "final.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for images, _labels in train_loader:
            hr = prepare_hr_batch(images, hr_size=args.hr_size).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            recon = model(hr)
            loss = loss_fn(recon, hr)
            loss.backward()
            optimizer.step()

            bs = hr.shape[0]
            epoch_loss_sum += float(loss.item()) * bs
            epoch_count += bs

        train_loss = epoch_loss_sum / max(1, epoch_count)
        val_loss, val_metrics, sample_rows = run_validation(
            model=model,
            loader=val_loader,
            device=device,
            hr_size=args.hr_size,
            loss_fn=loss_fn,
        )

        psnr = float(val_metrics["psnr_mean"])
        ssim = float(val_metrics["ssim_mean"])
        log_entry = {
            "epoch": epoch,
            "train_l1": train_loss,
            "val_l1": val_loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
        }
        history.append(log_entry)

        if args.save_samples:
            sample_path = sample_dir / f"epoch_{epoch:03d}.png"
            save_reconstruction_grid(sample_path, sample_rows)

        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": psnr,
                    "val_ssim": ssim,
                    "args": vars(args),
                },
                best_ckpt_path,
            )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_l1={train_loss:.6f} val_l1={val_loss:.6f} "
            f"val_psnr={psnr:.4f} val_ssim={ssim:.6f}"
        )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        final_ckpt_path,
    )

    final_metrics = history[-1]
    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_psnr": best_psnr,
        "best_val_ssim": next(
            (float(h["val_ssim"]) for h in history if int(h["epoch"]) == best_epoch),
            None,
        ),
        "final_epoch": int(final_metrics["epoch"]),
        "final_train_l1": float(final_metrics["train_l1"]),
        "final_val_l1": float(final_metrics["val_l1"]),
        "final_val_psnr": float(final_metrics["val_psnr"]),
        "final_val_ssim": float(final_metrics["val_ssim"]),
        "num_train_samples": data_sizes["train"],
        "num_val_samples": data_sizes["val"],
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "stage1_2_4_electronic_baseline",
        "model": {
            "name": "ElectronicBottleneckAutoencoder",
            "in_channels": 1,
            "hr_size": args.hr_size,
            "sr_factor": args.sr_factor,
            "latent_channels": args.latent_channels,
            "latent_hw": [args.hr_size // args.sr_factor, args.hr_size // args.sr_factor],
            "no_hr_skip": True,
            "output_range": "[0,1] via atan mapping",
        },
        "train_args": vars(args),
        "device": str(device),
        "data": {
            "dataset": "torchvision.datasets.EMNIST",
            "emnist_split": args.emnist_split,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "construction": (
                "EMNIST train subset -> dataset transform resize to HR -> "
                "encoder bottleneck latent -> decoder reconstruction"
            ),
            "total_selected": data_sizes["total"],
            "train_samples": data_sizes["train"],
            "val_samples": data_sizes["val"],
            "val_ratio": args.val_ratio,
            "overfit_single_sample": args.overfit_single_sample,
        },
        "history": history,
        "artifacts": {
            "metrics_json": str(metrics_path.resolve()),
            "best_checkpoint": str(best_ckpt_path.resolve()),
            "final_checkpoint": str(final_ckpt_path.resolve()),
            "sample_dir": str(sample_dir.resolve()) if args.save_samples else None,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "best_checkpoint": best_ckpt_path,
        "final_checkpoint": final_ckpt_path,
        "metrics": metrics_payload,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    args = resolve_runtime_args(args, cfg)
    result = train(args)
    print("Training finished.")
    print(f"best checkpoint: {result['best_checkpoint']}")
    print(f"final checkpoint: {result['final_checkpoint']}")
    print(f"metrics: {result['metrics_path']}")
    print(f"summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
