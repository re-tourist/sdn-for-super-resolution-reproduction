#!/usr/bin/env python
"""Dataset inspection utility for Stage 1/2 HR-LR sanity checking."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import EMNIST


UPSAMPLE_MODES = ("bicubic", "bilinear", "nearest")


@dataclass
class SampleRecord:
    sample_rank: int
    dataset_index: int
    source_index: int
    hr_shape: List[int]
    lr_shape: List[int]
    up_shape: List[int]
    hr_min: float
    hr_max: float
    lr_min: float
    lr_max: float
    up_min: float
    up_max: float
    hr_label: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect HR/LR pairs from EMNIST with random sampling visualizations."
    )
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument(
        "--split", type=str, choices=("train", "val", "test"), default="train"
    )
    parser.add_argument(
        "--emnist-split",
        type=str,
        default="letters",
        help="Torchvision EMNIST split (byclass/bymerge/balanced/letters/digits/mnist).",
    )
    parser.add_argument(
        "--hr-size",
        type=int,
        default=96,
        help="Target HR size. Default keeps Stage1/2 expectation: 96x96.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--downsample-mode",
        type=str,
        choices=UPSAMPLE_MODES,
        default="bicubic",
    )
    parser.add_argument(
        "--upsample", type=str, choices=UPSAMPLE_MODES, default="bicubic"
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inspection/train",
    )
    parser.add_argument(
        "--save-individual",
        dest="save_individual",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-save-individual", dest="save_individual", action="store_false"
    )
    parser.add_argument(
        "--save-overview",
        dest="save_overview",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-save-overview", dest="save_overview", action="store_false"
    )
    # Backward-compatible aliases
    parser.add_argument("--save-grid", dest="save_overview", action="store_true")
    parser.add_argument("--no-save-grid", dest="save_overview", action="store_false")
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        default=True,
        help="Download EMNIST if missing.",
    )
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument(
        "--fix-emnist-orientation",
        dest="fix_emnist_orientation",
        action="store_true",
        default=False,
        help="Rotate/flip EMNIST to common upright orientation for easier inspection.",
    )
    return parser.parse_args()


def load_yaml_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with p.open("r", encoding="utf-8") as f:
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
            if looks_like_emnist_root:
                args.dataset_root = cfg_root
            elif fallback_root.exists():
                args.dataset_root = str(fallback_root)
            else:
                args.dataset_root = str(fallback_root)
        elif fallback_root.exists():
            args.dataset_root = str(fallback_root)
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

    val_count = int(len(all_indices) * args.val_ratio)
    if args.split == "val":
        split_indices = all_indices[:val_count]
    else:
        split_indices = all_indices[val_count:]

    if len(split_indices) == 0:
        raise ValueError(
            f"Split '{args.split}' is empty. Adjust --val-ratio (current: {args.val_ratio})."
        )

    return Subset(base_dataset, split_indices), split_indices


def maybe_fix_emnist_orientation(img_tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return img_tensor
    # EMNIST is often perceived rotated/flipped in visualization.
    return torch.flip(torch.rot90(img_tensor, k=1, dims=(-2, -1)), dims=(-1,))


def extract_image_and_label(sample: Any) -> Tuple[torch.Tensor, int | None]:
    label: int | None = None
    img: Any
    if isinstance(sample, dict):
        if "hr" in sample:
            img = sample["hr"]
        elif "image" in sample:
            img = sample["image"]
        elif "img" in sample:
            img = sample["img"]
        else:
            img = next(iter(sample.values()))
        if "label" in sample:
            label = int(sample["label"])
    elif isinstance(sample, (tuple, list)):
        img = sample[0]
        if len(sample) > 1 and isinstance(sample[1], (int, np.integer, torch.Tensor)):
            label_raw = sample[1]
            if isinstance(label_raw, torch.Tensor):
                label = int(label_raw.item())
            else:
                label = int(label_raw)
    else:
        img = sample

    return as_chw_tensor(img), label


def as_chw_tensor(image: Any) -> torch.Tensor:
    if isinstance(image, Image.Image):
        return transforms.ToTensor()(image)
    if isinstance(image, np.ndarray):
        arr = torch.from_numpy(image)
        if arr.ndim == 2:
            arr = arr.unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
            arr = arr.permute(2, 0, 1)
        return arr.float()
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] not in (1, 3, 4) and t.shape[-1] in (1, 3, 4):
            t = t.permute(2, 0, 1)
        return t
    raise TypeError(f"Unsupported image type for visualization: {type(image)}")


def interpolate_tensor(
    image_chw: torch.Tensor, size: Tuple[int, int], mode: str
) -> torch.Tensor:
    image_nchw = image_chw.unsqueeze(0)
    kwargs: Dict[str, Any] = {"size": size, "mode": mode}
    if mode in ("bicubic", "bilinear"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = True
    return F.interpolate(image_nchw, **kwargs).squeeze(0)


def tensor_to_display(image_chw: torch.Tensor) -> Tuple[np.ndarray, bool]:
    if image_chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(image_chw.shape)}")
    c, _, _ = image_chw.shape
    if c == 1:
        return image_chw[0].numpy(), True
    if c in (3, 4):
        return image_chw.permute(1, 2, 0).numpy(), False
    # Fallback: use first channel as grayscale.
    return image_chw[0].numpy(), True


def min_max_str(image: torch.Tensor) -> str:
    return f"[{float(image.min()):.3f}, {float(image.max()):.3f}]"


def save_triplet_figure(
    output_path: Path,
    dataset_index: int,
    source_index: int,
    label: int | None,
    hr: torch.Tensor,
    lr: torch.Tensor,
    up: torch.Tensor,
    upsample_mode: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vis_items = [("HR", hr), ("LR", lr), (f"Upsampled ({upsample_mode})", up)]

    global_min = float(min(hr.min(), lr.min(), up.min()))
    global_max = float(max(hr.max(), lr.max(), up.max()))
    if np.isclose(global_min, global_max):
        global_min, global_max = 0.0, 1.0

    for ax, (name, tensor) in zip(axes, vis_items):
        arr, is_gray = tensor_to_display(tensor)
        if is_gray:
            ax.imshow(arr, cmap="gray", vmin=global_min, vmax=global_max)
        else:
            clipped = np.clip(arr, global_min, global_max)
            denom = max(global_max - global_min, 1e-8)
            ax.imshow((clipped - global_min) / denom)
        ax.set_title(
            f"{name}\nshape={tuple(tensor.shape)}\nmin/max={min_max_str(tensor)}", fontsize=9
        )
        ax.axis("off")

    suffix = f", label={label}" if label is not None else ""
    fig.suptitle(
        f"sample_idx={dataset_index}, source_idx={source_index}{suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_grid_figure(
    output_path: Path,
    records: List[SampleRecord],
    triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    upsample_mode: str,
) -> None:
    rows = len(records)
    fig, axes = plt.subplots(rows, 3, figsize=(12, max(3.2 * rows, 4.0)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (record, (hr, lr, up)) in enumerate(zip(records, triplets)):
        row_min = float(min(hr.min(), lr.min(), up.min()))
        row_max = float(max(hr.max(), lr.max(), up.max()))
        if np.isclose(row_min, row_max):
            row_min, row_max = 0.0, 1.0

        for col, (name, tensor) in enumerate(
            (("HR", hr), ("LR", lr), (f"Upsampled ({upsample_mode})", up))
        ):
            ax = axes[row, col]
            arr, is_gray = tensor_to_display(tensor)
            if is_gray:
                ax.imshow(arr, cmap="gray", vmin=row_min, vmax=row_max)
            else:
                clipped = np.clip(arr, row_min, row_max)
                denom = max(row_max - row_min, 1e-8)
                ax.imshow((clipped - row_min) / denom)

            if col == 0:
                ax.set_ylabel(
                    (
                        f"idx={record.dataset_index}\n"
                        f"src={record.source_index}\n"
                        f"min/max={record.hr_min:.3f}/{record.hr_max:.3f}"
                    ),
                    fontsize=8,
                )
            if row == 0:
                ax.set_title(name, fontsize=10)
            ax.axis("off")

    fig.suptitle("Dataset Inspection Grid: HR / LR / Upsampled LR", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_inspection(args: argparse.Namespace) -> Dict[str, Any]:
    if args.hr_size < 1:
        raise ValueError("--hr-size must be >= 1")
    if args.scale < 1:
        raise ValueError("--scale must be >= 1")
    if args.hr_size % args.scale != 0:
        raise ValueError(
            f"--hr-size ({args.hr_size}) must be divisible by --scale ({args.scale})."
        )

    dataset, source_indices = build_emnist_dataset(args)
    dataset_len = len(dataset)

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    if dataset_len < args.num_samples:
        raise ValueError(
            f"Requested {args.num_samples} samples, but split '{args.split}' only has {dataset_len}."
        )

    rng = random.Random(args.seed)
    chosen_dataset_indices = sorted(rng.sample(range(dataset_len), k=args.num_samples))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[SampleRecord] = []
    triplets: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for rank, dataset_index in enumerate(chosen_dataset_indices, start=1):
        sample = dataset[dataset_index]
        hr_raw, label = extract_image_and_label(sample)
        hr = maybe_fix_emnist_orientation(hr_raw, enabled=args.fix_emnist_orientation).float()
        if hr.ndim != 3:
            raise ValueError(f"Expected HR tensor with shape [C,H,W], got {tuple(hr.shape)}")
        if hr.shape[-2:] != (args.hr_size, args.hr_size):
            hr = interpolate_tensor(
                hr, size=(args.hr_size, args.hr_size), mode="bicubic"
            )

        lr_size = args.hr_size // args.scale
        lr = interpolate_tensor(hr, size=(lr_size, lr_size), mode=args.downsample_mode)
        up = interpolate_tensor(lr, size=(args.hr_size, args.hr_size), mode=args.upsample)

        source_index = int(source_indices[dataset_index])
        record = SampleRecord(
            sample_rank=rank,
            dataset_index=dataset_index,
            source_index=source_index,
            hr_shape=list(hr.shape),
            lr_shape=list(lr.shape),
            up_shape=list(up.shape),
            hr_min=float(hr.min()),
            hr_max=float(hr.max()),
            lr_min=float(lr.min()),
            lr_max=float(lr.max()),
            up_min=float(up.min()),
            up_max=float(up.max()),
            hr_label=label,
        )
        records.append(record)
        triplets.append((hr, lr, up))

        if args.save_individual:
            out_file = output_dir / f"sample_{rank:03d}_idx_{dataset_index:06d}.png"
            save_triplet_figure(
                output_path=out_file,
                dataset_index=dataset_index,
                source_index=source_index,
                label=label,
                hr=hr,
                lr=lr,
                up=up,
                upsample_mode=args.upsample,
            )

    overview_path = None
    if args.save_overview:
        overview_path = output_dir / "overview_grid.png"
        save_grid_figure(
            output_path=overview_path,
            records=records,
            triplets=triplets,
            upsample_mode=args.upsample,
        )

    summary: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "emnist_split": args.emnist_split,
        "hr_size": args.hr_size,
        "scale": args.scale,
        "lr_size": args.hr_size // args.scale,
        "downsample_mode": args.downsample_mode,
        "upsample_mode": args.upsample,
        "val_ratio": args.val_ratio,
        "save_individual": args.save_individual,
        "save_overview": args.save_overview,
        "fix_emnist_orientation": args.fix_emnist_orientation,
        "overview_file": str(overview_path.resolve()) if overview_path else None,
        "samples": [record.__dict__ for record in records],
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    args = resolve_runtime_args(args, cfg)
    summary = run_inspection(args)

    print("Inspection completed.")
    print(f"split={summary['split']}, num_samples={summary['num_samples']}, seed={summary['seed']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"upsample_mode={summary['upsample_mode']}, downsample_mode={summary['downsample_mode']}")
    print(f"summary={Path(summary['output_dir']) / 'summary.json'}")


if __name__ == "__main__":
    main()
