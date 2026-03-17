"""Stage 4 minimal EMNIST dataset path.

This module reuses the trusted EMNIST path from the electronic baseline and
defines the Stage 4 minimal data-side contract:

- `x_hr`: the single-channel HR tensor consumed by the minimal encoder
- `target_hr`: the real-image supervision source later adapted into `target_roi`

This is intentionally a small, debuggable Stage 4 data shell. It is not the
paper-final data protocol.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import EMNIST
from torchvision.transforms import InterpolationMode


DEFAULT_STAGE4_DATASET_ROOT = Path("data/raw/emnist")


def _normalize_hw(hw: int | Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(hw, int):
        if hw < 1:
            raise ValueError(f"{name} must be >= 1, got {hw}.")
        return (hw, hw)

    if not isinstance(hw, Sequence) or isinstance(hw, (str, bytes)) or len(hw) != 2:
        raise TypeError(f"{name} must be an int or a length-2 sequence, got {hw!r}.")

    height = int(hw[0])
    width = int(hw[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} values must be >= 1, got {(height, width)}.")
    return (height, width)


def resolve_stage4_dataset_root(
    dataset_root: str | Path | None = None,
    *,
    config_root: str | Path | None = None,
) -> Path:
    """Resolve the minimal Stage 4 EMNIST root without assuming a paper-final config."""
    if dataset_root is not None:
        return Path(dataset_root)

    if config_root is not None:
        cfg_root = Path(config_root)
        cfg_root_str = str(cfg_root).lower()
        looks_like_emnist_root = (
            "emnist" in cfg_root_str
            or (cfg_root / "EMNIST").exists()
            or (cfg_root / "raw").exists()
        )
        if looks_like_emnist_root:
            return cfg_root

    return DEFAULT_STAGE4_DATASET_ROOT


def _prepare_hr_tensor(
    image: torch.Tensor,
    *,
    expected_hw: tuple[int, int],
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected EMNIST transform to return a tensor, got {type(image)!r}.")

    x_hr = image.detach().clone().float()
    if x_hr.ndim == 2:
        x_hr = x_hr.unsqueeze(0)
    if x_hr.ndim != 3:
        raise ValueError(
            "Stage 4 dataset expects CHW image tensors, "
            f"got shape {tuple(x_hr.shape)}."
        )
    if x_hr.shape[0] != 1:
        raise ValueError(
            "Stage 4 minimal protocol expects single-channel HR input, "
            f"got shape {tuple(x_hr.shape)}."
        )
    if tuple(x_hr.shape[-2:]) != expected_hw:
        raise ValueError(
            "Unexpected HR shape after dataset transform, "
            f"got {tuple(x_hr.shape[-2:])}, expected {expected_hw}."
        )

    return torch.clamp(x_hr, 0.0, 1.0).contiguous()


class Stage4EMNISTDataset(Dataset[dict[str, Any]]):
    """Minimal Stage 4 dataset that yields real EMNIST HR tensors.

    Returned sample fields:
    - `x_hr`: `[1, H_hr, W_hr]`, float32, clamped to `[0, 1]`
    - `target_hr`: same image content as `x_hr`, reserved for explicit
      `target_hr -> target_roi` adaptation
    - `label`: EMNIST label for trace/debug only
    - `dataset_index`: source index inside the underlying EMNIST split

    This first Stage 4 version deliberately uses the same real image as both
    encoder input and supervision source. Later paper-aligned protocols may
    refine that contract, but this module does not assume them.
    """

    def __init__(
        self,
        *,
        dataset_root: str | Path | None = None,
        emnist_split: str = "letters",
        hr_hw: int | Sequence[int] = 96,
        train: bool = True,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_root = resolve_stage4_dataset_root(dataset_root)
        self.emnist_split = emnist_split
        self.hr_hw = _normalize_hw(hr_hw, name="hr_hw")
        self.train = bool(train)
        self.download = bool(download)

        resize_to_hr = transforms.Compose(
            [
                transforms.Resize(
                    self.hr_hw,
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )
        self.base_dataset = EMNIST(
            root=str(self.dataset_root),
            split=self.emnist_split,
            train=self.train,
            transform=resize_to_hr,
            download=self.download,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_hr, label = self.base_dataset[index]
        x_hr = _prepare_hr_tensor(image_hr, expected_hw=self.hr_hw)
        return {
            "x_hr": x_hr,
            "target_hr": x_hr.clone(),
            "label": int(label),
            "dataset_index": int(index),
        }


def build_stage4_train_val_datasets(
    *,
    dataset_root: str | Path | None = None,
    config_root: str | Path | None = None,
    emnist_split: str = "letters",
    hr_hw: int | Sequence[int] = 96,
    num_samples: int = 2000,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = False,
    overfit_single_sample: bool = False,
) -> tuple[Dataset[Any], Dataset[Any], dict[str, int | str | list[int]]]:
    """Build minimal Stage 4 train/val datasets from the trusted EMNIST path."""
    dataset = Stage4EMNISTDataset(
        dataset_root=resolve_stage4_dataset_root(dataset_root, config_root=config_root),
        emnist_split=emnist_split,
        hr_hw=hr_hw,
        train=True,
        download=download,
    )

    all_indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    if num_samples > 0:
        use_count = min(num_samples, len(all_indices))
        active_indices = all_indices[:use_count]
    else:
        active_indices = all_indices

    if not active_indices:
        raise ValueError("Need at least one Stage 4 sample from EMNIST.")

    if overfit_single_sample:
        overfit_index = active_indices[0]
        train_ds = Subset(dataset, [overfit_index])
        val_ds = Subset(dataset, [overfit_index])
        return train_ds, val_ds, {
            "train": 1,
            "val": 1,
            "total": 1,
            "hr_hw": list(dataset.hr_hw),
            "dataset_root": str(dataset.dataset_root),
        }

    if len(active_indices) < 2:
        raise ValueError("Need at least 2 samples to build Stage 4 train/val splits.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}.")

    val_count = max(1, int(len(active_indices) * val_ratio))
    if val_count >= len(active_indices):
        val_count = len(active_indices) - 1

    val_indices = active_indices[:val_count]
    train_indices = active_indices[val_count:]
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return train_ds, val_ds, {
        "train": len(train_ds),
        "val": len(val_ds),
        "total": len(active_indices),
        "hr_hw": list(dataset.hr_hw),
        "dataset_root": str(dataset.dataset_root),
    }


__all__ = [
    "DEFAULT_STAGE4_DATASET_ROOT",
    "Stage4EMNISTDataset",
    "build_stage4_train_val_datasets",
    "resolve_stage4_dataset_root",
]
