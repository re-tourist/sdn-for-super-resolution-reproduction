"""Stage 5 paper-aligned EMNIST 96x96 display dataset.

This module finalizes the Stage 5 data protocol choices owned by Issue 5.2:

- source dataset: EMNIST letters
- base letter size: 28x28, explicitly resized to 32x32
- display canvas: fixed 96x96 image made of a 3x3 grid of non-overlapping
  32x32 cells
- train/val letter-count choices: 1, 2, 3, 4
- test letter-count choices: 6, 7, 8, 9
- deterministic generation: the same `(seed, split, index)` always produces
  the same composite sample, independent of access order

The module intentionally does not define optics, encoder, loss, trainer, or
evaluation behavior.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset
from torchvision.datasets import EMNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from src.datasets.stage4_emnist import resolve_stage4_dataset_root


DEFAULT_STAGE5_SAMPLE_COUNTS: dict[str, int] = {
    "train": 60_000,
    "val": 6_000,
    "test": 6_000,
}

DEFAULT_STAGE5_LETTER_COUNT_CHOICES: dict[str, tuple[int, ...]] = {
    "train": (1, 2, 3, 4),
    "val": (1, 2, 3, 4),
    "test": (6, 7, 8, 9),
}

DEFAULT_STAGE5_CANVAS_HW = (96, 96)
DEFAULT_STAGE5_CELL_HW = (32, 32)
DEFAULT_STAGE5_GRID_SHAPE = (3, 3)
DEFAULT_STAGE5_ORIENTATION_FIX = True
DEFAULT_STAGE5_DATASET_ROOT = Path("data/raw/emnist")

_VALID_SPLITS = ("train", "val", "test")
_VALID_FLIP_MODES = {"horizontal", "vertical"}


def _normalize_hw(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
        return (value, value)

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be an int or a length-2 tuple, got {value!r}.")

    height = int(value[0])
    width = int(value[1])
    if height < 1 or width < 1:
        raise ValueError(f"{name} values must be >= 1, got {(height, width)}.")
    return (height, width)


def _normalize_split(split: str) -> str:
    normalized = str(split).lower()
    if normalized not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}.")
    return normalized


def _normalize_letter_count_choices(
    choices: tuple[int, ...] | list[int],
    *,
    split: str,
    num_cells: int,
) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in choices)
    if not normalized:
        raise ValueError(f"{split} letter_count_choices must not be empty.")
    if any(value < 1 for value in normalized):
        raise ValueError(f"{split} letter_count_choices must be >= 1, got {normalized}.")
    if any(value > num_cells for value in normalized):
        raise ValueError(
            f"{split} letter_count_choices cannot exceed the {num_cells} available cells, "
            f"got {normalized}."
        )
    return normalized


def _stable_seed(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _fix_emnist_orientation(image_chw: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.rot90(image_chw, k=1, dims=(-2, -1)), dims=(-1,))


@dataclass(frozen=True)
class Stage5AugmentationConfig:
    enable_rotation: bool
    rotation_choices: tuple[int, ...]
    enable_flip: bool
    flip_modes: tuple[str, ...]
    enable_contrast: bool
    contrast_range: tuple[float, float]

    @classmethod
    def default_for_split(cls, split: str) -> "Stage5AugmentationConfig":
        if split == "train":
            return cls(
                enable_rotation=True,
                rotation_choices=(0, 90, 180, 270),
                enable_flip=True,
                flip_modes=("horizontal", "vertical"),
                enable_contrast=True,
                contrast_range=(0.8, 1.2),
            )

        return cls(
            enable_rotation=False,
            rotation_choices=(0,),
            enable_flip=False,
            flip_modes=(),
            enable_contrast=False,
            contrast_range=(1.0, 1.0),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        split: str,
    ) -> "Stage5AugmentationConfig":
        default = cls.default_for_split(split)
        if payload is None:
            return default

        rotation_choices = tuple(
            int(value) for value in payload.get("rotation_choices", default.rotation_choices)
        )
        if any(value not in (0, 90, 180, 270) for value in rotation_choices):
            raise ValueError(
                "rotation_choices must be drawn from {0, 90, 180, 270}, "
                f"got {rotation_choices}."
            )

        flip_modes = tuple(
            str(value).lower() for value in payload.get("flip_modes", default.flip_modes)
        )
        invalid_flip_modes = [value for value in flip_modes if value not in _VALID_FLIP_MODES]
        if invalid_flip_modes:
            raise ValueError(
                f"flip_modes must be chosen from {_VALID_FLIP_MODES}, got {invalid_flip_modes}."
            )

        contrast_range_raw = payload.get("contrast_range", default.contrast_range)
        if not isinstance(contrast_range_raw, (list, tuple)) or len(contrast_range_raw) != 2:
            raise TypeError(
                "contrast_range must be a length-2 sequence, "
                f"got {contrast_range_raw!r}."
            )
        contrast_range = (float(contrast_range_raw[0]), float(contrast_range_raw[1]))
        if contrast_range[0] <= 0.0 or contrast_range[1] <= 0.0:
            raise ValueError(f"contrast_range must be positive, got {contrast_range}.")
        if contrast_range[0] > contrast_range[1]:
            raise ValueError(f"contrast_range must be increasing, got {contrast_range}.")

        return cls(
            enable_rotation=bool(payload.get("enable_rotation", default.enable_rotation)),
            rotation_choices=rotation_choices,
            enable_flip=bool(payload.get("enable_flip", default.enable_flip)),
            flip_modes=flip_modes,
            enable_contrast=bool(payload.get("enable_contrast", default.enable_contrast)),
            contrast_range=contrast_range,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_rotation": self.enable_rotation,
            "rotation_choices": list(self.rotation_choices),
            "enable_flip": self.enable_flip,
            "flip_modes": list(self.flip_modes),
            "enable_contrast": self.enable_contrast,
            "contrast_range": list(self.contrast_range),
        }


@dataclass(frozen=True)
class LetterPlacement:
    cell_index: int
    cell_row: int
    cell_col: int
    source_index: int
    label: int
    rotation_deg: int
    flip_horizontal: bool
    flip_vertical: bool
    contrast_factor: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_index": self.cell_index,
            "cell_row": self.cell_row,
            "cell_col": self.cell_col,
            "source_index": self.source_index,
            "label": self.label,
            "rotation_deg": self.rotation_deg,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "contrast_factor": self.contrast_factor,
        }


@dataclass(frozen=True)
class DisplaySampleSpec:
    dataset_index: int
    split: str
    sample_seed: int
    letter_count: int
    occupied_cells: tuple[int, ...]
    placements: tuple[LetterPlacement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "split": self.split,
            "sample_seed": self.sample_seed,
            "letter_count": self.letter_count,
            "occupied_cells": list(self.occupied_cells),
            "placements": [placement.to_dict() for placement in self.placements],
        }


class Stage5EMNISTDisplayDataset(Dataset[dict[str, Any]]):
    """Deterministic Stage 5 EMNIST display dataset.

    Finalized tiling rule:
    - the 96x96 display is a fixed 3x3 grid of 32x32 cells
    - each occupied cell contains exactly one resized EMNIST letter
    - occupied cells are sampled without replacement per composite image
    - letters never overlap and empty cells remain zero-valued
    """

    def __init__(
        self,
        *,
        split: str,
        dataset_root: str | Path | None = None,
        emnist_split: str = "letters",
        sample_count: int | None = None,
        letter_count_choices: tuple[int, ...] | list[int] | None = None,
        seed: int = 2026,
        canvas_hw: tuple[int, int] | int = DEFAULT_STAGE5_CANVAS_HW,
        cell_hw: tuple[int, int] | int = DEFAULT_STAGE5_CELL_HW,
        grid_shape: tuple[int, int] | int = DEFAULT_STAGE5_GRID_SHAPE,
        augmentation: Stage5AugmentationConfig | Mapping[str, Any] | None = None,
        fix_emnist_orientation: bool = DEFAULT_STAGE5_ORIENTATION_FIX,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.split = _normalize_split(split)
        self.dataset_root = resolve_stage4_dataset_root(
            dataset_root or DEFAULT_STAGE5_DATASET_ROOT
        )
        self.emnist_split = emnist_split
        self.canvas_hw = _normalize_hw(canvas_hw, name="canvas_hw")
        self.cell_hw = _normalize_hw(cell_hw, name="cell_hw")
        self.grid_shape = _normalize_hw(grid_shape, name="grid_shape")
        self.seed = int(seed)
        self.fix_emnist_orientation = bool(fix_emnist_orientation)
        self.download = bool(download)
        self.sample_count = int(sample_count or DEFAULT_STAGE5_SAMPLE_COUNTS[self.split])
        if self.sample_count < 1:
            raise ValueError(f"sample_count must be >= 1, got {self.sample_count}.")

        grid_height = self.grid_shape[0] * self.cell_hw[0]
        grid_width = self.grid_shape[1] * self.cell_hw[1]
        if (grid_height, grid_width) != self.canvas_hw:
            raise ValueError(
                "canvas_hw must exactly match grid_shape * cell_hw, "
                f"got canvas_hw={self.canvas_hw}, grid_shape={self.grid_shape}, "
                f"cell_hw={self.cell_hw}."
            )

        self.num_cells = self.grid_shape[0] * self.grid_shape[1]
        self.letter_count_choices = _normalize_letter_count_choices(
            letter_count_choices or DEFAULT_STAGE5_LETTER_COUNT_CHOICES[self.split],
            split=self.split,
            num_cells=self.num_cells,
        )

        if isinstance(augmentation, Stage5AugmentationConfig):
            self.augmentation = augmentation
        else:
            self.augmentation = Stage5AugmentationConfig.from_mapping(
                augmentation,
                split=self.split,
            )

        use_train_partition = self.split in {"train", "val"}
        self.base_dataset = EMNIST(
            root=str(self.dataset_root),
            split=self.emnist_split,
            train=use_train_partition,
            download=self.download,
        )
        self.source_data = self.base_dataset.data
        self.source_targets = self.base_dataset.targets
        self.source_partition = "train" if use_train_partition else "test"
        if len(self.source_data) < max(self.letter_count_choices):
            raise ValueError(
                "Underlying EMNIST partition is smaller than the requested max letter count."
            )

        self.tiling_rule = (
            "Fixed 3x3 tiling over a 96x96 canvas: each 32x32 cell holds at most "
            "one resized letter, occupied cells are sampled without replacement, "
            "and empty cells remain zero."
        )

    def __len__(self) -> int:
        return self.sample_count

    def _sample_rng(self, index: int) -> random.Random:
        if index < 0 or index >= self.sample_count:
            raise IndexError(f"Index {index} is out of bounds for split {self.split}.")
        sample_seed = _stable_seed("stage5_emnist_display", self.seed, self.split, index)
        return random.Random(sample_seed)

    def _sample_spec(self, index: int) -> DisplaySampleSpec:
        rng = self._sample_rng(index)
        sample_seed = _stable_seed("stage5_emnist_display", self.seed, self.split, index)
        letter_count = int(rng.choice(self.letter_count_choices))
        occupied_cells = tuple(sorted(rng.sample(range(self.num_cells), k=letter_count)))
        source_indices = tuple(rng.sample(range(len(self.source_data)), k=letter_count))

        placements: list[LetterPlacement] = []
        for cell_index, source_index in zip(occupied_cells, source_indices):
            row = cell_index // self.grid_shape[1]
            col = cell_index % self.grid_shape[1]
            rotation_deg = (
                int(rng.choice(self.augmentation.rotation_choices))
                if self.augmentation.enable_rotation
                else 0
            )
            flip_horizontal = (
                self.augmentation.enable_flip
                and "horizontal" in self.augmentation.flip_modes
                and (rng.random() < 0.5)
            )
            flip_vertical = (
                self.augmentation.enable_flip
                and "vertical" in self.augmentation.flip_modes
                and (rng.random() < 0.5)
            )
            if self.augmentation.enable_contrast:
                contrast_factor = rng.uniform(*self.augmentation.contrast_range)
            else:
                contrast_factor = 1.0

            placements.append(
                LetterPlacement(
                    cell_index=int(cell_index),
                    cell_row=int(row),
                    cell_col=int(col),
                    source_index=int(source_index),
                    label=int(self.source_targets[source_index]),
                    rotation_deg=int(rotation_deg),
                    flip_horizontal=bool(flip_horizontal),
                    flip_vertical=bool(flip_vertical),
                    contrast_factor=float(contrast_factor),
                )
            )

        return DisplaySampleSpec(
            dataset_index=int(index),
            split=self.split,
            sample_seed=int(sample_seed),
            letter_count=int(letter_count),
            occupied_cells=occupied_cells,
            placements=tuple(placements),
        )

    def _load_source_letter(self, source_index: int) -> torch.Tensor:
        image = self.source_data[source_index].detach().clone().float().unsqueeze(0) / 255.0
        if self.fix_emnist_orientation:
            image = _fix_emnist_orientation(image)
        image = TF.resize(
            image,
            size=list(self.cell_hw),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return torch.clamp(image, 0.0, 1.0).contiguous()

    def _apply_augmentations(
        self,
        image: torch.Tensor,
        placement: LetterPlacement,
    ) -> torch.Tensor:
        augmented = image
        if placement.rotation_deg:
            k = placement.rotation_deg // 90
            augmented = torch.rot90(augmented, k=k, dims=(-2, -1))
        if placement.flip_horizontal:
            augmented = torch.flip(augmented, dims=(-1,))
        if placement.flip_vertical:
            augmented = torch.flip(augmented, dims=(-2,))
        if placement.contrast_factor != 1.0:
            augmented = TF.adjust_contrast(augmented, placement.contrast_factor)
        return torch.clamp(augmented, 0.0, 1.0).contiguous()

    def compose_display(self, index: int) -> torch.Tensor:
        spec = self._sample_spec(index)
        canvas = torch.zeros((1, *self.canvas_hw), dtype=torch.float32)
        cell_height, cell_width = self.cell_hw
        for placement in spec.placements:
            patch = self._load_source_letter(placement.source_index)
            patch = self._apply_augmentations(patch, placement)
            top = placement.cell_row * cell_height
            left = placement.cell_col * cell_width
            canvas[:, top : top + cell_height, left : left + cell_width] = patch
        return canvas.contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        display = self.compose_display(index)
        spec = self._sample_spec(index)
        return {
            "x_hr": display,
            "target_hr": display.clone(),
            "letter_count": spec.letter_count,
            "dataset_index": int(index),
        }

    def get_sample_metadata(self, index: int) -> dict[str, Any]:
        spec = self._sample_spec(index)
        metadata = spec.to_dict()
        metadata.update(
            {
                "source_partition": self.source_partition,
                "canvas_hw": list(self.canvas_hw),
                "cell_hw": list(self.cell_hw),
                "grid_shape": list(self.grid_shape),
            }
        )
        return metadata

    def get_protocol_summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_partition": self.source_partition,
            "dataset_root": str(Path(self.dataset_root).resolve()),
            "emnist_split": self.emnist_split,
            "sample_count": self.sample_count,
            "canvas_hw": list(self.canvas_hw),
            "cell_hw": list(self.cell_hw),
            "grid_shape": list(self.grid_shape),
            "num_cells": self.num_cells,
            "tiling_rule": self.tiling_rule,
            "letter_count_choices": list(self.letter_count_choices),
            "seed": self.seed,
            "randomization_policy": (
                "Each sample is generated from a deterministic RNG seeded by "
                "(base_seed, split, dataset_index). Generation is order-independent."
            ),
            "fix_emnist_orientation": self.fix_emnist_orientation,
            "augmentation": self.augmentation.to_dict(),
        }


def build_stage5_emnist_display_dataset(
    *,
    split: str,
    dataset_root: str | Path | None = None,
    emnist_split: str = "letters",
    sample_count: int | None = None,
    letter_count_choices: tuple[int, ...] | list[int] | None = None,
    seed: int = 2026,
    canvas_hw: tuple[int, int] | int = DEFAULT_STAGE5_CANVAS_HW,
    cell_hw: tuple[int, int] | int = DEFAULT_STAGE5_CELL_HW,
    grid_shape: tuple[int, int] | int = DEFAULT_STAGE5_GRID_SHAPE,
    augmentation: Stage5AugmentationConfig | Mapping[str, Any] | None = None,
    fix_emnist_orientation: bool = DEFAULT_STAGE5_ORIENTATION_FIX,
    download: bool = False,
) -> Stage5EMNISTDisplayDataset:
    return Stage5EMNISTDisplayDataset(
        split=split,
        dataset_root=dataset_root,
        emnist_split=emnist_split,
        sample_count=sample_count,
        letter_count_choices=letter_count_choices,
        seed=seed,
        canvas_hw=canvas_hw,
        cell_hw=cell_hw,
        grid_shape=grid_shape,
        augmentation=augmentation,
        fix_emnist_orientation=fix_emnist_orientation,
        download=download,
    )


__all__ = [
    "DEFAULT_STAGE5_CANVAS_HW",
    "DEFAULT_STAGE5_CELL_HW",
    "DEFAULT_STAGE5_DATASET_ROOT",
    "DEFAULT_STAGE5_GRID_SHAPE",
    "DEFAULT_STAGE5_LETTER_COUNT_CHOICES",
    "DEFAULT_STAGE5_SAMPLE_COUNTS",
    "DisplaySampleSpec",
    "LetterPlacement",
    "Stage5AugmentationConfig",
    "Stage5EMNISTDisplayDataset",
    "build_stage5_emnist_display_dataset",
]
