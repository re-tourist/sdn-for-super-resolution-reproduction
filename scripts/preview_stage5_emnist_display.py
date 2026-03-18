#!/usr/bin/env python
"""Preview the Stage 5 paper-aligned EMNIST 96x96 display dataset."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import Stage5EMNISTDisplayDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate auditable preview artifacts for the Stage 5 EMNIST display dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="docs/plan/stage_plan/stage5/stage5_emnist_display_config.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument("--preview-count", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--download", action="store_true", default=False)
    return parser.parse_args()


def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load the Stage 5 dataset config.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level config mapping, got {type(payload)!r}.")
    return payload


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def resolve_stage5_dataset_section(config: dict[str, Any]) -> dict[str, Any]:
    data_section = get_nested(config, "data", "stage5_display", default=None)
    if not isinstance(data_section, dict):
        raise KeyError("Expected config path data.stage5_display to be present.")
    return data_section


def pick_preview_indices(
    dataset: Stage5EMNISTDisplayDataset,
    *,
    preview_count: int,
    preview_seed: int,
) -> list[int]:
    if preview_count < 1:
        raise ValueError(f"preview_count must be >= 1, got {preview_count}.")
    preview_count = min(preview_count, len(dataset))
    rng = np.random.default_rng(preview_seed)
    chosen = rng.choice(len(dataset), size=preview_count, replace=False)
    return sorted(int(index) for index in chosen.tolist())


def save_preview_grid(
    output_path: Path,
    *,
    dataset: Stage5EMNISTDisplayDataset,
    indices: list[int],
    grid_cols: int,
) -> None:
    if not indices:
        raise ValueError("Need at least one preview index to save a grid.")
    grid_cols = max(1, int(grid_cols))
    grid_rows = int(np.ceil(len(indices) / grid_cols))

    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(3.4 * grid_cols, 3.6 * grid_rows),
    )
    axes_array = np.atleast_1d(axes).reshape(grid_rows, grid_cols)

    for axis in axes_array.flat:
        axis.axis("off")

    for axis, dataset_index in zip(axes_array.flat, indices):
        sample = dataset[dataset_index]
        metadata = dataset.get_sample_metadata(dataset_index)
        image = sample["x_hr"][0].detach().cpu().numpy()
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        cell_text = ",".join(str(cell) for cell in metadata["occupied_cells"])
        axis.set_title(
            f"idx={dataset_index} count={metadata['letter_count']}\n"
            f"cells=[{cell_text}]",
            fontsize=9,
        )
        axis.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_dataset_from_config(
    stage5_cfg: dict[str, Any],
    *,
    split: str,
    download: bool,
) -> Stage5EMNISTDisplayDataset:
    sample_counts = stage5_cfg.get("sample_counts", {})
    letter_count_choices = stage5_cfg.get("letter_count_choices", {})
    augmentation = stage5_cfg.get("augmentation", {})

    return Stage5EMNISTDisplayDataset(
        split=split,
        dataset_root=stage5_cfg.get("dataset_root"),
        emnist_split=str(stage5_cfg.get("emnist_split", "letters")),
        sample_count=sample_counts.get(split),
        letter_count_choices=letter_count_choices.get(split),
        seed=int(stage5_cfg.get("seed", 2026)),
        canvas_hw=tuple(stage5_cfg.get("canvas_hw", (96, 96))),
        cell_hw=tuple(stage5_cfg.get("cell_hw", (32, 32))),
        grid_shape=tuple(stage5_cfg.get("grid_shape", (3, 3))),
        augmentation=augmentation.get(split),
        fix_emnist_orientation=bool(stage5_cfg.get("fix_emnist_orientation", True)),
        download=download,
    )


def save_split_artifacts(
    output_root: Path,
    *,
    dataset: Stage5EMNISTDisplayDataset,
    preview_indices: list[int],
    preview_seed: int,
    grid_cols: int,
) -> dict[str, Any]:
    split_dir = output_root / dataset.split
    split_dir.mkdir(parents=True, exist_ok=True)

    preview_path = split_dir / "preview_grid.png"
    save_preview_grid(
        preview_path,
        dataset=dataset,
        indices=preview_indices,
        grid_cols=grid_cols,
    )

    preview_samples = [dataset.get_sample_metadata(index) for index in preview_indices]
    summary = dataset.get_protocol_summary()
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preview_seed": preview_seed,
            "preview_indices": preview_indices,
            "preview_grid": str(preview_path.resolve()),
            "preview_samples": preview_samples,
        }
    )

    summary_path = split_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return {
        "split": dataset.split,
        "summary_path": str(summary_path.resolve()),
        "preview_path": str(preview_path.resolve()),
    }


def split_seed_offset(split: str) -> int:
    return {"train": 0, "val": 1_003, "test": 2_003}[split]


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    stage5_cfg = resolve_stage5_dataset_section(config)
    preview_cfg = stage5_cfg.get("preview", {})

    preview_count = int(args.preview_count or preview_cfg.get("preview_count", 8))
    grid_cols = int(preview_cfg.get("grid_cols", 4))
    preview_seed_offset = int(preview_cfg.get("seed_offset", 17))
    output_root = Path(args.output_root or preview_cfg.get("output_root", "outputs/stage5/dataset_preview"))

    artifact_records: list[dict[str, Any]] = []
    base_seed = int(stage5_cfg.get("seed", 2026))

    for split in args.splits:
        dataset = build_dataset_from_config(stage5_cfg, split=split, download=args.download)
        preview_seed = base_seed + preview_seed_offset + split_seed_offset(split)
        preview_indices = pick_preview_indices(
            dataset,
            preview_count=preview_count,
            preview_seed=preview_seed,
        )
        artifact_records.append(
            save_split_artifacts(
                output_root,
                dataset=dataset,
                preview_indices=preview_indices,
                preview_seed=preview_seed,
                grid_cols=grid_cols,
            )
        )

    print("Stage 5 dataset preview completed.")
    for record in artifact_records:
        print(f"{record['split']}: preview={record['preview_path']}")
        print(f"{record['split']}: summary={record['summary_path']}")


if __name__ == "__main__":
    main()

