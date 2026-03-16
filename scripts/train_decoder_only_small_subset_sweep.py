#!/usr/bin/env python
"""Fixed-protocol L=1/3/5 sweep for Stage 3 Issue 6."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# Keep `python scripts/train_decoder_only_small_subset_sweep.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_decoder_only_small_subset import normalize_device, run_subset_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 Issue 6 fixed-protocol decoder-only small-subset depth sweep."
    )
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr-phase", type=float, default=0.15)
    parser.add_argument("--lr-decoder", type=float, default=0.03)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/optics/decoder_only_small_subset_sweep",
    )
    return parser.parse_args()


def validate_depths(depths: list[int]) -> list[int]:
    allowed = {1, 3, 5}
    cleaned: list[int] = []
    for depth in depths:
        if depth not in allowed:
            raise ValueError(f"Unsupported depth {depth}. Expected a subset of [1, 3, 5].")
        if depth not in cleaned:
            cleaned.append(depth)
    if not cleaned:
        raise ValueError("At least one depth must be provided.")
    return cleaned


def write_depth_comparison_csv(output_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "depth",
        "subset_size",
        "steps",
        "success_count",
        "total_count",
        "success_rate",
        "mean_initial_loss",
        "mean_best_loss",
        "mean_loss_decrease",
        "summary_file",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    depths = validate_depths(list(args.depths))
    device = normalize_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_rows: list[dict[str, Any]] = []
    depth_summaries: list[dict[str, Any]] = []

    shared_protocol = {
        "depths": depths,
        "subset_size": args.subset_size,
        "steps": args.steps,
        "seed": args.seed,
        "device": str(device),
        "lr_phase": args.lr_phase,
        "lr_decoder": args.lr_decoder,
        "freeze_decoder": args.freeze_decoder,
        "loss_name": "normalized_mae_roi",
    }

    for depth in depths:
        depth_dir = output_dir / f"L{depth}"
        subset_summary = run_subset_validation(
            depth=depth,
            subset_size=args.subset_size,
            steps=args.steps,
            seed=args.seed,
            device=device,
            lr_phase=args.lr_phase,
            lr_decoder=args.lr_decoder,
            freeze_decoder=args.freeze_decoder,
            output_dir=depth_dir,
        )
        summary_file = str((depth_dir / "subset_summary.json").resolve())
        depth_summary = {
            "depth": subset_summary["depth"],
            "subset_size": subset_summary["subset_size"],
            "steps": subset_summary["steps"],
            "mean_initial_loss": subset_summary["mean_initial_loss"],
            "mean_best_loss": subset_summary["mean_best_loss"],
            "mean_loss_decrease": subset_summary["mean_loss_decrease"],
            "success_count": subset_summary["success_count"],
            "total_count": subset_summary["total_count"],
            "success_rate": subset_summary["success_rate"],
            "summary_file": summary_file,
        }
        depth_rows.append(depth_summary)
        depth_summaries.append(
            {
                **depth_summary,
                "protocol": subset_summary["protocol"],
                "sample_summaries": subset_summary["sample_summaries"],
            }
        )

    sweep_summary = {
        "shared_protocol": shared_protocol,
        "depth_summaries": depth_summaries,
    }
    with (output_dir / "sweep_summary.json").open("w", encoding="utf-8") as file:
        json.dump(sweep_summary, file, indent=2, ensure_ascii=False)

    write_depth_comparison_csv(output_dir / "depth_comparison.csv", depth_rows)

    print("Stage 3 Issue 6 fixed-protocol depth sweep completed.")
    print(
        f"depths={depths}, subset_size={args.subset_size}, steps={args.steps}, "
        f"freeze_decoder={args.freeze_decoder}"
    )
    for row in depth_rows:
        print(
            f"L={row['depth']}: success={row['success_count']}/{row['total_count']}, "
            f"mean_initial_loss={row['mean_initial_loss']:.6f}, "
            f"mean_best_loss={row['mean_best_loss']:.6f}"
        )
    print(f"sweep_summary={output_dir / 'sweep_summary.json'}")
    print(f"depth_comparison={output_dir / 'depth_comparison.csv'}")


if __name__ == "__main__":
    main()
