#!/usr/bin/env python
"""Sanity-check script for unified PSNR/SSIM evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Keep `python scripts/test_eval_pipeline.py` working from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.evaluator import evaluate_batch


def main() -> None:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    n, c, h, w = 8, 1, 96, 96
    hr = torch.rand(n, c, h, w, dtype=torch.float32)
    noise = 0.08 * torch.randn_like(hr)
    sr = torch.clamp(hr + noise, 0.0, 1.0)

    metrics = evaluate_batch(sr, hr)
    print(f"PSNR: {metrics['psnr_mean']:.2f}")
    print(f"SSIM: {metrics['ssim_mean']:.4f}")
    print(f"samples: {metrics['num_samples']}")


if __name__ == "__main__":
    main()
