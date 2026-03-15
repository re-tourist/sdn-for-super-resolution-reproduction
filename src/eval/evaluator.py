"""Unified evaluation entry for SR metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from src.eval.metrics import compute_psnr, compute_ssim


ArrayLike = np.ndarray | torch.Tensor


def _infer_num_samples(x: ArrayLike) -> int:
    if isinstance(x, (np.ndarray, torch.Tensor)):
        if x.ndim == 4:
            return int(x.shape[0])
        if x.ndim == 3 and int(x.shape[0]) not in (1, 2, 3, 4):
            return int(x.shape[0])
    return 1


def evaluate_batch(preds: ArrayLike, targets: ArrayLike) -> Dict[str, float | int]:
    """Evaluate a batch (or single sample) and return mean PSNR / SSIM."""
    psnr_mean = compute_psnr(preds, targets)
    ssim_mean = compute_ssim(preds, targets)
    return {
        "psnr_mean": float(psnr_mean),
        "ssim_mean": float(ssim_mean),
        "num_samples": _infer_num_samples(preds),
    }
