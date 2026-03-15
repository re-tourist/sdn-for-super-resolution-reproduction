"""Unified PSNR / SSIM metric helpers for Stage 1/2 evaluation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity


ArrayLike = np.ndarray | torch.Tensor


def _to_numpy_float32(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
    return arr.astype(np.float32, copy=False)


def _to_nchw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[None, None, :, :]
    if x.ndim == 3:
        # Assume [C,H,W] if first dim is channel-like; otherwise [N,H,W].
        if x.shape[0] in (1, 2, 3, 4):
            return x[None, :, :, :]
        return x[:, None, :, :]
    if x.ndim == 4:
        return x
    raise ValueError(
        f"Unsupported input shape {x.shape}. Expected [H,W], [C,H,W], [N,H,W], or [N,C,H,W]."
    )


def _normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin = float(arr.min())
    amax = float(arr.max())
    if amin >= 0.0 and amax <= 1.0:
        return arr
    if amin >= 0.0 and amax <= 255.0:
        return arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _prepare_pair(pred: ArrayLike, target: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    pred_arr = _normalize_to_unit(_to_nchw(_to_numpy_float32(pred)))
    target_arr = _normalize_to_unit(_to_nchw(_to_numpy_float32(target)))
    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_arr.shape} vs target {target_arr.shape}."
        )
    return pred_arr, target_arr


def _safe_ssim_win_size(h: int, w: int) -> int | None:
    min_hw = min(h, w)
    if min_hw < 3:
        return None
    if min_hw >= 7:
        return 7
    return min_hw if (min_hw % 2 == 1) else (min_hw - 1)


def compute_psnr(pred: ArrayLike, target: ArrayLike) -> float:
    """Compute mean PSNR over samples, with image range fixed to [0, 1]."""
    pred_arr, target_arr = _prepare_pair(pred, target)
    diff = pred_arr - target_arr
    mse_per_sample = np.mean(diff * diff, axis=(1, 2, 3), dtype=np.float64)
    with np.errstate(divide="ignore"):
        psnr_per_sample = 10.0 * np.log10(1.0 / mse_per_sample)
    return float(np.mean(psnr_per_sample))


def compute_ssim(pred: ArrayLike, target: ArrayLike) -> float:
    """Compute mean SSIM over samples using skimage.metrics.structural_similarity."""
    pred_arr, target_arr = _prepare_pair(pred, target)
    scores: list[float] = []
    for i in range(pred_arr.shape[0]):
        pred_i = pred_arr[i]
        target_i = target_arr[i]
        _, h, w = pred_i.shape
        win_size = _safe_ssim_win_size(h, w)
        if win_size is None:
            raise ValueError(
                f"SSIM requires spatial size >= 3, got H={h}, W={w} for sample {i}."
            )
        if pred_i.shape[0] == 1:
            score = structural_similarity(
                pred_i[0],
                target_i[0],
                data_range=1.0,
                win_size=win_size,
            )
        else:
            pred_hwc = np.transpose(pred_i, (1, 2, 0))
            target_hwc = np.transpose(target_i, (1, 2, 0))
            score = structural_similarity(
                pred_hwc,
                target_hwc,
                data_range=1.0,
                channel_axis=-1,
                win_size=win_size,
            )
        scores.append(float(score))
    return float(np.mean(scores))
