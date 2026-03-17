"""Stage 4 minimal ROI target adapter.

This module makes the `target_hr -> target_roi` logic explicit so the Stage 4
training path can supervise `I_out_roi` without hiding resize/crop logic in a
training script.

The current adapter is deliberately minimal and non-paper-final:

- input is a real HR target tensor
- output is a real ROI-aligned supervision tensor
- adaptation is explicit resize + clamp, not an implicit or learned head
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.optics.readout import ReadoutConfig


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


def _to_gray_numpy(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().cpu().float()
    if tensor.ndim == 2:
        return tensor.numpy()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].numpy()
    raise ValueError(f"Expected grayscale image tensor, got shape {tuple(tensor.shape)}.")


class Stage4RoiTargetAdapter:
    """Adapt real HR targets into ROI-aligned supervision tensors.

    The Stage 3 optical decoder already owns:
    `phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

    This adapter only defines how a real target image is mapped into a
    supervision tensor aligned with `I_out_roi`. It does not change the optical
    contract and it does not implement a loss.
    """

    def __init__(
        self,
        *,
        output_crop_hw: int | Sequence[int] | ReadoutConfig,
        resize_mode: str = "bicubic",
        align_corners: bool = False,
        clamp_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        if isinstance(output_crop_hw, ReadoutConfig):
            self.output_crop_hw = output_crop_hw.output_crop_hw
        else:
            self.output_crop_hw = _normalize_hw(output_crop_hw, name="output_crop_hw")

        clamp_min = float(clamp_range[0])
        clamp_max = float(clamp_range[1])
        if not clamp_min <= clamp_max:
            raise ValueError(f"clamp_range must be increasing, got {clamp_range}.")

        self.resize_mode = resize_mode
        self.align_corners = bool(align_corners)
        self.clamp_range = (clamp_min, clamp_max)

    def _validate_target_hr(self, target_hr: torch.Tensor) -> None:
        if not isinstance(target_hr, torch.Tensor):
            raise TypeError(f"target_hr must be a torch.Tensor, got {type(target_hr)!r}.")
        if target_hr.ndim != 4:
            raise ValueError(
                "target_hr must have shape [B, 1, H, W], "
                f"got {tuple(target_hr.shape)}."
            )
        if target_hr.shape[1] != 1:
            raise ValueError(
                "Stage 4 minimal target adapter expects single-channel targets, "
                f"got {tuple(target_hr.shape)}."
            )

    def __call__(self, target_hr: torch.Tensor) -> torch.Tensor:
        self._validate_target_hr(target_hr)
        target_float = target_hr.float()

        resize_kwargs: dict[str, object] = {
            "size": self.output_crop_hw,
            "mode": self.resize_mode,
        }
        if self.resize_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            resize_kwargs["align_corners"] = self.align_corners
        if self.resize_mode in {"bilinear", "bicubic"}:
            resize_kwargs["antialias"] = True

        target_roi = F.interpolate(target_float, **resize_kwargs)
        clamp_min, clamp_max = self.clamp_range
        return torch.clamp(target_roi, min=clamp_min, max=clamp_max).contiguous()

    def extra_repr(self) -> str:
        return (
            f"output_crop_hw={self.output_crop_hw}, "
            f"resize_mode='{self.resize_mode}', "
            f"clamp_range={self.clamp_range}"
        )


@torch.no_grad()
def save_stage4_input_target_preview(
    output_path: str | Path,
    *,
    x_hr: torch.Tensor,
    target_roi: torch.Tensor,
    max_rows: int = 4,
) -> Path:
    """Save a small preview grid for Stage 4 input/target sanity inspection."""
    if x_hr.ndim != 4 or x_hr.shape[1] != 1:
        raise ValueError(f"x_hr must have shape [B,1,H,W], got {tuple(x_hr.shape)}.")
    if target_roi.ndim != 4 or target_roi.shape[1] != 1:
        raise ValueError(
            f"target_roi must have shape [B,1,H,W], got {tuple(target_roi.shape)}."
        )
    if x_hr.shape[0] != target_roi.shape[0]:
        raise ValueError(
            "x_hr and target_roi batch sizes must match, "
            f"got {x_hr.shape[0]} vs {target_roi.shape[0]}."
        )

    row_count = min(max_rows, x_hr.shape[0])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_roi_up = F.interpolate(
        target_roi[:row_count],
        size=tuple(x_hr.shape[-2:]),
        mode="nearest",
    )

    fig, axes = plt.subplots(row_count, 3, figsize=(9, max(3.0, 3.0 * row_count)))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ("x_hr", "target_roi", "target_roi (upsampled)")
    for row_idx in range(row_count):
        panels = (
            _to_gray_numpy(x_hr[row_idx]),
            _to_gray_numpy(target_roi[row_idx]),
            _to_gray_numpy(target_roi_up[row_idx]),
        )
        ranges = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        )
        for col_idx, ax in enumerate(axes[row_idx]):
            vmin, vmax = ranges[col_idx]
            ax.imshow(panels[col_idx], cmap="gray", vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(titles[col_idx])
            ax.axis("off")
        axes[row_idx, 0].set_ylabel(f"sample {row_idx + 1}", rotation=90, labelpad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


__all__ = [
    "Stage4RoiTargetAdapter",
    "save_stage4_input_target_preview",
]
