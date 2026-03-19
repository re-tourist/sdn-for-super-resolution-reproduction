"""Deterministic blind line-pair target generation for Stage 5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


def _normalize_hw(hw: Sequence[int] | int, *, name: str) -> tuple[int, int]:
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


def _gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _apply_gaussian_blur(
    image: torch.Tensor,
    *,
    sigma: float,
    kernel_size: int,
) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    kernel_1d = _gaussian_kernel1d(kernel_size, sigma).to(device=image.device, dtype=image.dtype)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    blurred = F.conv2d(image.unsqueeze(0), kernel_2d, padding=padding)
    return blurred.squeeze(0)


@dataclass(frozen=True)
class BlindLinePairSpec:
    """Single deterministic blind line-pair target specification."""

    target_id: str
    orientation: str
    line_width_px: int
    gap_px: int
    margin_px: int
    line_length_px: int
    render_mode: str
    blur_sigma: float
    blur_kernel_size: int

    def to_summary(self) -> dict[str, Any]:
        return asdict(self)


def _validate_orientation(orientation: str) -> str:
    normalized = str(orientation).lower()
    if normalized not in {"horizontal", "vertical"}:
        raise ValueError(
            "orientation must be one of {'horizontal', 'vertical'}, "
            f"got {orientation!r}."
        )
    return normalized


def build_blind_line_pair_image(
    spec: BlindLinePairSpec,
    *,
    canvas_hw: tuple[int, int],
    line_value: float,
    background_value: float,
) -> torch.Tensor:
    """Render a centered line-pair target as a single-channel tensor."""
    height, width = canvas_hw
    line_value = float(line_value)
    background_value = float(background_value)
    image = torch.full((1, height, width), background_value, dtype=torch.float32)

    orientation = _validate_orientation(spec.orientation)
    line_width = int(spec.line_width_px)
    gap_px = int(spec.gap_px)
    margin_px = int(spec.margin_px)
    line_length_px = int(spec.line_length_px)

    if line_width < 1:
        raise ValueError(f"line_width_px must be >= 1, got {line_width}.")
    if gap_px < 1:
        raise ValueError(f"gap_px must be >= 1, got {gap_px}.")
    if margin_px < 0:
        raise ValueError(f"margin_px must be >= 0, got {margin_px}.")

    total_span = (2 * line_width) + gap_px
    if orientation == "horizontal":
        if line_length_px > width - (2 * margin_px):
            raise ValueError(
                "line_length_px must fit inside the horizontal canvas span, "
                f"got line_length_px={line_length_px}, canvas_hw={canvas_hw}, margin_px={margin_px}."
            )
        if total_span > height:
            raise ValueError(
                f"line pair total span {total_span} exceeds canvas height {height}."
            )
        top = (height - total_span) // 2
        left = (width - line_length_px) // 2
        image[:, top : top + line_width, left : left + line_length_px] = line_value
        lower_top = top + line_width + gap_px
        image[:, lower_top : lower_top + line_width, left : left + line_length_px] = line_value
    else:
        if line_length_px > height - (2 * margin_px):
            raise ValueError(
                "line_length_px must fit inside the vertical canvas span, "
                f"got line_length_px={line_length_px}, canvas_hw={canvas_hw}, margin_px={margin_px}."
            )
        if total_span > width:
            raise ValueError(
                f"line pair total span {total_span} exceeds canvas width {width}."
            )
        left = (width - total_span) // 2
        top = (height - line_length_px) // 2
        image[:, top : top + line_length_px, left : left + line_width] = line_value
        right_left = left + line_width + gap_px
        image[:, top : top + line_length_px, right_left : right_left + line_width] = line_value

    if spec.render_mode != "binary":
        raise ValueError(
            f"Only render_mode='binary' is currently supported, got {spec.render_mode!r}."
        )
    return torch.clamp(
        _apply_gaussian_blur(
            image,
            sigma=float(spec.blur_sigma),
            kernel_size=int(spec.blur_kernel_size),
        ),
        min=min(background_value, line_value),
        max=max(background_value, line_value),
    )


def build_stage5_blind_line_pair_targets(
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Build the full deterministic Stage 5 blind line-pair target bank."""
    canvas_hw = _normalize_hw(config["canvas_hw"], name="canvas_hw")
    render_mode = str(config.get("render_mode", "binary"))
    blur_sigma = float(config.get("blur_sigma", 0.0))
    blur_kernel_size = int(config.get("blur_kernel_size", 5))
    margin_px = int(config["margin_px"])
    line_widths_px = [int(value) for value in config["line_widths_px"]]
    gap_values_px = [int(value) for value in config["gap_px"]]
    orientations = [_validate_orientation(value) for value in config["orientations"]]

    if margin_px < 0:
        raise ValueError(f"margin_px must be >= 0, got {margin_px}.")
    if not line_widths_px:
        raise ValueError("line_widths_px must not be empty.")
    if not gap_values_px:
        raise ValueError("gap_px must not be empty.")
    if not orientations:
        raise ValueError("orientations must not be empty.")

    line_length_px = int(config.get("line_length_px", min(canvas_hw) - (2 * margin_px)))
    max_length = min(canvas_hw[0] - (2 * margin_px), canvas_hw[1] - (2 * margin_px))
    if line_length_px < 1 or line_length_px > max_length:
        raise ValueError(
            "line_length_px must be in [1, max_length], "
            f"got {line_length_px} with max_length={max_length}."
        )

    images: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for orientation in orientations:
        for line_width_px in line_widths_px:
            for gap_px in gap_values_px:
                spec = BlindLinePairSpec(
                    target_id=f"{orientation}_w{line_width_px}_g{gap_px}",
                    orientation=orientation,
                    line_width_px=line_width_px,
                    gap_px=gap_px,
                    margin_px=margin_px,
                    line_length_px=line_length_px,
                    render_mode=render_mode,
                    blur_sigma=blur_sigma,
                    blur_kernel_size=blur_kernel_size,
                )
                image = build_blind_line_pair_image(
                    spec,
                    canvas_hw=canvas_hw,
                    line_value=float(config.get("line_value", 1.0)),
                    background_value=float(config.get("background_value", 0.0)),
                )
                images.append(image)
                metadata.append(spec.to_summary())

    return torch.stack(images, dim=0), metadata


__all__ = [
    "BlindLinePairSpec",
    "build_blind_line_pair_image",
    "build_stage5_blind_line_pair_targets",
]
