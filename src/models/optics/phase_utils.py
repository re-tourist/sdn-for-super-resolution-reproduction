"""Phase and field helpers for the Stage 3 optical decoder contract.

The upstream project exposes optics tensors through ambiguous `amp/phase`
names even when the tensors are really real/imag parts of a coherent field.
This rewrite keeps the Stage 3 contract explicit:

- `phi_*` means phase in radians.
- complex tensors represent coherent optical fields directly.
- padding / embedding rules are separate for input fields vs. phase masks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def normalize_hw(hw: int | Sequence[int], *, name: str) -> tuple[int, int]:
    """Normalize an integer or pair-like shape into `(height, width)`."""
    if isinstance(hw, int):
        if hw <= 0:
            raise ValueError(f"{name} must be positive, got {hw}.")
        return (hw, hw)

    if len(hw) != 2:
        raise ValueError(f"{name} must be an int or length-2 sequence, got {hw}.")

    height = int(hw[0])
    width = int(hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain positive values, got {hw}.")
    return (height, width)


def normalize_spacing(
    spacing: float | Sequence[float], *, name: str
) -> tuple[float, float]:
    """Normalize a scalar or pair-like sampling interval into `(dy, dx)`."""
    if isinstance(spacing, (int, float)):
        value = float(spacing)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {spacing}.")
        return (value, value)

    if len(spacing) != 2:
        raise ValueError(f"{name} must be a float or length-2 sequence, got {spacing}.")

    dy = float(spacing[0])
    dx = float(spacing[1])
    if dy <= 0.0 or dx <= 0.0:
        raise ValueError(f"{name} must contain positive values, got {spacing}.")
    return (dy, dx)


def validate_single_channel_image(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Validate a `[B, 1, H, W]` real-valued tensor."""
    if tensor.ndim != 4:
        raise ValueError(f"{name} must be 4D NCHW, got shape {tuple(tensor.shape)}.")
    if tensor.shape[1] != 1:
        raise ValueError(f"{name} must have channel size 1, got {tensor.shape[1]}.")
    if expected_hw is not None and tuple(tensor.shape[-2:]) != tuple(expected_hw):
        raise ValueError(
            f"{name} must have spatial shape {expected_hw}, got {tuple(tensor.shape[-2:])}."
        )
    if tensor.is_complex():
        raise ValueError(f"{name} must be real-valued, got dtype {tensor.dtype}.")
    return tensor


def ensure_complex_field(
    field: torch.Tensor,
    *,
    name: str,
    expected_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Validate a `[B, 1, H, W]` complex field tensor."""
    if field.ndim != 4:
        raise ValueError(f"{name} must be 4D NCHW, got shape {tuple(field.shape)}.")
    if field.shape[1] != 1:
        raise ValueError(f"{name} must have channel size 1, got {field.shape[1]}.")
    if not field.is_complex():
        raise ValueError(f"{name} must be complex-valued, got dtype {field.dtype}.")
    if expected_hw is not None and tuple(field.shape[-2:]) != tuple(expected_hw):
        raise ValueError(
            f"{name} must have spatial shape {expected_hw}, got {tuple(field.shape[-2:])}."
        )
    return field


def resolve_complex_dtype(
    dtype_or_tensor: torch.dtype | torch.Tensor,
) -> torch.dtype:
    """Resolve an explicit complex dtype from a real or complex reference."""
    dtype = dtype_or_tensor.dtype if isinstance(dtype_or_tensor, torch.Tensor) else dtype_or_tensor

    if dtype == torch.complex128:
        return torch.complex128
    if dtype in (torch.float64,):
        return torch.complex128
    if dtype in (torch.complex64, torch.float16, torch.bfloat16, torch.float32):
        return torch.complex64
    raise TypeError(f"Unsupported dtype for complex field construction: {dtype}.")


def complex_to_real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    """Return the matching real dtype for a complex dtype."""
    if complex_dtype == torch.complex128:
        return torch.float64
    if complex_dtype == torch.complex64:
        return torch.float32
    raise TypeError(f"Unsupported complex dtype: {complex_dtype}.")


def map_phase(
    phase_raw: torch.Tensor,
    *,
    mode: str = "identity",
    phase_range: tuple[float, float] = (-math.pi, math.pi),
) -> torch.Tensor:
    """Map raw phase parameters into physical phase values in radians.

    The Stage 3 docs leave the exact parameterization open, but require the
    mapping to be explicit rather than hidden inside the decoder. This helper
    keeps that rule local and configurable.
    """
    if phase_raw.is_complex():
        raise ValueError("phase_raw must be real-valued.")

    phase_min, phase_max = float(phase_range[0]), float(phase_range[1])
    if phase_max <= phase_min:
        raise ValueError(f"phase_range must satisfy max > min, got {phase_range}.")

    if mode == "identity":
        return phase_raw

    center = 0.5 * (phase_min + phase_max)
    half_span = 0.5 * (phase_max - phase_min)

    if mode == "tanh":
        return center + half_span * torch.tanh(phase_raw)
    if mode == "sigmoid":
        return phase_min + (phase_max - phase_min) * torch.sigmoid(phase_raw)

    raise ValueError(f"Unsupported phase mapping mode: {mode}.")


def phase_to_field(
    phase: torch.Tensor,
    *,
    amplitude: torch.Tensor | None = None,
    phase_mapping: str = "identity",
    phase_range: tuple[float, float] = (-math.pi, math.pi),
    complex_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a phase-only pattern into a complex coherent field.

    By Stage 3 contract the mainline uses phase-only modulation with amplitude
    fixed to 1. This helper keeps that default while allowing an explicit
    amplitude tensor for future ablations.
    """
    validate_single_channel_image(phase, name="phase")

    mapped_phase = map_phase(
        phase,
        mode=phase_mapping,
        phase_range=phase_range,
    )
    if amplitude is None:
        amplitude = torch.ones_like(mapped_phase)
    else:
        validate_single_channel_image(
            amplitude,
            name="amplitude",
            expected_hw=tuple(mapped_phase.shape[-2:]),
        )
        if amplitude.shape != mapped_phase.shape:
            raise ValueError(
                "amplitude and phase must have the same full shape, "
                f"got {tuple(amplitude.shape)} vs {tuple(mapped_phase.shape)}."
            )

    complex_dtype = complex_dtype or resolve_complex_dtype(mapped_phase)
    real_dtype = complex_to_real_dtype(complex_dtype)
    mapped_phase = mapped_phase.to(dtype=real_dtype)
    amplitude = amplitude.to(dtype=real_dtype)

    field_real = amplitude * torch.cos(mapped_phase)
    field_imag = amplitude * torch.sin(mapped_phase)
    return torch.complex(field_real, field_imag).to(dtype=complex_dtype)


def center_embed_tensor(
    tensor: torch.Tensor,
    target_hw: int | Sequence[int],
    *,
    fill_value: float | complex = 0.0,
) -> torch.Tensor:
    """Embed a tensor into a larger spatial grid with centered placement."""
    target_h, target_w = normalize_hw(target_hw, name="target_hw")
    source_h, source_w = tensor.shape[-2:]

    if source_h > target_h or source_w > target_w:
        raise ValueError(
            "Cannot center-embed tensor into a smaller target grid: "
            f"source={(source_h, source_w)}, target={(target_h, target_w)}."
        )

    if source_h == target_h and source_w == target_w:
        return tensor

    offset_h = (target_h - source_h) // 2
    offset_w = (target_w - source_w) // 2

    output_shape = (*tensor.shape[:-2], target_h, target_w)
    output = tensor.new_full(output_shape, fill_value)
    output[..., offset_h : offset_h + source_h, offset_w : offset_w + source_w] = tensor
    return output


__all__ = [
    "center_embed_tensor",
    "complex_to_real_dtype",
    "ensure_complex_field",
    "map_phase",
    "normalize_hw",
    "normalize_spacing",
    "phase_to_field",
    "resolve_complex_dtype",
    "validate_single_channel_image",
]
