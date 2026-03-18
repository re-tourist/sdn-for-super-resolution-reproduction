"""Stage 5 paper-aligned super-resolution loss."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def _non_batch_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dims, got shape {tuple(tensor.shape)}.")
    return tuple(range(1, tensor.ndim))


def _validate_real_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.is_complex():
        raise ValueError(f"{name} must be real-valued, got dtype {tensor.dtype}.")


def _normalize_gamma_by_depth(gamma_by_depth: Mapping[int | str, float] | None) -> dict[int, float]:
    defaults = {1: 0.005, 3: 0.015, 5: 0.0}
    if gamma_by_depth is None:
        return defaults

    normalized: dict[int, float] = {}
    for key, value in gamma_by_depth.items():
        depth = int(key)
        normalized[depth] = float(value)

    missing = sorted(set(defaults) - set(normalized))
    if missing:
        raise ValueError(f"gamma_by_depth is missing required depth keys: {missing}.")
    return normalized


def _normalize_batch_power(
    value: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if isinstance(value, (int, float)):
        scalar = float(value)
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive, got {scalar}.")
        return torch.full((batch_size,), scalar, device=device, dtype=dtype)

    _validate_real_tensor(value, name=name)
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        scalar = float(tensor.item())
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive, got {scalar}.")
        return torch.full((batch_size,), scalar, device=device, dtype=dtype)

    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{name} batch dimension must match prediction batch size {batch_size}, got {tuple(tensor.shape)}."
        )
    flattened = tensor.reshape(batch_size, -1)
    if flattened.shape[1] != 1:
        raise ValueError(
            f"{name} must be an already reduced per-sample power tensor, got shape {tuple(tensor.shape)}."
        )
    reduced = flattened[:, 0]
    if (reduced <= 0.0).any():
        raise ValueError(f"{name} must be positive for every sample, got {reduced}.")
    return reduced


class Stage5SuperResolutionLoss(nn.Module):
    """Paper-aligned Stage 5 loss on the ROI path.

    Target contract:
      L = mean(|y - sigma * y_hat|) + gamma * exp(-eta)

    with:
      sigma = sum(y) / (sum(y_hat) + epsilon)
      eta = 100 * P_o / P_i
    """

    VALID_SIGMA_MODES = ("per_sample", "per_batch")

    def __init__(
        self,
        *,
        epsilon: float = 1e-8,
        sigma_mode: str = "per_sample",
        gamma_by_depth: Mapping[int | str, float] | None = None,
        enable_efficiency_term: bool = True,
        eta_scale: float = 100.0,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        sigma_mode = str(sigma_mode)
        if sigma_mode not in self.VALID_SIGMA_MODES:
            raise ValueError(
                f"sigma_mode must be one of {self.VALID_SIGMA_MODES}, got {sigma_mode!r}."
            )
        if eta_scale <= 0.0:
            raise ValueError(f"eta_scale must be positive, got {eta_scale}.")

        self.epsilon = float(epsilon)
        self.sigma_mode = sigma_mode
        self.gamma_by_depth = _normalize_gamma_by_depth(gamma_by_depth)
        self.enable_efficiency_term = bool(enable_efficiency_term)
        self.eta_scale = float(eta_scale)

    def resolve_gamma(self, depth: int) -> float:
        depth = int(depth)
        if depth not in self.gamma_by_depth:
            raise ValueError(
                f"Unsupported depth {depth}; gamma_by_depth has keys {sorted(self.gamma_by_depth)}."
            )
        return self.gamma_by_depth[depth]

    def compute_sigma(
        self,
        prediction_roi: torch.Tensor,
        target_roi: torch.Tensor,
    ) -> torch.Tensor:
        reduce_dims = _non_batch_dims(prediction_roi)
        if self.sigma_mode == "per_sample":
            return target_roi.sum(dim=reduce_dims, keepdim=True) / (
                prediction_roi.sum(dim=reduce_dims, keepdim=True) + self.epsilon
            )

        batch_scalar = target_roi.sum() / (prediction_roi.sum() + self.epsilon)
        broadcast_shape = [1] * prediction_roi.ndim
        return batch_scalar.reshape(*broadcast_shape)

    def forward(
        self,
        prediction_roi: torch.Tensor,
        target_roi: torch.Tensor,
        *,
        depth: int,
        input_power: torch.Tensor | float | None = None,
        output_power: torch.Tensor | float | None = None,
    ) -> dict[str, Any]:
        _validate_real_tensor(prediction_roi, name="prediction_roi")
        _validate_real_tensor(target_roi, name="target_roi")
        if prediction_roi.shape != target_roi.shape:
            raise ValueError(
                "prediction_roi and target_roi must match, "
                f"got {tuple(prediction_roi.shape)} vs {tuple(target_roi.shape)}."
            )

        reduce_dims = _non_batch_dims(prediction_roi)
        sigma = self.compute_sigma(prediction_roi, target_roi)
        prediction_rescaled = sigma * prediction_roi
        per_sample_mae = torch.mean(torch.abs(target_roi - prediction_rescaled), dim=reduce_dims)
        mae_term = per_sample_mae.mean()

        gamma_value = self.resolve_gamma(depth)
        gamma = prediction_roi.new_tensor(gamma_value)
        efficiency_term = prediction_roi.new_zeros(())
        eta: torch.Tensor | None = None
        output_power_tensor: torch.Tensor | None = None
        input_power_tensor: torch.Tensor | None = None

        if self.enable_efficiency_term:
            batch_size = prediction_roi.shape[0]
            if output_power is None:
                output_power_tensor = prediction_roi.sum(dim=reduce_dims)
            else:
                output_power_tensor = _normalize_batch_power(
                    output_power,
                    batch_size=batch_size,
                    device=prediction_roi.device,
                    dtype=prediction_roi.dtype,
                    name="output_power",
                )

            if input_power is None:
                raise ValueError(
                    "input_power must be provided when enable_efficiency_term=True."
                )
            input_power_tensor = _normalize_batch_power(
                input_power,
                batch_size=batch_size,
                device=prediction_roi.device,
                dtype=prediction_roi.dtype,
                name="input_power",
            )
            eta = self.eta_scale * output_power_tensor / (input_power_tensor + self.epsilon)
            efficiency_term = gamma * torch.exp(-eta).mean()

        loss = mae_term + efficiency_term
        return {
            "loss": loss,
            "mae_term": mae_term,
            "efficiency_term": efficiency_term,
            "sigma": sigma,
            "eta": eta,
            "gamma": gamma,
            "input_power": input_power_tensor,
            "output_power": output_power_tensor,
            "prediction_rescaled": prediction_rescaled,
        }

    def extra_repr(self) -> str:
        return (
            f"epsilon={self.epsilon}, "
            f"sigma_mode='{self.sigma_mode}', "
            f"enable_efficiency_term={self.enable_efficiency_term}, "
            f"eta_scale={self.eta_scale}, "
            f"gamma_by_depth={self.gamma_by_depth}"
        )


__all__ = ["Stage5SuperResolutionLoss"]

