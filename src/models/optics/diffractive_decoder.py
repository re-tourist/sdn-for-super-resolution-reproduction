"""Paper-aligned diffractive decoder skeleton for Stage 3 Issues 2 and 3.

This module now implements the frozen Stage 3 front-half optical contract:

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`

The scheduling remains explicit and config-driven:

- `L` means the number of trainable diffractive phase masks.
- propagation from the last mask to the sensor plane is explicit.
- there is no hidden `layer == 2` or `depth + 1` logic.

Optical loss / metric computation still belongs to later Stage 3 issues.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.optics.phase_provider import PhaseProvider
from src.models.optics.phase_utils import (
    center_embed_tensor,
    ensure_complex_field,
    map_phase,
    normalize_hw,
    normalize_spacing,
    phase_to_field,
    validate_single_channel_image,
)
from src.models.optics.propagation import PropagationOperator, apply_phase_modulation
from src.models.optics.readout import ReadoutConfig, center_crop_2d, intensity_readout


@dataclass(frozen=True)
class DistanceSchedule:
    """Explicit Stage 3 propagation distances."""

    input_to_first: float
    inter_layer: tuple[float, ...]
    last_to_sensor: float

    @classmethod
    def from_config(cls, config: "DistanceSchedule | Mapping[str, object]") -> "DistanceSchedule":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("distance_schedule must be a DistanceSchedule or a mapping.")

        inter_layer_raw = config.get("inter_layer", ())
        if not isinstance(inter_layer_raw, Sequence) or isinstance(inter_layer_raw, (str, bytes)):
            raise TypeError("distance_schedule.inter_layer must be a sequence of distances.")

        return cls(
            input_to_first=float(config["input_to_first"]),
            inter_layer=tuple(float(distance) for distance in inter_layer_raw),
            last_to_sensor=float(config["last_to_sensor"]),
        )

    def validate_for_layers(self, num_diffractive_layers: int) -> None:
        if num_diffractive_layers < 1:
            raise ValueError("num_diffractive_layers must be >= 1.")
        expected_inter_layers = max(0, num_diffractive_layers - 1)
        if len(self.inter_layer) != expected_inter_layers:
            raise ValueError(
                "distance_schedule.inter_layer must contain exactly "
                f"{expected_inter_layers} entries for L={num_diffractive_layers}, "
                f"got {len(self.inter_layer)}."
            )
        all_distances = (self.input_to_first, *self.inter_layer, self.last_to_sensor)
        if any(distance <= 0.0 for distance in all_distances):
            raise ValueError(f"All propagation distances must be positive, got {all_distances}.")


@dataclass(frozen=True)
class OpticalGridConfig:
    """Grid and padding contract used by the Stage 3 optical decoder."""

    input_pattern_hw: tuple[int, int]
    layer_hw: tuple[int, int]
    propagation_hw: tuple[int, int]

    @classmethod
    def from_config(cls, config: "OpticalGridConfig | Mapping[str, object]") -> "OpticalGridConfig":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("grid_config must be an OpticalGridConfig or a mapping.")

        return cls(
            input_pattern_hw=normalize_hw(config["input_pattern_hw"], name="input_pattern_hw"),
            layer_hw=normalize_hw(config["layer_hw"], name="layer_hw"),
            propagation_hw=normalize_hw(config["propagation_hw"], name="propagation_hw"),
        )

    def __post_init__(self) -> None:
        input_h, input_w = self.input_pattern_hw
        layer_h, layer_w = self.layer_hw
        prop_h, prop_w = self.propagation_hw

        if input_h > prop_h or input_w > prop_w:
            raise ValueError(
                "input_pattern_hw must fit inside propagation_hw, "
                f"got input={self.input_pattern_hw}, propagation={self.propagation_hw}."
            )
        if layer_h > prop_h or layer_w > prop_w:
            raise ValueError(
                "layer_hw must fit inside propagation_hw, "
                f"got layer={self.layer_hw}, propagation={self.propagation_hw}."
            )

    @property
    def input_padding_hw(self) -> tuple[int, int]:
        return (
            self.propagation_hw[0] - self.input_pattern_hw[0],
            self.propagation_hw[1] - self.input_pattern_hw[1],
        )

    @property
    def layer_padding_hw(self) -> tuple[int, int]:
        return (
            self.propagation_hw[0] - self.layer_hw[0],
            self.propagation_hw[1] - self.layer_hw[1],
        )


@dataclass(frozen=True)
class PhaseMaskConfig:
    """Explicit phase-mask parameterization for trainable diffractive layers."""

    init_mode: str = "zeros"
    init_scale: float = 0.05
    phase_mapping: str = "tanh"
    phase_range: tuple[float, float] = (-math.pi, math.pi)

    @classmethod
    def from_config(cls, config: "PhaseMaskConfig | Mapping[str, object] | None") -> "PhaseMaskConfig":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("phase_mask_config must be a PhaseMaskConfig, mapping, or None.")

        phase_range_raw = config.get("phase_range", (-math.pi, math.pi))
        if not isinstance(phase_range_raw, Sequence) or len(phase_range_raw) != 2:
            raise TypeError("phase_mask_config.phase_range must be a length-2 sequence.")

        return cls(
            init_mode=str(config.get("init_mode", "zeros")),
            init_scale=float(config.get("init_scale", 0.05)),
            phase_mapping=str(config.get("phase_mapping", "tanh")),
            phase_range=(float(phase_range_raw[0]), float(phase_range_raw[1])),
        )


class DiffractiveDecoder(nn.Module):
    """Stage 3 optical decoder skeleton with explicit layer semantics.

    Stable Stage 3 input entrypoints:

    - `forward_from_phase(phi_lr, ...)`
      The caller provides a real-valued phase tensor. The decoder owns the
      phase-to-field mapping and then runs the full optical/readout chain.
    - `forward_from_field(U0, ...)`
      The caller provides an already constructed coherent complex field on the
      propagation grid. The decoder skips phase mapping and starts from optics.

    Future Stage 4 should plug in above `forward_from_phase(...)` by producing
    a phase tensor that satisfies the same contract. The optical core itself
    remains responsible only for consuming phase or field, not generating it.
    """

    VALID_LAYER_COUNTS = (1, 3, 5)

    def __init__(
        self,
        *,
        num_diffractive_layers: int,
        wavelength: float,
        pixel_pitch: float | Sequence[float],
        grid_config: OpticalGridConfig | Mapping[str, object],
        distance_schedule: DistanceSchedule | Mapping[str, object],
        readout_config: ReadoutConfig | Mapping[str, object],
        phase_mask_config: PhaseMaskConfig | Mapping[str, object] | None = None,
        kernel_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        if num_diffractive_layers not in self.VALID_LAYER_COUNTS:
            raise ValueError(
                "Stage 3 currently supports only L in {1, 3, 5}, "
                f"got L={num_diffractive_layers}."
            )
        if wavelength <= 0.0:
            raise ValueError(f"wavelength must be positive, got {wavelength}.")

        self.num_diffractive_layers = int(num_diffractive_layers)
        self.wavelength = float(wavelength)
        self.pixel_pitch = normalize_spacing(pixel_pitch, name="pixel_pitch")
        self.grid_config = OpticalGridConfig.from_config(grid_config)
        self.distance_schedule = DistanceSchedule.from_config(distance_schedule)
        self.readout_config = ReadoutConfig.from_config(readout_config)
        self.phase_mask_config = PhaseMaskConfig.from_config(phase_mask_config)
        self.distance_schedule.validate_for_layers(self.num_diffractive_layers)

        self.phase_masks_raw = nn.Parameter(
            torch.empty(
                self.num_diffractive_layers,
                1,
                *self.grid_config.layer_hw,
                dtype=torch.float32,
            )
        )
        self._reset_phase_masks()

        self.input_to_first = PropagationOperator(
            self.grid_config.propagation_hw,
            pixel_pitch=self.pixel_pitch,
            wavelength=self.wavelength,
            distance=self.distance_schedule.input_to_first,
            kernel_dtype=kernel_dtype,
        )
        self.inter_layer = nn.ModuleList(
            PropagationOperator(
                self.grid_config.propagation_hw,
                pixel_pitch=self.pixel_pitch,
                wavelength=self.wavelength,
                distance=distance,
                kernel_dtype=kernel_dtype,
            )
            for distance in self.distance_schedule.inter_layer
        )
        self.last_to_sensor = PropagationOperator(
            self.grid_config.propagation_hw,
            pixel_pitch=self.pixel_pitch,
            wavelength=self.wavelength,
            distance=self.distance_schedule.last_to_sensor,
            kernel_dtype=kernel_dtype,
        )

    def _reset_phase_masks(self) -> None:
        mode = self.phase_mask_config.init_mode
        scale = abs(self.phase_mask_config.init_scale)

        with torch.no_grad():
            if mode == "zeros":
                self.phase_masks_raw.zero_()
            elif mode == "uniform_small":
                self.phase_masks_raw.uniform_(-scale, scale)
            elif mode == "normal_small":
                self.phase_masks_raw.normal_(mean=0.0, std=scale)
            else:
                raise ValueError(f"Unsupported phase-mask init mode: {mode}.")

    def _mapped_phase_masks(self) -> torch.Tensor:
        return map_phase(
            self.phase_masks_raw,
            mode=self.phase_mask_config.phase_mapping,
            phase_range=self.phase_mask_config.phase_range,
        )

    def _embedded_phase_masks(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mapped_masks = self._mapped_phase_masks().to(device=device, dtype=dtype)
        return center_embed_tensor(
            mapped_masks,
            self.grid_config.propagation_hw,
            fill_value=0.0,
        )

    def build_input_field(
        self,
        phi_lr: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert a phase-domain tensor into `U0` on the propagation grid.

        Args:
            phi_lr: Real-valued phase tensor shaped `[B, 1, H, W]`, where
                `H, W == grid_config.input_pattern_hw`. Batch dimension is
                allowed and preserved. The tensor may live on any device, and
                `forward_from_phase(...)` keeps computation on that device.
            amplitude: Optional real-valued amplitude tensor with the same
                shape/device as `phi_lr`. Stage 3 mainline keeps this `None`,
                which means amplitude is fixed to 1 everywhere.

        Returns:
            Complex coherent input field `U0` shaped
            `[B, 1, propagation_h, propagation_w]`.

        Contract notes:
            - `phi_lr` is interpreted as a phase-domain tensor, not as a field.
            - The decoder owns phase-to-field construction for this path.
            - The default Stage 3 phase mapping responsibility is:
              caller/provider supplies phase, decoder maps phase to a coherent
              field via `exp(j * phi)` with unit amplitude unless an explicit
              amplitude tensor is supplied.
        """
        validate_single_channel_image(
            phi_lr,
            name="phi_lr",
            expected_hw=self.grid_config.input_pattern_hw,
        )
        if amplitude is not None:
            validate_single_channel_image(
                amplitude,
                name="amplitude",
                expected_hw=self.grid_config.input_pattern_hw,
            )
            if amplitude.shape != phi_lr.shape:
                raise ValueError(
                    "amplitude and phi_lr must have identical shapes, "
                    f"got {tuple(amplitude.shape)} vs {tuple(phi_lr.shape)}."
                )

        input_field = phase_to_field(
            phi_lr,
            amplitude=amplitude,
            phase_mapping="identity",
        )
        return center_embed_tensor(
            input_field,
            self.grid_config.propagation_hw,
            fill_value=0.0,
        )

    def _propagate_from_field(
        self,
        U0: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run `U0 -> U_out_full` with explicit Stage 3 propagation semantics."""
        ensure_complex_field(
            U0,
            name="U0",
            expected_hw=self.grid_config.propagation_hw,
        )

        propagation_device = U0.device
        propagation_dtype = U0.real.dtype
        phase_masks_full = self._embedded_phase_masks(
            dtype=propagation_dtype,
            device=propagation_device,
        )

        field_after_input_to_first = self.input_to_first(U0)
        field = field_after_input_to_first
        fields_after_mask_modulation: list[torch.Tensor] = []
        fields_after_inter_layer: list[torch.Tensor] = []

        for layer_index in range(self.num_diffractive_layers):
            field = apply_phase_modulation(field, phase_masks_full[layer_index : layer_index + 1])
            if return_intermediates:
                fields_after_mask_modulation.append(field)

            if layer_index < self.num_diffractive_layers - 1:
                field = self.inter_layer[layer_index](field)
                if return_intermediates:
                    fields_after_inter_layer.append(field)

        U_out_full = self.last_to_sensor(field)
        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "U_out_full": U_out_full,
        }
        if return_intermediates:
            output.update(
                {
                    "field_after_input_to_first": field_after_input_to_first,
                    "phase_masks_full": phase_masks_full,
                    "fields_after_mask_modulation": fields_after_mask_modulation,
                    "fields_after_inter_layer": fields_after_inter_layer,
                }
            )
        return output

    def readout_from_field(self, U_out_full: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert full-grid output field into full-grid and ROI intensities."""
        ensure_complex_field(
            U_out_full,
            name="U_out_full",
            expected_hw=self.grid_config.propagation_hw,
        )

        I_out_full = intensity_readout(U_out_full)
        I_out_roi = center_crop_2d(I_out_full, self.readout_config)
        return {
            "I_out_full": I_out_full,
            "I_out_roi": I_out_roi,
        }

    def forward_from_field(
        self,
        U0: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run `U0 -> U_out_full -> I_out_full -> I_out_roi`.

        Args:
            U0: Complex coherent field shaped
                `[B, 1, propagation_h, propagation_w]`. This path requires a
                complex tensor and skips all phase-domain mapping logic.
            return_intermediates: Whether to include propagation-stage tensors
                used by the existing Stage 3 debugging scripts.

        Returns:
            A mapping containing at least:
            - `U_out_full`: complex field on the sensor-plane full grid
            - `I_out_full`: real nonnegative full-grid intensity
            - `I_out_roi`: real nonnegative center-cropped ROI intensity

        Responsibility boundary:
            `forward_from_field(...)` assumes the caller has already built a
            physically meaningful coherent field. Unlike
            `forward_from_phase(...)`, it does not interpret radians or perform
            phase-to-field conversion.
        """
        propagation_output = self._propagate_from_field(
            U0,
            return_intermediates=return_intermediates,
        )
        U_out_full = propagation_output["U_out_full"]
        if not isinstance(U_out_full, torch.Tensor):
            raise RuntimeError("U_out_full must be a tensor.")

        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "U_out_full": U_out_full,
            **self.readout_from_field(U_out_full),
        }
        if return_intermediates:
            output.update(propagation_output)
        return output

    def forward_from_phase(
        self,
        phi_lr: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the Stage 3 optical chain from phase input.

        Args:
            phi_lr: Real-valued phase tensor shaped `[B, 1, H, W]` with
                `H, W == grid_config.input_pattern_hw`. Batch dimension is
                allowed. The tensor device determines the optics execution
                device for this forward call.
            amplitude: Optional real-valued amplitude tensor with the same
                shape as `phi_lr`. Stage 3 defaults to `None`, which means the
                phase-only mainline uses amplitude 1.
            return_intermediates: Whether to include propagation-stage tensors
                for debugging and validation.

        Returns:
            A mapping containing at least:
            - `U0`
            - `U_out_full`
            - `I_out_full`
            - `I_out_roi`

        Contract notes:
            - This is the stable Stage 3 entrypoint for any phase-domain input,
              including future encoder output.
            - The caller/provider is responsible for supplying a real phase
              tensor with correct batch/channel/spatial layout.
            - The decoder is responsible for converting that phase tensor into
              a coherent field before propagation.
        """
        U0 = self.build_input_field(phi_lr, amplitude=amplitude)
        output = self.forward_from_field(U0, return_intermediates=return_intermediates)
        output["U0"] = U0
        return output

    def forward_from_phase_provider(
        self,
        phase_provider: PhaseProvider,
        upstream_input: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Resolve phase through a provider, then delegate to `forward_from_phase`.

        This is a Stage 3/Stage 4 boundary hook, not an encoder implementation.
        It exists so future learned encoders only need to satisfy the
        `PhaseProvider` contract and can stay outside the optical core.

        Args:
            phase_provider: Callable that receives `upstream_input` and returns
                a real phase tensor shaped `[B, 1, H, W]`.
            upstream_input: Arbitrary upstream tensor owned by the caller. In
                Stage 3 this may simply already be `phi_lr`; in future Stage 4
                it may be an encoder input tensor.
            amplitude: Optional amplitude tensor forwarded to
                `forward_from_phase(...)`.
            return_intermediates: Forwarded to `forward_from_phase(...)`.
        """
        if not callable(phase_provider):
            raise TypeError("phase_provider must be callable.")

        phi_lr = phase_provider(upstream_input)
        if not isinstance(phi_lr, torch.Tensor):
            raise TypeError(
                "phase_provider must return a torch.Tensor phase representation."
            )

        return self.forward_from_phase(
            phi_lr,
            amplitude=amplitude,
            return_intermediates=return_intermediates,
        )

    def forward(
        self,
        phi_lr: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self.forward_from_phase(
            phi_lr,
            amplitude=amplitude,
            return_intermediates=return_intermediates,
        )


__all__ = [
    "DiffractiveDecoder",
    "DistanceSchedule",
    "OpticalGridConfig",
    "PhaseMaskConfig",
]
