"""Minimal Stage 4 hybrid wrapper for encoder-to-optics closed-loop forward.

This module implements the Stage 4 model-side shell:

`HR input -> minimal encoder -> phi_lr -> diffractive decoder`

It is intentionally not a trainer, dataset adapter, ROI target adapter, or
paper-final system. The Stage 3 optical contract remains inherited and
unchanged:

`phi_lr -> U0 -> U_out_full -> I_out_full -> I_out_roi`
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.encoders import MinimalPhaseEncoder
from src.models.optics.diffractive_decoder import DiffractiveDecoder


class MinimalHybridWrapper(nn.Module):
    """Compose a minimal phase encoder with the frozen optical decoder.

    This wrapper exists for Stage 4 closed-loop integration and debugging. It
    only wires the current minimal encoder into
    `DiffractiveDecoder.forward_from_phase(...)` and returns key intermediate
    tensors needed by later dataset/target-adapter and training issues.
    """

    REQUIRED_OUTPUT_KEYS = ("U0", "U_out_full", "I_out_full", "I_out_roi")

    def __init__(
        self,
        *,
        encoder: MinimalPhaseEncoder,
        decoder: DiffractiveDecoder,
    ) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoder must be an nn.Module, got {type(encoder)!r}.")
        if not isinstance(decoder, DiffractiveDecoder):
            raise TypeError(
                "decoder must be a DiffractiveDecoder so the Stage 3 optical "
                f"contract stays explicit, got {type(decoder)!r}."
            )

        self.encoder = encoder
        self.decoder = decoder

    @property
    def expected_phi_hw(self) -> tuple[int, int]:
        """Expose the decoder input phase shape expected by the optical core."""
        return self.decoder.grid_config.input_pattern_hw

    def _validate_phi_lr(self, phi_lr: torch.Tensor) -> None:
        if not isinstance(phi_lr, torch.Tensor):
            raise TypeError(f"phi_lr must be a torch.Tensor, got {type(phi_lr)!r}.")
        if phi_lr.ndim != 4:
            raise ValueError(
                "phi_lr must have shape [B, 1, H, W], "
                f"got {tuple(phi_lr.shape)}."
            )
        if phi_lr.shape[1] != 1:
            raise ValueError(
                "Stage 4 minimal wrapper expects single-channel phi_lr, "
                f"got {tuple(phi_lr.shape)}."
            )
        if tuple(phi_lr.shape[-2:]) != self.expected_phi_hw:
            raise ValueError(
                "Encoder output shape must match decoder input_pattern_hw, "
                f"got {tuple(phi_lr.shape[-2:])} vs {self.expected_phi_hw}."
            )

    def encode_phase(
        self,
        x_hr: torch.Tensor,
        *,
        return_raw_phase: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Encode HR input into phase-domain `phi_lr`.

        The raw-to-phase mapping remains owned by the encoder side. This
        wrapper does not duplicate phase mapping logic.
        """
        if return_raw_phase:
            raw_phase = self.encoder.encode_raw_phase(x_hr)
            phi_lr = self.encoder.map_raw_phase(raw_phase)
            self._validate_phi_lr(phi_lr)
            return {
                "raw_phase": raw_phase,
                "phi_lr": phi_lr,
            }

        phi_lr = self.encoder(x_hr)
        self._validate_phi_lr(phi_lr)
        return {"phi_lr": phi_lr}

    def forward(
        self,
        x_hr: torch.Tensor,
        *,
        amplitude: torch.Tensor | None = None,
        return_intermediates: bool = False,
        return_encoder_debug: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the minimal Stage 4 end-to-end forward on HR-domain input."""
        encoded = self.encode_phase(
            x_hr,
            return_raw_phase=return_encoder_debug,
        )
        phi_lr = encoded["phi_lr"]
        decoder_output = self.decoder.forward_from_phase(
            phi_lr,
            amplitude=amplitude,
            return_intermediates=return_intermediates,
        )

        for key in self.REQUIRED_OUTPUT_KEYS:
            if key not in decoder_output:
                raise RuntimeError(f"Decoder output is missing required key: {key}.")

        output: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "phi_lr": phi_lr,
            **decoder_output,
        }
        if "raw_phase" in encoded:
            output["raw_phase"] = encoded["raw_phase"]
        return output


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains NaN or Inf values.")


@torch.no_grad()
def _run_self_check() -> None:
    """Run a fake-batch smoke check for the Stage 4 hybrid forward."""
    encoder = MinimalPhaseEncoder(
        in_channels=1,
        target_hw=(24, 24),
        base_channels=8,
        hidden_channels=16,
        phase_range=(-math.pi, math.pi),
    )
    decoder = DiffractiveDecoder(
        num_diffractive_layers=1,
        wavelength=532e-9,
        pixel_pitch=8e-6,
        grid_config={
            "input_pattern_hw": (24, 24),
            "layer_hw": (32, 32),
            "propagation_hw": (48, 48),
        },
        distance_schedule={
            "input_to_first": 0.01,
            "inter_layer": [],
            "last_to_sensor": 0.03,
        },
        readout_config={
            "output_crop_hw": (20, 20),
        },
        phase_mask_config={
            "init_mode": "zeros",
            "phase_mapping": "tanh",
            "phase_range": (-math.pi, math.pi),
        },
    )
    wrapper = MinimalHybridWrapper(encoder=encoder, decoder=decoder)

    x_hr = torch.rand(2, 1, 96, 96, dtype=torch.float32)
    output = wrapper(
        x_hr,
        return_intermediates=False,
        return_encoder_debug=True,
    )

    expected_shapes = {
        "raw_phase": (2, 1, 24, 24),
        "phi_lr": (2, 1, 24, 24),
        "U0": (2, 1, 48, 48),
        "U_out_full": (2, 1, 48, 48),
        "I_out_full": (2, 1, 48, 48),
        "I_out_roi": (2, 1, 20, 20),
    }
    for key, expected_shape in expected_shapes.items():
        value = output[key]
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"{key} must be a tensor, got {type(value)!r}.")
        if tuple(value.shape) != expected_shape:
            raise AssertionError(
                f"{key} shape mismatch: got {tuple(value.shape)}, expected {expected_shape}."
            )
        _assert_finite(key, value)

    print("MinimalHybridWrapper self-check passed.")
    for key in ("phi_lr", "U0", "U_out_full", "I_out_full", "I_out_roi"):
        tensor = output[key]
        assert isinstance(tensor, torch.Tensor)
        print(f"{key}.shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    _run_self_check()


__all__ = ["MinimalHybridWrapper"]
