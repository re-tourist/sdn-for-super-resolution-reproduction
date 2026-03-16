"""Minimal phase-provider hook for Stage 3 / future Stage 4 integration.

This module intentionally does not implement an encoder. It only defines the
contract for "who provides phase" so the optical decoder can remain agnostic to
whether phase comes from:

- the current Stage 3 direct phase tensor workflow, or
- a future Stage 4 learned encoder.

The provider contract is deliberately small:

- input: an upstream tensor owned by the caller
- output: a real-valued phase tensor in `[B, 1, H, W]`
- device / dtype / batch alignment remain the caller/provider responsibility
- the optical decoder still owns phase-to-field mapping inside
  `forward_from_phase(...)`
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PhaseProvider(Protocol):
    """Callable contract for producing a phase-domain tensor.

    A conforming provider accepts some upstream tensor and returns a phase
    tensor shaped like `[B, 1, H, W]`. The returned tensor must be real-valued;
    the optical decoder will validate spatial size and then apply its own
    phase-to-field mapping.
    """

    def __call__(self, upstream_input: torch.Tensor, /) -> torch.Tensor:
        """Produce a phase tensor for the optical decoder."""


class DirectPhaseProvider:
    """Stage 3 default adapter: treat caller-provided phase as the provider output.

    This is a minimal stub for code paths that want to depend on the provider
    contract without introducing a learned encoder yet.
    """

    def __call__(self, upstream_input: torch.Tensor, /) -> torch.Tensor:
        if not isinstance(upstream_input, torch.Tensor):
            raise TypeError(
                "DirectPhaseProvider expects a torch.Tensor and returns it unchanged."
            )
        return upstream_input


__all__ = [
    "DirectPhaseProvider",
    "PhaseProvider",
]
