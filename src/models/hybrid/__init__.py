"""Hybrid model wrappers used for end-to-end model composition."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "MinimalHybridWrapper":
        from src.models.hybrid.minimal_hybrid_wrapper import MinimalHybridWrapper

        return MinimalHybridWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MinimalHybridWrapper"]
