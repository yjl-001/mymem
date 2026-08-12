"""MemGen package with lazy top-level exports.

Keeping model imports lazy lets lightweight offline data utilities run without
initializing the full Torch/PEFT training stack.
"""

from typing import Any

__all__ = [
    "MemGenModel",
    "MemGenRunner",
]


def __getattr__(name: str) -> Any:
    if name == "MemGenModel":
        from .model.modeling_memgen import MemGenModel

        return MemGenModel
    if name == "MemGenRunner":
        from .runner import MemGenRunner

        return MemGenRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
