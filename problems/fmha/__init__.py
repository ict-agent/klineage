"""FP16 non-causal FMHA paper workload."""

from .reference import (
    BATCH,
    HEAD_DIM,
    HEADS,
    SEQUENCE_LENGTH,
    make_inputs,
    torch_ref,
)

__all__ = [
    "BATCH",
    "HEADS",
    "HEAD_DIM",
    "SEQUENCE_LENGTH",
    "make_inputs",
    "torch_ref",
]
