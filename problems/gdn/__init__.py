"""BF16 Gated Delta Net paper workload."""

from .reference import (
    BATCH,
    CHUNK_SIZE,
    HEAD_DIM,
    QK_HEADS,
    SEQUENCE_LENGTH,
    VALUE_HEADS,
    make_inputs,
    naive_chunk_gated_delta_rule,
    naive_recurrent_gated_delta_rule,
    torch_ref,
)

__all__ = [
    "BATCH",
    "CHUNK_SIZE",
    "HEAD_DIM",
    "QK_HEADS",
    "SEQUENCE_LENGTH",
    "VALUE_HEADS",
    "make_inputs",
    "naive_chunk_gated_delta_rule",
    "naive_recurrent_gated_delta_rule",
    "torch_ref",
]
