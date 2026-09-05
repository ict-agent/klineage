"""FP32 Top-K paper workload."""

from .reference import BATCH, SEQUENCE_LENGTH, K, make_inputs, torch_ref

__all__ = ["BATCH", "SEQUENCE_LENGTH", "K", "make_inputs", "torch_ref"]
