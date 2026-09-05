"""FP16 channels-last Conv2d paper workload."""

from .reference import (
    BATCH,
    HEIGHT,
    IN_CHANNELS,
    KERNEL_SIZE,
    OUT_CHANNELS,
    PADDING,
    STRIDE,
    WIDTH,
    make_inputs,
    torch_ref,
)

__all__ = [
    "BATCH",
    "HEIGHT",
    "IN_CHANNELS",
    "KERNEL_SIZE",
    "OUT_CHANNELS",
    "PADDING",
    "STRIDE",
    "WIDTH",
    "make_inputs",
    "torch_ref",
]
