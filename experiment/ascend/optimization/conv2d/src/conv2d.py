"""Fixed-shape NHWC Conv2d on Ascend with FlagTree TLE."""

import torch
import torch_npu

from . import tle_kernels

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH = 8
INPUT_SHAPE = (BATCH, tle_kernels.HEIGHT, tle_kernels.WIDTH, tle_kernels.CHANNELS)
WEIGHT_SHAPE = (KERNEL_SIZE * KERNEL_SIZE * tle_kernels.CHANNELS, tle_kernels.FILTERS)
OUTPUT_SHAPE = (*INPUT_SHAPE[:-1], tle_kernels.FILTERS)


def check_inputs(x, weight):
    if tuple(x.shape) != INPUT_SHAPE or tuple(weight.shape) != WEIGHT_SHAPE:
        raise ValueError(f"Expected input {INPUT_SHAPE} and weight {WEIGHT_SHAPE}")
    if x.device.type != "npu" or x.device != weight.device:
        raise ValueError("Inputs must be on the same NPU")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("Expected FP16 inputs")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Inputs must be contiguous")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("Expected inference tensors")


def _launch(data, weight):
    output = torch.empty(OUTPUT_SHAPE, device=data.device, dtype=data.dtype)
    return tle_kernels.launch(data, weight, output)


def run(data_nhwc, weight_hwcf, *, kernel_size=KERNEL_SIZE,
        stride=STRIDE, padding=PADDING):
    """Return contiguous [8,56,56,128] FP16 output."""
    if (kernel_size, stride, padding) != (KERNEL_SIZE, STRIDE, PADDING):
        raise ValueError("Expected kernel=3, stride=1, padding=1")
    check_inputs(data_nhwc, weight_hwcf)
    if torch.npu.current_device() == data_nhwc.device.index:
        return _launch(data_nhwc, weight_hwcf)
    with torch.npu.device(data_nhwc.device):
        return _launch(data_nhwc, weight_hwcf)
