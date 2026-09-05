"""PyTorch reference for the paper's channels-last Conv2d ABI.

The kernel-facing input is NHWC.  The filter is a flattened HWCF matrix with
shape ``[kernel_h * kernel_w * in_channels, out_channels]``.  ``torch_ref``
converts those views to NCHW/OIHW for ``torch.nn.functional.conv2d`` and
returns an NHWC tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BATCH = 8
IN_CHANNELS = 64
HEIGHT = 56
WIDTH = 56
OUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DTYPE = torch.float16


def make_inputs(
    *,
    batch: int = BATCH,
    in_channels: int = IN_CHANNELS,
    height: int = HEIGHT,
    width: int = WIDTH,
    out_channels: int = OUT_CHANNELS,
    kernel_size: int = KERNEL_SIZE,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create contiguous NHWC activation and flattened HWCF filter tensors."""

    dimensions = (
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_size,
    )
    if min(dimensions) <= 0:
        raise ValueError("all Conv2d dimensions must be positive")
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    return {
        "data_nhwc": torch.randn(
            (batch, height, width, in_channels),
            dtype=dtype,
            device=device,
            generator=generator,
        ).contiguous(),
        "weight_hwcf": torch.randn(
            (kernel_size * kernel_size * in_channels, out_channels),
            dtype=dtype,
            device=device,
            generator=generator,
        ).contiguous(),
    }


def torch_ref(
    data_nhwc: torch.Tensor,
    weight_hwcf: torch.Tensor,
    *,
    kernel_size: int = KERNEL_SIZE,
    stride: int = STRIDE,
    padding: int = PADDING,
) -> torch.Tensor:
    """Compute Conv2d and return ``[N, output_h, output_w, F]``."""

    if data_nhwc.ndim != 4:
        raise ValueError("data_nhwc must have shape [N, H, W, C]")
    if weight_hwcf.ndim != 2:
        raise ValueError("weight_hwcf must have shape [kernel_h * kernel_w * C, F]")
    if kernel_size <= 0 or stride <= 0 or padding < 0:
        raise ValueError("kernel_size/stride must be positive and padding non-negative")
    if data_nhwc.device != weight_hwcf.device:
        raise ValueError("activation and filter must be on the same device")
    if data_nhwc.dtype != weight_hwcf.dtype:
        raise ValueError("activation and filter must have the same dtype")

    in_channels = data_nhwc.shape[-1]
    expected_rows = kernel_size * kernel_size * in_channels
    if weight_hwcf.shape[0] != expected_rows:
        raise ValueError(
            f"filter has {weight_hwcf.shape[0]} rows; expected {expected_rows}"
        )
    out_channels = weight_hwcf.shape[1]

    data_nchw = data_nhwc.permute(0, 3, 1, 2)
    weight_oihw = (
        weight_hwcf.transpose(0, 1)
        .reshape(out_channels, kernel_size, kernel_size, in_channels)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    output_nchw = F.conv2d(
        data_nchw,
        weight_oihw,
        bias=None,
        stride=stride,
        padding=padding,
    )
    return output_nchw.permute(0, 2, 3, 1).contiguous()


__all__ = [
    "BATCH",
    "DTYPE",
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
