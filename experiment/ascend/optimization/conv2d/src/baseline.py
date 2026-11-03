"""The klineage reference ABI backed by torch-npu's native convolution."""

import torch.nn.functional as functional

from .conv2d import KERNEL_SIZE, PADDING, STRIDE


def prepare(data_nhwc, weight_hwcf):
    c = data_nhwc.shape[-1]
    f = weight_hwcf.shape[1]
    data_nchw = data_nhwc.permute(0, 3, 1, 2)
    weight_oihw = (weight_hwcf.transpose(0, 1)
                   .reshape(f, KERNEL_SIZE, KERNEL_SIZE, c)
                   .permute(0, 3, 1, 2).contiguous())
    return data_nchw, weight_oihw


def native(data_nchw, weight_oihw):
    return functional.conv2d(data_nchw, weight_oihw, bias=None,
                             stride=STRIDE, padding=PADDING)


def run(data_nhwc, weight_hwcf):
    x, weight = prepare(data_nhwc, weight_hwcf)
    return native(x, weight).permute(0, 2, 3, 1).contiguous()
