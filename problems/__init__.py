"""PyTorch correctness references for the five paper workloads.

The subpackages deliberately are not imported here.  This keeps ``problems``
importable in environments where PyTorch is supplied only by the CUDA image.
"""

__all__ = ["conv2d", "fmha", "gdn", "gemm", "topk"]
