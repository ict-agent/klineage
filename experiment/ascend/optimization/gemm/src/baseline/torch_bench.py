"""torch-npu baseline runner for GEMM 4096x4096x4096 BF16 on Ascend 910B1.

Launches `launches` matmuls back to back and nothing else. run_benchmark.sh
wraps it in `msprof op --warm-up=3 --launch-count=10`, which profiles the last
10 launches; the reported kernel time comes from that profile, not from here.

Usage: python3 torch_bench.py [device_id] [launches]
"""
import sys

import torch
import torch_npu

M = N = K = 4096
WARMUP = 3
LAUNCHES = 10


def main() -> int:
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    launches = int(sys.argv[2]) if len(sys.argv) > 2 else WARMUP + LAUNCHES

    torch.npu.set_device(device_id)
    torch.manual_seed(1234)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=f"npu:{device_id}")
    b = torch.randn(K, N, dtype=torch.bfloat16, device=f"npu:{device_id}")

    for _ in range(launches):
        c = torch.matmul(a, b)
    torch.npu.synchronize()

    print(f"shape: {M}x{N}x{K} bf16, launches: {launches}, out: {c.dtype} {tuple(c.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
