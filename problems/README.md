# Paper workload Torch references

This directory contains dependency-light PyTorch correctness references for
the five main-tier workloads in the paper.  Each problem exposes the same two
entry points:

- `make_inputs(...)` creates deterministic inputs in the kernel's actual ABI.
- `torch_ref(...)` computes the reference result from those inputs.

Example:

```python
from problems.gemm import make_inputs, torch_ref

inputs = make_inputs(device="cuda", seed=17)
expected = torch_ref(**inputs)
```

The default shapes are the paper shapes:

| Problem | Input contract | Default shape and dtype |
| --- | --- | --- |
| GEMM | `x[M,K]`, `weight[N,K]`; computes `x @ weight.T` | `4096^3`, BF16 |
| Conv2d | NHWC input, flattened HWCF filter | N8 C64 H56 W56 F128 K3, FP16 |
| FMHA | packed NHD Q/K/V plus cumulative sequence lengths | B8 H64 S2048 D128, FP16 |
| GDN | BTHD Q/K/V, log-space gate, beta, initial state | Hq16 Hv48 S4096 D128 C64, BF16 |
| Top-K | row-major `[batch, sequence]` values | B64 S4096 K512, FP32 |

GEMM, Conv2d, FMHA, and Top-K follow the `torch_ref` and correctness paths in
`ict-agent/fgw_workspace/experiments/_archive/trajectory_transfer`. GDN's
recurrent and chunk-naive definitions are adapted directly from FLA's official
[`naive.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py),
with a thin GVA adapter for the paper's 16 Q/K heads and 48 value heads. The
code keeps the submitted paper's Top-K sequence length of 4096 (the archived
benchmark has since changed that workload to 8192).

The references allocate outputs and are intended as correctness oracles.  A
performance evaluator should benchmark the candidate kernel separately and
may use a preallocated `out` tensor where a reference explicitly supports it.
