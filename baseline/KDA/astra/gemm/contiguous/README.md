# Contiguous output comparison

The saved GEMM writes contiguous output in **255.325 us**, versus
**284.235 us** for `torch.matmul`: **1.113x**, or **10.17% lower latency**.
Adding `.contiguous()` after the original padded output instead costs
**302.264 us**, or **6.34% higher latency** than the reference.

| Case | Output stride | Median us | Round median range, us | Reference / case |
| --- | --- | ---: | ---: | ---: |
| Reference | `[4096, 1]` | 284.235 | 279.812–291.125 | 1.000x |
| Original padded kernel | `[4288, 1]` | 253.620 | 249.756–258.749 | 1.121x |
| Direct contiguous output | `[4096, 1]` | 255.325 | 251.837–260.942 | 1.113x |
| Padded kernel + `.contiguous()` | `[4096, 1]` | 302.264 | 301.975–309.417 | 0.940x |

Direct output passes a newly allocated contiguous tensor through the saved
kernel's existing `out=` argument. Its CUDA, Python and configuration files are
unchanged. The copy case calls `.contiguous()` inside the measured callable.
Both cases allocate on each call, as does the reference.

All four cases pass seeds 17, 43, 101 and 211, plus seed 307 written into the
same timed input addresses. Inputs remain unchanged. Both contiguous candidates
pass `assert_close(check_stride=True)` and `is_contiguous()`; the padded case
retains the original numerical contract as a control. GEMM tolerances remain
`rtol=0.02, atol=0.02`; observed maximum absolute error is zero in these checks.
A CPU negative control confirms padded outputs fail the strict stride check.

Eight rounds rotate and reverse case order, placing each case twice in each
position. Every case uses 3 trials of 50 measurements with 10 warmups and cold
L2: 1,200 samples per case, 4,800 total. Each round reduces by median of trial
medians; the table takes the median across rounds. Ranges describe round
medians, not confidence intervals. All cases use the same timed inputs.

Timing uses FlashInfer CUPTI without CUDA graphs. Each sample spans the first
GPU activity's start through the last activity's end. The copy case includes
both GEMM and the copy, including any gap between them. Host setup before the
first GPU activity and allocation are excluded. These are not application
end-to-end timings. TF32 and reduced-precision BF16/FP16 reductions are disabled.

H100 PCIe, driver 590.48.01, PyTorch 2.10.0a0+b4e4ee81d3.nv25.12,
CUDA 13.1, FlashInfer 0.6.18.post1. GPU clocks were not locked. These results
compare with the selected PyTorch reference path; cuBLASLt algorithm tuning
was not performed. Padding does not explain the remaining contiguous-output
advantage; exact bottleneck attribution remains unmeasured.

Measured 2026-09-06 06:55:52–06:56:01 UTC. Source hashes match the original
final verdict. [results.json](results.json) contains raw samples, correctness
records, environment and hashes. [solution/](solution/) preserves the sources.
This supplementary comparison does not replace the original experiment.

From the repository root, use a new output directory:

```bash
PYTHONPATH=src python baseline/compare_gemm_layout.py \
  --output agent-workspace/audits/gemm-contiguous-repeat
```
