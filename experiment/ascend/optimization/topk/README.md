# TopK on Ascend: B=64 S=4096 K=512, FP32

## 1. Operator

- Compute: row-wise largest-K selection, unsorted.
- Input: `values` FP32 `[B, S]`, contiguous row-major.
- Outputs: `top_values` FP32 `[B, K]`, `indices` int64 `[B, K]`.
- Test shape: `B=64`, `S=4096`, `K=512`; FP32, from
  `problems/definitions/topk.json` and `problems/workloads/topk.jsonl`.
- Correctness: the definition compares the *sorted* selected-value sets, so the
  descending order of this implementation is fine. `src/ours/topk-triton.py`
  checks that set plus `x.gather(1, indices) == values` at rtol/atol 1e-3.

## 2. Baseline

**torch-npu `torch.topk(values, 512, dim=1)`** (torch 2.9.0 + torch_npu
2.9.0.post2), `src/baseline/torch_bench.py`.

Two kernels per call: `aclnnTopk_TopkV2AiCore_TopKV2` (17.128 us) and
`aclnnTopk_CastAiCore_Cast` (2.824 us), sum 19.952 us.

## 3. Timing

One number per implementation and invocation: the sum of the device-side kernel
`Duration(us)` the torch_npu profiler reports for that invocation. Both sides
run 2 warm-up + 5 active invocations back to back and the mean of the 5 active
ones is reported, so the ratio is a like-for-like device-time speedup of the
whole operator. `src/ours/topk-triton.py --bench` measures the baseline and ours
in the same process with the same helper; the standalone baseline runner must
reproduce the same number.

## 4. Results

910B1 (`aicc-01`), NPU 6, `seg_len=2048`, us.

| Implementation | Kernels per call | Total (us) | Speedup |
|----------------|------------------|-----------:|---------|
| torch-npu baseline | `aclnnTopk_TopkV2AiCore_TopKV2` + `aclnnTopk_CastAiCore_Cast` | **19.832** | 1.000x |
| ours | `sort_kernel` 11.048 + `smallk_stream_merge4_unpack_kernel` 4.436 | **15.484** | **1.281x** |

- In-process cross-check (`topk-triton.py --bench`): torch.topk 19.952 us,
  triton 15.484 us, 1.288x, correctness PASSED.
- Repeat runs, both sides re-sampled: 20.561 / 15.449 (1.330x), 19.637 / 15.588
  (1.260x), 19.832 / 15.484 (1.281x).

## 5. Reproduce

Environment: container `topk-eval-o_zhangchenqing`, image
`harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base`,
CANN 9.0.0, triton-ascend 3.5.0 source tree, locally built
`bishengir-compile`. Python 3.11.15.

```sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PATH=/home/topk/topk_build/build/install/bin:$PATH
export PYTHONPATH=/home/topk/local_src/triton-ascend-src865/triton-ascend/python
bash run_benchmark.sh 0
```

`run_benchmark.sh` runs the baseline, then `src/ours/topk-triton.py --bench`,
then prints both numbers and the speedup.

## 6. Notes

- `seg_len=4096` (one segment per row, `S=1`) does not compile for this shape:
  `final_merge_unpack_kernel` fails in triton-to-linalg. `run_benchmark.sh`
  uses `seg_len=2048`, which dispatches to the small-K stream merge path.
- The custom ops need CANN 9.0.0 with the locally built bishengir. The CANN 9.1
  containers reject the same bitcode with `hivm.hir.store` / `hivm.hir.load`
  bishengir errors.
- `custom-topk/*.bc` are the shipped bitcode; `topk-triton.py`, its
  `CUSTOM_OP_USAGE.md` and `topk-triton.md` come unmodified from the TopK
  Triton package, and its default bitcode directory is overridden by
  `TOPK_BC_DIR` / `TOPK_SMALLK_BC_DIR` / `TOPK_LAYERED_BC_DIR`.
