# TopK Triton Package Usage

## Package Contents

```text
python/topk-triton.py                  # current standalone Triton TopK implementation
custom-topk/*.bc                       # bitcode files used by registered custom ops
custom-topk/*.cpp                      # custom op source files matching those bitcodes
topk-triton.md                         # implementation/path/performance report snapshot
```

This package intentionally includes only custom ops referenced by the current `topk-triton.py` registration table. Removed experiment files such as the old large-K `stream16_merge_unpack_kernel` path are not required by this package.

## Runtime Setup

On the Ascend container, copy `custom-topk/` to the bitcode directory used by the Python script, for example:

```bash
mkdir -p /home/topk/topk_import_check/custom-topk
cp custom-topk/*.bc /home/topk/topk_import_check/custom-topk/
chmod 755 /home/topk/topk_import_check/custom-topk/*.bc
```

Then run with:

```bash
export TOPK_BC_DIR=/home/topk/topk_import_check/custom-topk
export TOPK_SMALLK_BC_DIR=/home/topk/topk_import_check/custom-topk
export TOPK_LAYERED_BC_DIR=/home/topk/topk_import_check/custom-topk
python python/topk-triton.py --device 7 --m 1 --n 16384 --k 8193 --seg-len 4096 --bench
```

If `TOPK_SMALLK_BC_DIR` or `TOPK_LAYERED_BC_DIR` are not set, `topk-triton.py` first tries `/home/topk/topk/custom-topk` and then falls back to `TOPK_BC_DIR`.

## Python API

The main callable is:

```python
import torch
from topk_triton import topk

x = torch.randn((batch, n), device="npu", dtype=torch.float32)
values, indices = topk(x, k, seg_len=4096)
```

Current assumptions:

- Input is a 2D tensor shaped `(batch, n)`.
- Output values are sorted descending.
- Output indices are column indices in the input row.
- Main tested dtype is `float32`.
- The implementation targets Ascend NPU through Triton Ascend custom ops.

Because the filename contains a hyphen, direct Python import requires either renaming it to `topk_triton.py` or using `importlib.util.spec_from_file_location`.

## Custom Op Mapping

| Python custom op | Symbol | Bitcode | Source | Role |
| --- | --- | --- | --- | --- |
| `sort_1d_topk_proposals` | `custom_sort_1d_topk_proposals_float` | `sort_topk.bc` | `custom-sort.cpp` | Base 1D segment topK proposal sort. Reads one raw value segment and writes sorted proposal pairs `[value, index_as_f32]`. |
| `sort_1d_topk_proposals_4x1024` | `custom_sort_1d_topk_proposals_4x1024_float` | `sort_topk_smallk.bc` | `custom-sort-smallk.cpp` | Small-K 4x1024 sort specialization used by `sort_kernel` for selected small-K cases. |
| `sort_1d_topk_proposals_layered_4096` | `custom_sort_1d_topk_proposals_layered_4096_float` | `sort_topk_layered.bc` | `custom-sort-layered.cpp` | Layered 4096 segment sort specialization for mid-K cases. |
| `vmrgsort4_exhaust_step` | `custom_vmrgsort4_exhaust_step_float` | `merge-sort-exhaust-1.bc` | `custom-merge-sort-exhaust.cpp` | 4-way proposal merge step. Consumes up to four sorted proposal windows and returns merged proposals plus per-way consumed counts. |
| `unpack_topk_float` | `custom_unpack_topk_float` | `merge-sort-exhaust-1.bc` | `custom-merge-sort-exhaust.cpp` | Converts proposal pairs `[value, index_as_f32]` into final `values` and `indices`. |
| `gm_to_ub_copy_float` | `custom_gm_to_ub_copy_float` | `gm-to-ub.bc` | `custom-gm-to-ub.cpp` | Fixed-size GM to UB copy. Used for full segment/full chunk reads. |
| `gm_to_ub_copy_n_float` | `custom_gm_to_ub_copy_n_float` | `gm-to-ub-n.bc` | `custom-gm-to-ub-n.cpp` | Runtime-length GM to UB copy. Used for tail segments and partial chunks to avoid over-read. |
| `ub_to_gm_copy_float` | `custom_ub_to_gm_copy_float` | `ub-to-gm.bc` | `custom-ub-to-gm.cpp` | UB to GM copy. Used for sorted proposal writes and intermediate merge writes. |
| `merge4_static_gm_to_output_float` | `custom_merge4_static_gm_to_output_float` | `merge4-static-gm-to-output.bc` | `custom-merge4-static-gm-to-output.cpp` | Static small-run final path. Reads up to four proposal runs from GM, merges, unpacks, and writes final output. |

## Kernel Path Summary

The Python implementation dispatches among these main paths:

| Path | Main condition | Kernel sequence |
| --- | --- | --- |
| Tiny-S static final | Small batch, very small `S=ceil(n/seg_len)`, small K | `sort_kernel -> static_merge4_unpack_kernel` |
| Small-K direct stream final | `K <= 2048` and no core-local reduction needed | `sort_kernel -> smallk_stream_merge*_unpack_kernel` |
| Small-K batch-local stream + final merge | Large S with `K <= 2048` | `sort_kernel -> smallk_core_local_stream_merge_kernel -> final/group merge + unpack` |
| Large-K batch-local + final merge | Large S with `K > 2048` or large-K fallback | `sort_kernel -> core_local_nocopy_merge_kernel -> final_merge_unpack_kernel` or `unpack_kernel` |
| Base fallback | Other verified cases | `sort_kernel -> generic_merge_unpack_kernel` or `final_merge_unpack_kernel` |

## Important Buffer Notes

Proposal format is always two f32 slots:

```text
proposal = [value, index_as_f32]
```

For the chunked 4-way merge kernels, a merge chunk of `C` proposal per way requires roughly:

```text
in_buf  = 4 * C * 2 f32
out_buf = 4 * C * 2 f32 + 8 f32
```

Current defaults in `topk-triton.py`:

```text
CHUNK = 2048              # final/direct merge chunk
CORE_LOCAL_CHUNK = 1152   # core-local/group merge chunk
```

`CORE_LOCAL_CHUNK` is smaller because core-local kernels carry more live UB state and intermediate merge/copy bookkeeping than the final merge path.
