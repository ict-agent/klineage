# Expert knowledge (Setting B)

One directory per kernel: `expert/<kernel>/`. Files (Markdown or source excerpts)
are copied into the run workspace as `expert/` and referenced from `AGENTS.md`.

Guidelines:
- Prefer implementation strategy over prose: data movement (GM/UB/L1), pipeline
  structure, Vector/Cube split, tiling, alignment, stream/event ordering.
- Keep the evaluation interface out of these files; Setting A and B must share
  the same gate, or the comparison is invalid.
- Sources: CANN operator samples, vllm-ascend / cann-ops kernels, profiling notes.

`with_memory` units refuse to start while the kernel directory is empty.
