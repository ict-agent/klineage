# Optimization technique matrices

The two figures compare the selected final AscendC submissions for all five operators. Every A/B submission config in this directory pins the language to AscendC.

- [Setting A: without expert knowledge](without-expert.svg)
- [Setting A PNG](without-expert.png)
- [Setting B: with expert knowledge](with-expert.svg)
- [Setting B PNG](with-expert.png)
- [Shared matrix data](matrix.csv)

The row labels and order come from one shared matrix, so the two figures are directly comparable. A check means the implementation mechanism is visible in the selected source. A dash means it was not observed in that source or is not applicable. These marks describe code, not an isolated performance effect or proof that a technique caused a speedup. The audit uses each setting's final submission solution; rejected versions are not treated as the final implementation.

Setting A is without_memory; Setting B is with_memory. These names describe the initial expert-knowledge condition, not a language difference. Any later environment reads or run deviations are documented in the corresponding operator audit and are not encoded by this source-technique matrix.

## Source checks

The selected code bundles inspected are:

- KDA: [A kernel](../kda/without_memory/submission/solution/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp#L87), [B kernel](../kda/with_memory/submission/solution/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp#L48), and [B state-layout wrapper](../kda/with_memory/submission/solution/op_host/launch.cpp#L110).
- Top-P: [A kernel](../top_p/without_memory/submission/solution/kernel.asc#L41) and [B kernel](../top_p/with_memory/submission/solution/kernel.asc#L18).
- GQA: [A kernel](../gqa/without_memory/submission/solution/kernel.asc#L18) and [B kernel](../gqa/with_memory/submission/solution/kernel.asc#L18).
- Sparse Attention: [A kernel](../sparse_attention/without_memory/submission/solution/kernel.asc#L120) and [B staged kernel](../sparse_attention/with_memory/submission/solution/kernel.asc#L110).
- Fused Add + RMSNorm: [A kernel](../fused_add_rmsnorm/without_memory/submission/solution/kernel.asc#L64) and [B kernel](../fused_add_rmsnorm/with_memory/submission/solution/kernel.asc#L34).

Notable source-level differences include Cube QK/PV and event-ordered cross-stream phases in GQA B and Sparse Attention B; a nonuniform stream split in GQA B; a two-slot softmax queue and c+2 gather scheduling in Sparse Attention B; Top-P B's threshold-band GatherMask; and Fused Add + RMSNorm B's vector-core-aware row split and output L2 write hint. Both Sparse Attention submissions use max-subtracted softmax; GQA B does as well, while GQA A exponentiates before a max reduction.

## Rebuild

From this directory, run python3 build.py. The script uses the Python standard library. It verifies that all ten selected configs are AscendC, then rebuilds both SVG figures and the CSV from matrix.json. On macOS, it also uses Quick Look to write the two PNG files.
