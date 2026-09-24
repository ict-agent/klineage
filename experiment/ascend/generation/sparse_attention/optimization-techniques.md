# sparse_attention：A/B优化技巧

按最终选中源码判断；✓为存在，✓*为给定初始实现已有，—为未观察到或不适用。未单独消融，不将代码机制直接视为独立性能收益。

| 优化技巧 | A | B |
|---|:---:|:---:|
| Cube Matmul for both QK and PV | — | ✓ |
| Cube / Vector stages on separate streams | — | ✓ |
| Event-driven cross-chunk pipeline | — | ✓ |
| Gather KV once; reuse across QK and PV | — | ✓ |
| Explicit L1 depth / L0 double-buffer tiling | — | ✓ |
| TQue-managed producer / consumer buffers | — | ✓ |
| ReduceSum API instead of a manual sum tree | ✓* | ✓ |
| Double buffering and input prefetch | — | ✓ |
| Vector arithmetic / UB reuse / mixed precision | ✓* | ✓ |

[PNG](plots/optimization-techniques.png) · [PDF](plots/optimization-techniques.pdf) · [A源码](without_memory/submission/solution/kernel.asc) · [B源码](with_memory/submission/solution/kernel.asc)

A保留初始实现，已有技巧不算本轮新增优化。B主要区别是Cube QK/PV、跨流事件流水、KV gather复用及L1/L0缓冲配置。
