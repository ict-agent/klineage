# fused_add_rmsnorm：A/B优化技巧

按最终选中源码判断；✓为存在，✓*为给定初始实现已有，—为未观察到或不适用。未单独消融，不将代码机制直接视为独立性能收益。

| 优化技巧 | A | B |
|---|:---:|:---:|
| Explicit output L2 write-cache hint | — | ✓ |
| Query hardware core count for launch grid | — | ✓ |
| Even row partition (core loads differ by <= 1) | — | ✓ |
| TQue-managed producer / consumer buffers | — | ✓ |
| ReduceSum API instead of a manual sum tree | — | ✓ |
| Double buffering and input prefetch | ✓ | ✓ |
| Vector arithmetic / UB reuse / mixed precision | ✓ | ✓ |

[PNG](plots/optimization-techniques.png) · [PDF](plots/optimization-techniques.pdf) · [A源码](without_memory/submission/solution/kernel.asc) · [B源码](with_memory/submission/solution/kernel.asc)

A/B都有双缓冲和预取；B改用TQue并不表示A没有流水。B的WRITE缓存提示仅作用于输出，不是输入读缓存。A采用向量树归约并保留32项标量尾部；B调用ReduceSum。
