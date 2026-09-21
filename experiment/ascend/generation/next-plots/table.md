| Kernel | Setting | Correct | Latency (us) | vs. Baseline |
| --- | --- | --- | --- | --- |
| FUSED_ADD_RMSNORM | With Expert Knowledge | ✓ | 457.0 us | 13.17× |
| FUSED_ADD_RMSNORM | Without Expert Knowledge | ✓ | 565.9 us | 10.62× |
| SPARSE_ATTENTION | With Expert Knowledge | ✓ | 117719.3 us | 3.88× |
| SPARSE_ATTENTION | Without Expert Knowledge | ✗ | - | - |
