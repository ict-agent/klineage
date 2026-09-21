# Expert knowledge: AddRmsNorm

Pinned CANN sources are listed with hashes in source.json; original licenses are retained.
Read ops_nn/norm/add_rms_norm/README.md, op_host/add_rms_norm_tiling.cpp and the
op_kernel headers for fusion, row grouping, reduction and buffer management strategies.
These are reference materials, not a prebuilt candidate. Adapt the design to the problem
ABI and verify every output against the supplied torch reference. In particular, use the
unrounded FP32 sum for normalization; residual_out is the separately rounded BF16 sum.
The CANN operator also returns rstd, which this problem does not request.
