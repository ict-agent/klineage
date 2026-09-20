# GEMM on Ascend: M=N=K=4096, BF16

## 1. Operator

- Compute: `C = A @ B`, both 2-D ND.
- Inputs: `A` BF16 `[4096, 4096]` (RowMajor), `B` BF16 `[4096, 4096]` (RowMajor).
- Output: `C` BF16 `[4096, 4096]` (RowMajor).
- Test shape: `M=N=K=4096`; dtype: BF16 in / BF16 out (FP32 accumulation in Cube).
- Correctness: element-wise against a torch-npu FP32 reference, `err_cnt: 0/16777216`,
  `max_rel_err ≈ 0.0039` (128³, 256x512x1024, 512x256x2048 and 1024³ are exact too).

## 2. Baseline

**torch-npu `torch.matmul`** (torch 2.10.0 + torch_npu 2.10.0.post4, see
`src/baseline/torch_bench.py`).

- Timing (identical for baseline and ours): `msprof op --kernel-name=<kernel>
  --warm-up=3 --launch-count=10`. The app launches 13 times back to back; we take
  the median of the last 10 device-side `Task Duration(us)` samples.
- Baseline kernel is `MatMulV3_ND_ND_ND_ND_BF16_BF16_BF16_BF16`;
  910B1 (CANN 9.1.0), device 3, samples
  444.95 / 446.69 / 444.23 / 447.99 / 443.99 / 445.53 / 446.47 / 443.89 / 441.77 / 437.23 us
  → **baseline_median_us = 444.59**.

## 3. Results

### 3.1 Main result (910B1, device 3, us)

| Implementation | 10 samples (us) | Median (us) |
|----------------|-----------------|-------------|
| torch-npu baseline | 444.95 / 446.69 / 444.23 / 447.99 / 443.99 / 445.53 / 446.47 / 443.89 / 441.77 / 437.23 | **444.59** |
| ours | 427.93 / 424.37 / 423.23 / 425.27 / 423.33 / 424.03 / 426.75 / 427.85 / 422.79 / 421.99 | **424.20** |

- **Speedup = 444.59 / 424.20 = 1.048x**.
- Whole script repeated 6 times, both sides re-sampled each round:

| Round | baseline median | ours median | Speedup |
|-------|-----------------|-------------|---------|
| 1 | 446.26 | 425.24 | 1.049x |
| 2 | 445.01 | 423.69 | 1.050x |
| 3 | 443.64 | 424.34 | 1.045x |
| 4 | 446.89 | 424.02 | 1.054x |
| 5 | 445.89 | 425.04 | 1.049x |
| 6 | 444.59 | 424.20 | 1.048x |

Median speedup **1.049x**, range 1.045x ~ 1.054x: right on the target line, and the
baseline varies more round to round (±0.6%) than ours (±0.2%).

### 3.2 Pipeline occupancy (msprof, 4096³, single core, us)

| Total | Cube busy | MTE1 busy | MTE2 busy | Fixpipe | MTE2 instr/core | GM→L1/core |
|-------|-----------|-----------|-----------|---------|-----------------|------------|
| 421.7 | 393.0 (93%) | 303.0 (72%) | 405.4 (96%) | 29.8 | 706 | 66.0 MB |

- L2 read hit 96.1%; L1 read bandwidth 153 GB/s, profiler reports 58% MTE2
  bandwidth utilisation — nd2nz with 512 B row segments and an 8 KB row stride
  cannot saturate DMA. That 58% is measured against the single-core theoretical
  peak, which is unreachable with all cores running.
- Tail wave: 512 blocks / 24 cores = 21.33, so 8 cores run one extra block — those
  cores show cube busy 393.0 us in a 421.7 us window, the other 16 show 375.0 us /
  405~416 us; the tail is ≈ 4%.

### 3.3 Tuning log (4096³, same setup, us)

| Variant | Median | Verdict |
|---------|--------|---------|
| Adopted: 128x256, L1_K=256, L0_K=64, 2 stages, swizzle 1/0 | 424.2 | — |
| shuffleK off (every core starts at K=0) | 456.2 / 460.0 | 7% slower, neighbouring cores contend for the same K slice in L2 |
| BLK_M=256, BLK_N=128 (same traffic as adopted) | 555.5 / 560.0 | 31% slower, B row segment shrinks 512 B → 256 B |
| BLK_N=128 (+33% traffic) | 675.0 / 676.6 | 59% slower, confirms we are traffic-bound |
| swizzle = 2/4/8, dir 1 | 428.5 ~ 438.6 | no gain (one M row is 16 blocks, the 1 MB A panel stays in L2) |
| Last M row split into N halves (fills all 24 cores) | 432.1 / 431.6 | 1.8% slower: each half re-reads the A panel, the extra fixed cost outweighs the tail gain |

Conclusions:

- The tile can only be 64x512 / 128x256 / 256x128 / 512x64 (L0C holds 32K elements
  and the tile must divide 4096); 128x256 has the lowest traffic of the four,
  measured GM→L1 = 1.61 GB.
- Cube 93% and MTE2 96% are both near saturation and mask each other; the rest is
  L2→L1 latency and the tail wave.
- MTE2 already sits at the measured ceiling for this access pattern. The only
  remaining lever is double buffering and scheduling under the current tile and
  row-major input, and the table above exhausts it.

## 4. Reproducing

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh   # adjust to your install path
bash run_benchmark.sh [device_id]                # default device 0
```

The script builds (offline, no external dependencies) → samples baseline and ours
with `msprof op --warm-up=3 --launch-count=10` → prints both 10-sample sets, the
medians and the speedup.
Correctness (the CPU reference takes ~10 min):

```bash
./build_artifact/build/our_matmul 4096 4096 4096 [device_id]
# expect: Compare success.
```

## 5. Implementation

Hand-written AscendC kernel (`src/ours/op_kernel/our_matmul_kernel.h` +
`entry.cpp`), no Matmul high-level API; host side in `src/ours/host/our_main.cpp`.
Tile shapes follow the 910B L1 / L0 / L0C capacities, every buffer level is double
buffered, and MTE2/MTE1 traffic overlaps Cube compute (measured occupancy: 3.2).

### 5.1 Tiling and buffers

| Level | Shape (M x N x K) | Depth | Footprint |
|-------|-------------------|-------|-----------|
| L1 A | 128 x 256 | 2 | 2 x 64 KB |
| L1 B | 256 x 256 | 2 | 2 x 128 KB |
| L0A  | 128 x 64  | 2 | 2 x 16 KB |
| L0B  | 64 x 256  | 2 | 2 x 32 KB |
| L0C  | 128 x 256 (FP32) | 1 | 128 KB |

- Output tile is `128x256`; K is split into `L1_K=256` tiles, each fed to the Cube
  in `L0_K=64` steps.
- A and B both enter L1 in 16x16-fractal zN layout, each with a **single**
  `DataCopy(Nd2NzParams)` call for the whole tile: A uses `nValue=128, dValue=256,
  dstNzC0Stride=128`; B uses `nValue=256, dValue=256, dstNzC0Stride=256,
  dstNzNStride=1`.
- L1→L0A uses a non-transposed `LoadData2D` (one call per 16 rows,
  `srcStride=8` fractals); L1→L0B uses a transposed `LoadData2D`
  (`ifTranspose=true`, one call per 16 K rows).

### 5.2 Pipeline

```text
GM(A) --MTE2--> L1A[2] --MTE1--> L0A[2] --\
GM(B) --MTE2--> L1B[2] --MTE1--> L0B[2] ---Mmad--> L0C --Fixpipe--> GM(C)
```

- **Prefetch**: while computing the current K tile, MTE2 loads the next K tile into
  the other L1 stage; during a block's last K tile it also prefetches the first K
  tile of this core's next output block (cross-block prefetch).
- **L0 double buffering**: two copies each of L0A/L0B, paired by `M_MTE1` /
  `MTE1_MTE2` / `MTE1_M` events.
- **L0C dependency**: Mmad uses `unitFlag=0b10`, and the block's last `0b11`
  triggers Fixpipe to drain L0C to GM; cross-block read-after-write on L0C is
  ordered in hardware by the unit flag, with no software flag.
- **shuffleK**: each core's K start is `blockIdx % kTiles`, so cores do not read the
  same K slice at the same time.
- **Block order**: serpentine along N inside one M row (`swizzle=1, dir=0`), keeping
  the A panel (1 MB) in L2 across its 16 N blocks; windows of 2/4 rows or walking N
  by column are all slower (see 3.3).
- Coverage: M/N/K multiples of 16, with K, N < 65536.

### 5.3 Trade-off: copy granularity of L1 B

B was initially split into one `nd2nz` per 16 K rows (16 calls per tile), which cost
5986 MTE2 instructions/core, reached only 41% GM→L1 bandwidth utilisation and took
605 us per core; a single `nd2nz` for the whole tile gives 706 calls/core, 58%
utilisation and ≈ 428 us per core (see 3.2).

## 6. Layout

```text
gemm/
├── README.md          # this file
├── src/
│   ├── baseline/torch_bench.py     # baseline: torch-npu matmul, back-to-back launches
│   └── ours/                       # our implementation
│       ├── CMakeLists.txt          # standalone build (CANN only)
│       ├── host/our_main.cpp       # host launch, data generation and verification
│       └── op_kernel/              # kernel entry / body / tiling
└── run_benchmark.sh   # one-shot reproduction
```
