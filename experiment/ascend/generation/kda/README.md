# kda — AI-generated Ascend implementation

Both settings run the same prompt, the same gate, and the same budget;
the only difference is the expert material in `work/expert/`.

## Measurement protocol

- Gate: `scripts/ascend/eval.py`, one call per version, evaluated on 910B1
  inside the `vllm0.23.0-zcj` container.
- Timer: NPU stream events (`npu-events`) around one operator call,
  10 warmup + 50 timed iterations, 3 trials, median of trial medians.
- Baseline: the problem's torch reference on the NPU, same timer and
  policy, measured once per device and cached in `baseline.json`.
- Speedup: `baseline_ms / latency_ms`.

## Results

| Kernel | Setting | Correct | Latency (us) | vs. Baseline |
| --- | --- | --- | --- | --- |
| KDA | Without Expert Knowledge | ✓ | 22312.0 us | 74.72× |
| KDA | With Expert Knowledge | ✓ | 23075.3 us | 66.44× |