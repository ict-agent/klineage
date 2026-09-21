# Ascend generation runs (task 2) — how to run them

For the next SparseAttention and FusedAddRmsNorm runs, use [NEXT_RUNS.md](NEXT_RUNS.md).
The host, model configuration, language policy and trace protocol there supersede
the historical defaults below. SparseAttention and FusedAddRmsNorm runs are complete; see their audited READMEs.

Scope: the runs published in this directory. One Codex session per
(kernel, setting) writes an Ascend kernel for the 910B1. This file is the
operator's runbook and the contract a session's agent is held to.

Setting names map to the task: `without_memory` = A (bare prompt),
`with_memory` = B (same prompt plus the expert pack). Same gate, same budget;
the expert material is the only difference.

```text
Codex (Mac) --Responses API--> UsageProxy --HTTPS--> provider
    |                             `--> events.jsonl   (api_response)
    | codex exec --json
    v
trace.jsonl
    |
    | agent runs, from the unit root:
    |   .venv/bin/python scripts/ascend/eval.py --work work --device <npu>
    v
eval.py --rsync--> 910b1 --> docker exec --> evaluate --> NPU
    |                                          |
    |                                          `--> remote .klineage/versions/<N>
    `--> events.jsonl (evaluate) + versions/version<N>
```

## 1. Prerequisites

- Host `910b1`, container `vllm0.23.0-zcj` (torch-npu, CANN 9.1.0). The
  harness lives in `~/ascend-harness` there: `repo/` (synced checkout) and
  `runs/` (one evaluation copy per unit).
- Local venv `.venv-ascend` owns klineage. Codex runs on this Mac; no Codex CLI
  is installed on the host or in the container.
- Problem packages: `experiment/<kernel>/problems/{definitions,workloads}`.
  Data the packages never shipped is generated once, seed 0, identically for
  both settings:

  ```sh
  .venv-ascend/bin/python scripts/ascend/gen_inputs.py --kernel top_p [--force]
  ```

- Expert packs live in `experiment/ascend/generation/<kernel>/with_memory/expert/`
  and carry their provenance in `source.json` (repository, revision, path).
  `with_memory` sessions read a copy at `work/expert/`; `without_memory` runs
  must stay free of these files.

## 2. Bring the remote up

```sh
scripts/ascend/run.sh sync          # rsync this checkout to 910b1:~/ascend-harness/repo
scripts/ascend/run.sh bootstrap     # pip install -e . inside the container
scripts/ascend/run.sh probe         # remote platform string
scripts/ascend/run.sh smoke 2       # one remote evaluate on NPU 2
```

`sync` mirrors a sanitized subset: the expert packs, the local batch scripts
and the CUDA transfer runs are excluded, so an agent that reads the remote
checkout cannot find them.

## 3. Start a run

```sh
scripts/ascend/run.sh start --kernels kda \
  --settings without_memory,with_memory --devices 2,3 --timeout 7200
```

- One unit = one Codex session = one NPU. `--timeout` is the whole budget in
  seconds (7200 = 2 h); at expiry the process group is killed and whatever the
  unit measured so far is still packaged, with `status: error` in `result.json`
  to mark the kill.
- The workspace is `~/klineage-runs/<kernel>/<setting>/work`, deliberately
  outside this checkout: Codex walks up from its cwd for `AGENTS.md`, so a
  workspace inside the repo would expose the expert packs and the other setting.
- Other kernels in the batch: `sparse_attention`, `top_p`, `fused_add_rmsnorm`,
  `gqa` (`--kernels` takes a comma list; a unit without an expert pack cannot
  run `with_memory`).
- `scripts/ascend/run.sh status` shows progress, `logs` tails the batch log.
  Units with a `result.json` are skipped on restart (`--retry-failed` reruns
  them, `--dry-run` only prints the plan).
- A session that ended before its budget (agent declared convergence, provider
  error) can continue the same thread for the remaining time:

  ```sh
  scripts/ascend/run.sh resume --kernel kda --setting without_memory
  ```

## 4. What the agent gets

- `work/AGENTS.md` (rendered from `scripts/ascend/agents.md.tmpl`) and the
  prompt (`scripts/ascend/prompt.md.tmpl`): two phases — land a correct version
  as fast as possible, then spend the rest of the budget on one measured change
  per gate call, keeping only what passes and gets faster.
- `work/problems/{definitions,workloads}` and, for `with_memory`,
  `work/expert/` plus a line in `AGENTS.md` telling it to read that first.
- `work/.agents/skills/{bench,ascendc}`: the gate documentation and the bundle
  format spec (`src/klineage/skills/ascendc/SKILL.md`).
- Its copies of `scripts/` and `src/` are read-only and path-scrubbed: the gate
  is the only measurement, and the checkout must not be reachable by path.

## 5. Contract

Bundle: `work/submission/config.toml` + `work/submission/solution/`, holding
the artifact the language pins ask for (`language`, `entry_point` in
`config.toml`). `kda` ran in triton-ascend (`language = "python"`), `top_p` in
AscendC (`language = "ascendc"`, `kernel.asc::kernel`). A python bundle must
launch a device kernel; a torch-only bundle is rejected.

Gate: exactly one command, called from the unit root after every meaningful
change (the unit carries its own interpreter copy at `.venv/`):

```sh
.venv/bin/python scripts/ascend/eval.py --work work --device <npu>
```

It serializes `submission/` into `kernel.json`, rsyncs the workspace to 910b1,
compiles and measures in the container, freezes the evaluated kernel under
`work/versions/version<N>/` (with the submitted bundle next to it), appends one
`evaluate` record to `work/.klineage/stats/events.jsonl` and prints the
`ValidationResult` JSON. Never write a private timer: that latency is the only
performance signal.

Measurement: device-side latency of one operator call, timed with NPU stream
events on the current stream — 10 warmup + 50 timed iterations, 3 trials,
median of trial medians (`TimingPolicy` in `src/klineage/harness/timing.py`).
Compilation and input preparation are outside the timed interval. The baseline
is the definition's torch reference on the same NPU under the same policy,
measured once per device and cached in `baseline.json` next to the unit;
`speedup = baseline_ms / latency_ms`. `eval.py --baseline` only refreshes that
cache.

Correctness: output shape, dtype and device as the definition asks, then
`torch.allclose(rtol=atol=1e-2)` (`NUMERICAL_TOLERANCE`,
`src/klineage/constants.py`), and the definition's own `check_outputs` when it
defines one — `top_p` uses that to demand the exact support and rows summing
to one. Only passing versions count; rejected ones still cost tokens.

## 6. Artifacts

Per unit (`~/klineage-runs/<kernel>/<setting>/`), and afterwards per setting in
this directory:

```text
<kernel>/<setting>/
|- events.jsonl      api_response (proxy: timestamps, tokens) + evaluate (gate)
|- trace.jsonl      Codex session trace (codex exec --json)
|- baseline.json    cached torch-npu baseline per device
|- versions/version<N>/   kernel bundle of every gate call
|- work/            agent workspace (AGENTS.md, problems, expert, submission)
|- submission/      final bundle copy
|- result.json      status, counters, duration
`- final_message.txt   the agent's last message
```

The task's three mandatory files are `events.jsonl`, `trace.jsonl` and
`versions/`; the extra files are what makes a run auditable.

## 7. Collect and plot

```sh
.venv-ascend/bin/python scripts/ascend/collect.py --kernel kda
.venv-ascend/bin/python scripts/ascend/plot.py --root ~/klineage-runs \
  --out experiment/ascend/generation/plots --x tokens --y latency   # or --x seconds
```

`collect.py` rebuilds `experiment/ascend/generation/<kernel>/` from the run
root (keeping `expert/`), expands every frozen `kernel.json` into a readable
`versions/version<N>/submission/` and regenerates `README.md` from
`events.jsonl`. `plot.py` writes one CSV per setting, `table.md` and a curve;
it emits every kernel it finds under `--root`, so point it at a root holding a
single kernel (a directory of symlinks) when producing per-kernel plots.

## 8. Reproducing the published runs

```sh
scripts/ascend/run.sh sync && scripts/ascend/run.sh bootstrap
# inputs the packages never shipped, once per kernel
.venv-ascend/bin/python scripts/ascend/gen_inputs.py --kernel kda
.venv-ascend/bin/python scripts/ascend/gen_inputs.py --kernel top_p

# one batch per kernel: A and B run side by side, one NPU each
scripts/ascend/run.sh start --kernels kda --devices 2,3 \
  --settings without_memory,with_memory
scripts/ascend/run.sh start --kernels top_p --devices 2,3 \
  --settings without_memory,with_memory
scripts/ascend/run.sh status

# package the runs: artifact tree + README + curves
.venv-ascend/bin/python scripts/ascend/collect.py --kernel kda
.venv-ascend/bin/python scripts/ascend/collect.py --kernel top_p
```

Results of the runs published here, and the deviations that apply to them, are
in `README.md` next to this file.

## 9. Gotchas

- A setting keeps the NPU it started on for the whole batch: a second batch on
  the same devices collides with the first. Check `run.sh status` first.
- Devices differ between settings (A on NPU 2, B on NPU 3 in the published
  runs), so each is scored against its own baseline; speedups are comparable,
  absolute latencies are not.
- `~/klineage-runs/.../work` is deleted and re-created by `prepare()` on the
  next start of that unit. `baseline.json` survives on purpose — it is a
  property of the device, not of the run.
- The container runs as root and leaves root-owned files under `runs/`, which
  blocks the next `rsync --delete`; `run.sh sync` hands the tree back to the
  host user first.
- A long first gate call is the baseline measurement (the torch reference is
  50-500× slower than a device kernel, minutes per protocol), not a hang.
