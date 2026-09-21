# Ascend generation automation

One local Codex session per (kernel, setting); evaluation runs on 910b1 inside
the `vllm0.23.0-zcj` container through the fixed gate `scripts/ascend/eval.py`.
Every path below is relative to the project root; on the host the harness lives
in `~/ascend-harness` (`repo/` checkout, `runs/` evaluation copies).

```text
Codex (Mac) --Responses API--> UsageProxy --HTTPS--> provider
    |                             `--> events.jsonl   (api_response)
    | codex exec --json
    v
trace.jsonl
    |
    | agent runs, from the project root:
    |   .venv-ascend/bin/python scripts/ascend/eval.py \
    |     --work ~/klineage-runs/<kernel>/<setting>/work
    v
eval.py --rsync--> 910b1 --> docker exec --> evaluate --> NPU
    |                                          |
    |                                          `--> remote .klineage/versions/<N>
    `--> events.jsonl (evaluate) + versions/version<N>
```

Usage:

```sh
scripts/ascend/run.sh sync            # push repo (needed for remote eval)
scripts/ascend/run.sh bootstrap       # install klineage in the container
scripts/ascend/run.sh probe           # print remote platform string
scripts/ascend/run.sh smoke 2         # remote evaluate smoke test
scripts/ascend/run.sh start --kernels kda --settings without_memory,with_memory --devices 2,3,4,5
scripts/ascend/run.sh status          # batch progress
scripts/ascend/run.sh logs            # batch log
scripts/ascend/run.sh plot            # latency vs tokens/time curves
```

Rules:

- The local venv `.venv-ascend` owns klineage; create it with
  `uv venv --python 3.12 .venv-ascend && uv pip install --python .venv-ascend/bin/python -e .`.
- `with_memory` units read `experiment/ascend/generation/<kernel>/with_memory/expert/`;
  `without_memory` must stay free of those files.
- One unit = one 2h Codex session; finished units are skipped on restart
  (`--retry-failed` reruns them).
