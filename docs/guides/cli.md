# Command-line tools

KLineage provides four commands for generating kernels, extracting skills,
optimizing implementations and hosting evaluation. Install them using
[Getting started](../getting-started.md).

| Command | Purpose | Output |
| --- | --- | --- |
| `klineage-generation` | Generate a kernel from a problem directory | Kernel JSON |
| `klineage-init-memory` | Extract optimization skills from an expert | Memory Artifact JSON |
| `klineage-optimize` | Improve an existing kernel | Final Artifact JSON |
| `klineage-server` | Host evaluation for a set of problems | Service endpoint information |

The first three commands use the model configuration described in
[Configure model access](../getting-started.md#configure-model-access).
Set `KLINEAGE_RUNNER=claude-code` for Anthropic Messages or leave it unset for
Codex/Responses. The evaluator configuration is independent of that choice.
Use a fresh work directory for each run.

## Generate a kernel

Pass a directory containing `definition.json`, `workloads.jsonl` and any fixed
inputs. Generation requires at least one workload:

```bash
klineage-generation problems/gemm \
  --workdir agent-workspace/gemm --timeout 600
```

By default, the command detects a local backend and starts evaluation. Use
`--server` or `KLINEAGE_SERVER` to select an existing evaluator. Remote evaluation
needs no local accelerator; the input must match a registered problem, whose
original directory must also be accessible on the controller.

The CLI passes the loaded Problem to the `Generation` action, which authors and
evaluates candidates and returns the selected Kernel in its Artifact.

| Argument | Meaning |
| --- | --- |
| `PROBLEM_DIR` | Problem directory with its definition and workloads |
| `--workdir PATH` | Fresh output directory; defaults to `experiment/generation-<id>` |
| `--timeout SECONDS` | Time per agent attempt and evaluation request |
| `--max-retries N` | Additional attempts after failure |
| `--server ENDPOINT` | Existing evaluator; otherwise use `KLINEAGE_SERVER` or start locally |
| `--problem-id ID` | Registered problem; defaults to the matching input name or sole task |

A successful run prints Kernel JSON. Complete versions, measurements and agent
traces stay in the work directory; see the
[result layout](../getting-started.md#generate-a-kernel-using-klineage).
Python diagnostics go to stderr. The equivalent module command is
`python -m klineage.generation PROBLEM_DIR`.

## Extract skill memory

Start an evaluator with the matching problem, then supply an expert repository
and the source entry to adapt:

```bash
klineage-init-memory \
  --server http://127.0.0.1:9100 \
  --problem problems/gemm \
  --repo /path/to/expert-repository \
  --expert-kernel path/inside/repository/kernel.cu \
  --memory-dir memory \
  --workdir agent-workspace/extract \
  --max-decompose-step 15
```

Init adapts the expert, then Decompose extracts skills until the implementation
stops changing or reaches the step limit. Each card is saved to `memory` before
the next round. Replace the expert paths with your own. If it needs external
headers, add their evaluator-host paths to the server's `--include` options.

## Optimize a Kernel

Start from a Kernel directory, an earlier action directory, or serialized
Kernel/action JSON:

```bash
klineage-optimize \
  --server http://127.0.0.1:9100 \
  --start-kernel agent-workspace/extract/decompose/0 \
  --memory-dir memory \
  --workdir agent-workspace/optimize \
  --max-apply-step 15
```

Apply repeats until the implementation stops changing or reaches the step limit.
Omit `--memory-dir` to optimize without memory. `--max-apply-step 0` returns the
input without running an action.

A Kernel directory contains `metadata.json` and `submission/`. An action directory
supplies its last numbered `versions/` snapshot. Keep the optimization work
directory outside the memory directory.

## Start an evaluation server

Use `klineage-server --server ENDPOINT` to choose its listener. For a complete
startup command, problem collections and device configuration, follow the
[Evaluation guide](evaluation.md#start-a-cuda-evaluation-server).

`KLINEAGE_SERVER` selects client endpoints; the server reads its listener from
the command line. `KLINEAGE_PROBLEM` selects the problem directory or collection
served by that process.

## Shared service and agent options

Generation, extraction and optimization accept these options:

| Option | Meaning |
| --- | --- |
| `--server ENDPOINT` | Evaluation endpoint; overrides `KLINEAGE_SERVER` |
| `--problem-id ID` | Select a registered problem explicitly |
| `--timeout SECONDS` | Time per agent attempt and evaluation request |
| `--max-retries N` | Additional attempts after the first failure |
| `--agent-python PATH` | Interpreter exposed to the agent |
| `--agent-device-node PATH` | Device for local diagnostics; repeat for each node |
| `--agent-read-only PATH` | Additional runtime or tool directory; repeat as needed |
| `--agent-env NAME=VALUE` | Agent runtime environment variable |

Extraction can choose a work directory automatically; optimization requires
`--workdir`. Timeouts apply to individual attempts, so retries and later rounds
can increase total elapsed time. See [Runtime settings](evaluation.md#deadlines-runtime-information-and-credentials)
for service time limits and [Optional device diagnostics](evaluation.md#optional-device-diagnostics)
for agent device access.

## Run a bounded campaign

For a generation run and independent audit under one total deadline, use the
repository's [campaign script](../development.md#run-a-bounded-campaign).

## Command help

Each command lists its full options:

```bash
klineage-generation --help
klineage-init-memory --help
klineage-optimize --help
klineage-server --help
```
