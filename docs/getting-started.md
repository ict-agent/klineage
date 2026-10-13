# Getting started

Klineage requires Linux, Python 3.12+, uv, and bubblewrap with user namespaces.
Install either Codex or native Claude Code (tested with 2.1.267, requiring `--bare`)
on PATH. The runner is independent of the accelerator backend.

## Install Klineage

Choose one backend extra and install from this repository. For CUDA:

```bash
uv tool install '.[cuda]' --python 3.12
```

Replace `cuda` with the extra for your device; these extras are mutually exclusive:

| Extra | Python dependencies | Required system stack |
| --- | --- | --- |
| `cuda` | PyTorch 2.14.x from the CUDA 13.0 index, FlashInfer 0.7, CUPTI 13.0, Ninja | Compatible NVIDIA driver, CUDA toolkit and C++ compiler |
| `ascend` | CPU PyTorch 2.9.0, `torch-npu` 2.9.0, NumPy | Ascend driver, CANN 8.5 and a C++17 toolchain |
| `hygon` | PyTorch `2.9.0+das.opt1.dtk2604`, NumPy, Ninja | Hygon driver and DTK 26.04 |

## Configure model access

### Codex / Responses API

`KLINEAGE_RUNNER=codex` is the default. KLineage reads model and provider settings from
`$CODEX_HOME/config.toml`, defaulting to `~/.codex/config.toml`. It supports a
configured host API-key variable or Codex login saved in `auth.json`.
These environment overrides are optional:

```bash
export KLINEAGE_API_BASE_URL=https://provider.example/v1
export KLINEAGE_API_KEY_ENV=MODEL_API_KEY
export KLINEAGE_MODEL=provider-model-id
```

When using these overrides, replace the endpoint and model ID and populate
`MODEL_API_KEY` on the host. Explicit Python runner arguments take precedence
over these variables, which take precedence over config. See the
[Evaluation guide](guides/evaluation.md#deadlines-runtime-information-and-credentials)
for credential handling.

### Claude Code / Anthropic Messages API

Select Claude Code for Anthropic-compatible endpoints:

```bash
export KLINEAGE_RUNNER=claude-code
export ANTHROPIC_BASE_URL=https://provider.example/anthropic
export ANTHROPIC_MODEL=provider-model-id
export ANTHROPIC_API_KEY=your-key
```

For `Authorization: Bearer` authentication, use `ANTHROPIC_AUTH_TOKEN` instead
of `ANTHROPIC_API_KEY`. If both are set, the auth token takes precedence.
The default endpoint is `https://api.anthropic.com`.
Supply the gateway root, optionally ending in `/v1`; the bridge forwards
`/v1/messages` and `/v1/messages/count_tokens`, including `?beta=true`.
The provider must support streaming Messages and tool use.

| Variable | Meaning |
| --- | --- |
| `KLINEAGE_RUNNER` | `codex` (default) or `claude-code` |
| `KLINEAGE_MODEL` | Override `ANTHROPIC_MODEL` |
| `KLINEAGE_API_BASE_URL` | Override `ANTHROPIC_BASE_URL` |
| `KLINEAGE_API_KEY_ENV` | Host variable holding the selected credential |
| `KLINEAGE_AUTH_MODE` | Claude authentication: `api-key` or `bearer` |

Custom key variable names default to `api-key`. Set `KLINEAGE_AUTH_MODE=bearer`
when the gateway expects a bearer token. Changing the endpoint requires an
explicit key variable or Python `api_key`; credentials from another endpoint
are not inherited. Clear stale `KLINEAGE_*` overrides when switching providers.
Explicit runner arguments take precedence over these variables.

Claude Code runs in `--bare` mode with isolated session storage. KLineage does
not load host Claude settings, hooks, MCP servers, subscription login or keychain
credentials. The credential stays in the host bridge; the CLI receives a
placeholder key and a sandbox-local endpoint. Use a native `claude` executable
that supports `--bare`, `--print`, `--output-format stream-json` and `--resume`.

The selection applies to generation, extraction and optimization. See
[Runtime API](reference/runtime.md) for direct Python use.

## Generate a kernel using Klineage

Klineage uses the [FlashInfer Trace](<https://github.com/flashinfer-ai/flashinfer-trace>)
format for problem definitions and workloads. The repository includes
`problems/gemm`, with the following structure:

```text
problems/gemm/
  definition.json             # Ordered operator ABI and executable reference
  workloads.jsonl             # Required; one Trace record per workload
  inputs/                     # Optional fixed data, referenced relative to this directory
```

Use `klineage-generation` to generate a kernel from a problem directory:

```bash
klineage-generation problems/gemm \
  --workdir agent-workspace/gemm --timeout 600
```

The command runs Generation with the Problem and evaluates the kernel on all
workloads. The problem must have at least one workload. Results are stored as
Kernel artifacts:

| Path under the work directory | Contents |
| --- | --- |
| `current_kernel/` | Candidate scratch directory; may differ from the final Kernel |
| `versions/<N>/` | Automatic complete Kernel snapshots, starting at 0; failed evaluations are retained |
| `versions/<N>/metadata.json` | Validation, per-workload timings and diagnostics |
| `problems/<problem_id>` | Symlink to the original problem directory |
| `workspace/` | Diagnostic scripts, source backups, binaries and logs |
| `workspace/AGENTS.md` | Agent instructions and available Python APIs |
| `workspace/.agents/skills/` | Backend guidance and supplied skill memory |
| `workspace/.klineage/agent-runs/` | Agent prompts, traces, final replies and stderr |
| `workspace/.klineage/agent-runs/<attempt>/stats.jsonl` | Model request usage and evaluation statistics |

Each version keeps its source and evaluation together, so its metadata lets you
compare correctness and latency across workloads and versions. A successful run
returns the last numbered version after every workload passes. The problem link
points to the original inputs; keep those files available.

Codex reuses your existing `CODEX_HOME`; no separate Codex home is created for
each run. Claude Code instead stores resumable sessions under
`workspace/.klineage/claude-home/`. Keep it to resume retries. Artifact layouts
are queried through Python when needed.

Ascend generation keeps NPU access in the evaluation worker because its driver
reserves devices per sandbox. Agents compile locally and submit measurements
through `klineage.evaluate()`; query the device through `klineage.info()`.

## Start an evaluation server

The CLI starts a local evaluator by default. To reuse an evaluator across
generation, skill extraction and optimization, start it separately. For CUDA
device 0, run from the repository root in the installed backend environment:

```bash
klineage-server \
  --backend cuda --problem problems/gemm \
  --root server-work/evaluator --server http://127.0.0.1:9100 \
  --device-node /dev/nvidia0 --device-node /dev/nvidiactl \
  --device-node /dev/nvidia-uvm --env CUDA_VISIBLE_DEVICES=0
```

Keep it running and connect from another terminal:

```bash
klineage-generation problems/gemm \
  --server http://127.0.0.1:9100 --workdir agent-workspace/gemm-server
```

`klineage-init-memory` and `klineage-optimize` also accept `--server`.
For an evaluator on another host, make its registered problem directory accessible
at the same path on the controller. See [Evaluation guide](guides/evaluation.md) for
other runtimes and [Command-line tools](guides/cli.md) for options.

## Choose your next step

- **Extract skills from an expert:** use `klineage-init-memory` with an expert
  repository and source path. See [Command-line tools](guides/cli.md).
- **Optimize an existing Kernel:** use `klineage-optimize`, optionally supplying
  an extracted memory directory.
- **Compose actions in Python:** call `Generation`, `Init`, `Decompose` and `Apply`;
  see [Action Primitive](concepts/actions.md).
- **Author and evaluate a Kernel:** follow the
  [CUDA file layout](concepts/artifacts.md) and submit the completed directory
  with `klineage.evaluate`, which saves the measured Kernel automatically.

Work directories must be fresh. Backend services own acceptance measurements;
an authorized local `Kernel.load_kernel` call builds a callable but does not
establish service acceptance.
