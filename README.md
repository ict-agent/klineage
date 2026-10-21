# KLineage
Klineage is an agentic harness for accelerator kernel generation and optimization, with reusable skills extracted from expert implementations.
Klineage builds upon the ability of existing harness like Codex and Claude Code. It is designed to provide minimal abstraction for artifact organization, knowledge management and benchmarking
for kernel agents across hardwares.

[Documentation](https://qiuchu.site/project/klineage/) · [Paper](https://arxiv.org/abs/2605.28213) · [API reference](https://qiuchu.site/project/klineage/reference/artifacts.html)

## Quick start

Requires Python 3.12+, uv, Codex or native Claude Code, and Linux bubblewrap
with user namespaces.
From the repository root, install the CLI with one backend:

```bash
uv tool install '.[cuda]' --python 3.12
```

Use `.[ascend]` or `.[hygon]` for those backends. Each extra installs its Python
dependencies; configure the matching system driver and toolchain separately.
See [runtime requirements](docs/getting-started.md).
If the commands are not on PATH, run `uv tool update-shell` and reopen the terminal.

Codex is the default runner. Its model and provider settings come from `$CODEX_HOME/config.toml`
(`~/.codex/config.toml` when `CODEX_HOME` is unset). These overrides are optional:

```bash
export KLINEAGE_API_BASE_URL=https://provider.example/v1
export KLINEAGE_API_KEY_ENV=MODEL_API_KEY
export KLINEAGE_MODEL=provider-model-id
```

When overriding the key variable, populate `MODEL_API_KEY` on the host.

For an Anthropic Messages-compatible endpoint, select Claude Code:

```bash
export KLINEAGE_RUNNER=claude-code
export ANTHROPIC_BASE_URL=https://provider.example/anthropic
export ANTHROPIC_MODEL=provider-model-id
export ANTHROPIC_API_KEY=your-key
```

Use `ANTHROPIC_AUTH_TOKEN` instead for bearer authentication. `KLINEAGE_MODEL`,
`KLINEAGE_API_BASE_URL` and `KLINEAGE_API_KEY_ENV` override the Anthropic variables;
clear previous overrides when switching providers.
See [model access](docs/getting-started.md#configure-model-access).

Generate a kernel from a problem directory containing `definition.json` and
at least one workload in `workloads.jsonl`:

```bash
klineage-generation problems/gemm \
  --workdir agent-workspace/gemm --timeout 600
```

By default, the command detects the backend and starts local evaluation. Add
`--server http://127.0.0.1:9100` to use an existing evaluator. It prints the accepted
Kernel as JSON; sources and per-workload results remain under `versions/`.
See [Getting started](docs/getting-started.md) for setup and output details.
