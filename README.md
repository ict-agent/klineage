# KLineage

Generate accelerator kernels and reuse optimization skills.

[Documentation](https://qiuchu.site/project/klineage/) · [API reference](https://qiuchu.site/project/klineage/reference/artifacts.html)

## Quick start

Requires Python 3.12+, uv, Codex and Linux bubblewrap with user namespaces.
From the repository root, install the CLI with one backend:

```bash
uv tool install '.[cuda]' --python 3.12
```

Use `.[ascend]` or `.[hygon]` for those backends. Each extra installs its Python
dependencies; configure the matching system driver and toolchain separately.
See [runtime requirements](docs/getting-started.md).
If the commands are not on PATH, run `uv tool update-shell` and reopen the terminal.

Model and provider settings default to `$CODEX_HOME/config.toml`
(`~/.codex/config.toml` when `CODEX_HOME` is unset). These overrides are optional:

```bash
export KLINEAGE_API_BASE_URL=https://provider.example/v1
export KLINEAGE_API_KEY_ENV=MODEL_API_KEY
export KLINEAGE_MODEL=provider-model-id
```

When overriding the key variable, populate `MODEL_API_KEY` on the host.

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
