# Development

Use a dedicated documentation environment for site changes and a backend
environment for code that executes on a device.

## Source layout

Keep implementation beside the component that owns it:

| Directory | Contents |
| --- | --- |
| `src/klineage/artifact/` | File trees, artifact formats and repository inputs |
| `src/klineage/backend/` | Device selection, kernel builders and timing |
| `src/klineage/action/` | Action execution and result collection |
| `src/klineage/prompts/` | Prompt templates, action output descriptions and rendering |
| `src/klineage/harness/` | Agent sessions, isolation and evaluation workers |
| `src/klineage/server/` | Evaluation service, client and transport |
| `src/klineage/memory/` | Skill cards, storage and retrieval |
| `src/klineage/cli/` | Command implementations and shared options |

Tests and repository scripts live in `tests/` and `scripts/`. Local experiments
and generated outputs under `experiment/` are ignored by Git.

## Build the documentation

Start the local documentation server:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --only-group docs mkdocs serve
```

Open <http://127.0.0.1:8000>. To generate static HTML:

```bash
UV_PROJECT_ENVIRONMENT=.venv-docs uv run --only-group docs mkdocs build --strict
```

The result is `site/index.html`. API pages are generated from Python source
through static analysis, so the build needs no device libraries or compiler.
Use the local server when testing browser search. The generated site and docs
environment are ignored by Git.

## Deploy the documentation

The **Documentation** workflow checks pull requests and publishes relevant
`master` changes to <https://qiuchu.site/project/klineage/>. It can also be run
manually from the Actions tab on `master`.

Deployment uses the repository secret `DOCS_SSH_PRIVATE_KEY` and a dedicated
`klineage-docs` account on Tencent. The account needs write access to
`/var/www/klineage-docs`, without sudo. The server's SSH host key is pinned in
`.github/docs_known_hosts`.

Each build uploads a release, then switches the `current` symlink. Previous
releases remain under `/var/www/klineage-docs/releases/` for rollback.
Nginx serves `current/` at `/project/klineage/`; later deployments need no reload.
CI checks the server's HTTPS response over SSH against the built homepage.

## Write documentation

Use [Overview](index.md) and [Getting started](getting-started.md) as style references:
introduce the purpose, show a concrete example, then link to the next step.
Keep implementation details near the API or extension that needs them.

- Put guides in `docs/` and public API docstrings beside the code. Add new pages
  to `mkdocs.yml`.
- Use short paragraphs, task-based headings and tables for comparable options.
  State prerequisites before an example.
- Use Google-style docstrings for arguments, results and constraints. API pages
  include them through `::: klineage...` directives.
- Link to existing explanations instead of repeating them. Use relative links;
  a strict build checks pages and section anchors.
- Keep diagrams as editable SVGs in `assets/diagrams/`, with a title and description.
  Use `figure.diagram` and reserve diagrams for the main relationships.

## Run tests

In a development environment with KLineage, PyTorch and NumPy installed:

```bash
uv run --no-sync python -m unittest discover -s tests
```

CPU tests cover artifact formats, transport, dispatch and isolation. Set
`KLINEAGE_CUDA_TESTS=1` in a configured CUDA environment to enable optional CUDA
tests. New backends also need the
[hardware checks](guides/extending.md#check-the-implementation).

## Run a bounded campaign

`scripts/server_campaign.py` runs generation and an independent service audit
under one total deadline. It requires GNU `timeout` and a worker environment
containing KLineage and the chosen backend.

The default model is `DeepSeek-V4.1-Flash` at the configured AIPing endpoint.
Set `AIPING_API_KEY` on the host, then run from the repository:

```bash
uv run --no-project --with-editable . python scripts/server_campaign.py \
  --server-python /opt/device-env/bin/python \
  --backend cuda --problem problems/fmha \
  --device-node /dev/nvidia0 --device-node /dev/nvidiactl \
  --device-node /dev/nvidia-uvm --budget 600
```

The ten-minute budget includes setup and audit; generation stops two minutes
before the deadline. Results, traces and source hashes are saved in a fresh
directory under `agent-workspace/`.

Use `--model`, `--base-url` and `--output` to override defaults, and `--agent-*`
options to configure local diagnostics. The campaign passes its model explicitly,
overriding `KLINEAGE_MODEL`. Run the script with `--help` for all options.
