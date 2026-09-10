# KLineage

KLineage extracts reusable optimization skills from expert accelerator kernels.
Codex removes one optimization per Decompose step and saves it as `SKILL.md`.
The full workflow then selects and applies skills using measured bottlenecks.

Supports CUDA, Hygon HIP, and AscendC.

## Build

Requires Python 3.12+, uv, Codex, and the backend's hardware, toolchain, and
PyTorch runtime. CUDA evaluation requires FlashInfer with CUPTI 13+.

```bash
uv venv --system-site-packages
uv sync
uv build
codex login
```

## CLI

Choose a [problem](problems/README.md) and replace the expert paths below.

Extract skills into a memory directory. Verification defaults to off; stdout is
a JSON array of saved SKILL.md paths.

```bash
uv run klineage-init-memory \
  --problem problems/definitions/gemm.json \
  --repo /path/to/expert-repo \
  --expert-kernel path/inside/repo/kernel.cu \
  --memory-dir memory \
  --workdir agent-workspace/extract
```

Run extraction followed by skill selection and application. Verification defaults
to on; stdout is the final kernel as JSON.

```bash
uv run klineage-workflow \
  --problem problems/definitions/gemm.json \
  --repo /path/to/expert-repo \
  --expert-kernel path/inside/repo/kernel.cu \
  --workdir agent-workspace/run \
  --max-decompose-step 15 \
  --max-apply-step 15
```

Optimize an existing kernel using a memory directory:

```bash
uv run klineage-optimize \
  --start_kernel agent-workspace/run/decompose/0 \
  --memory_dir memory \
  --workdir agent-workspace/optimize
```

Omit `--memory_dir`, or pass an empty/whitespace value, to run the independent
optimization baseline:

```bash
uv run klineage-optimize \
  --start_kernel agent-workspace/run/decompose/0 \
  --workdir agent-workspace/baseline
```

Baseline steps choose their own optimization without reading or mounting memory
or producing SKILL.md. Changed kernels continue to the next step; an unchanged
kernel or the step limit stops the run. An existing empty memory directory keeps
skill selection enabled and produces an unchanged result when no card applies.

The input is a `kernel.json` file or its directory, with the complete problem and
source bundle; a bare source directory is insufficient. In memory mode, Apply reads
SKILL.md files through `.agents/skills/memory` and selects one optimization per step.
Ranking memory candidates currently requires CUDA and Nsight Compute. Optimize
enables verification by default and writes the final kernel as JSON to stdout.
Every handoff preserves the problem and ordered input/output ABI, including when
verification is disabled.

Workdirs must not already exist. Extraction and workflow generate one if omitted;
optimize requires `--workdir`, outside the memory directory when one is supplied.
Use `--verifier` / `--no-verifier` to override verification defaults.

```bash
uv run klineage-init-memory --help
uv run klineage-workflow --help
uv run klineage-optimize --help
```

See the [workflow contract](src/klineage/message/workflow.md) for artifacts,
stop rules, and failure handling.
