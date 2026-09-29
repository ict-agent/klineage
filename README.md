# KLineage

KLineage extracts reusable optimization skills from expert accelerator kernels.
Codex removes one optimization per Decompose step and saves it as `SKILL.md`.
The full workflow uses those skills to guide kernel optimization rounds.

Supports CUDA, Hygon HIP, and AscendC.

## Paper

*KLineage: Recovering the Missing When of Kernel Optimization by Deoptimizing
Experts.* Source and figures are in [`paper/`](paper); build it with
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Its measurements live on several branches, one per operator study.
[`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) maps every reported number to the branch
and directory that holds it.

## Repository

| Path | Contents |
| --- | --- |
| `src/klineage/` | Framework: actions, harness, memory, messages, per-language skills |
| `problems/` | Five problem definitions, workloads, and fixed input samples |
| `skillcards/` | Induced CUDA skill cards, one `SKILL.md` per optimization |
| `experiment/` | Transfer runs, one directory per timestamped run |
| `paper/` | arXiv source, flat layout |
| `docs/` | Where the paper's results live |
| `tests/` | Test suite |

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

Run extraction followed by kernel optimization rounds. Verification defaults
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

Each Apply round can combine techniques and iterate on measured results. Memory
provides optional guidance; an empty directory does not stop optimization. Apply
emits the best validated kernel, without SKILL.md. An unchanged kernel ends the
loop; `--max-apply-step` limits rounds.

The input is a `kernel.json` file or its directory, with the complete problem and
source bundle; a bare source directory is insufficient. When supplied, memory is
read through `.agents/skills/memory`. CUDA/NCU counter profiling is optional. Optimize
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
