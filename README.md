# KLineage

Extract accelerator optimization skills from expert kernels, then apply them using
measured bottlenecks and verified SkillCards.

```text
problem + expert -> Init -> Decompose × N -> Apply × N
                                |               ^
                                +-- SkillCards -+
```

Each action runs Codex. Native CUDA, Hygon HIP, and AscendC kernels connect to Python
through pybind11.
[Action message contracts](src/klineage/message/README.md) define inputs, artifact
layouts and downstream handoffs; the prompts include or reference those contracts.

## Setup

Requires Python 3.12, uv, Codex, and the selected backend's toolchain and hardware:

| Backend | Problem language | Build/runtime | Timing |
| --- | --- | --- | --- |
| NVIDIA CUDA | `cuda` | CUDA PyTorch, CUDA toolkit, NVIDIA GPU | FlashInfer with CUPTI 13+ |
| Hygon HIP | `hip` | DTK/hipcc, compatible HIP PyTorch, Hygon DCU | HIP events |
| AscendC | `ascendc` | CANN/bisheng, compatible PyTorch + torch_npu, Ascend NPU | NPU events |

Nsight Compute provides CUDA counter profiling. Hygon and Ascend counter profiling
is not implemented. Init, Decompose, `init_memory`, and Apply with a supplied card
use the backend evaluator independently of counter profiling.

```bash
uv venv --system-site-packages
uv sync
codex login
```

Codex and evaluation workers run locally with host filesystem and network access.
For Hygon, point `HIP_HOME` or `ROCM_HOME` at the installed DTK when needed.
Ascend native builds require `ASCEND_HOME_PATH` and an explicit `ASCEND_ARCH`
(`dav-...`) supported by the installed compiler and actual SoC.
Init detects the installed accelerator. Set `KLINEAGE_BACKEND=cuda`, `hygon`, or
`ascend` to select it explicitly; the selected runtime must be available.

## Workflow

```python
from pathlib import Path
from klineage.action import workflow

candidate = workflow(
    problem="problems/definitions/gemm.json",
    repo="https://github.com/NVIDIA/cutlass.git",
    expert_kernel="include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp",
    workdir=Path("agent-workspace/experiment"),
    max_decompose_step=15,
    max_apply_step=15,
    enable_verifier=True,
)
```

Use a fresh workdir and hardware supporting the selected expert. The SM90 example
requires Hopper; set `TORCH_CUDA_ARCH_LIST=9.0a`
when its instantiated instructions require the architecture-specific target.
Init resolves a header to a concrete expert instance and records the host call.

```text
experiment/
  init/kernel.json
  decompose/0/{kernel.json,SKILL.md}
  decompose/1/{kernel.json,SKILL.md}
  apply/0/{kernel.json,SKILL.md}
  apply/0/evaluations/ncu-<id>/
  apply/1/{kernel.json,SKILL.md}
  ...
```

Each Decompose invocation removes one optimization and emits a standalone SkillCard
in `SKILL.md` alongside the simplified `kernel.json`. Its prompt receives `input_kernel`.
The final response records remaining mechanisms and a
revisable step estimate. An already naive input is preserved with no new card.
The loop stops on that unchanged result without a card, or its run limit. Count
completed removals by cards; retries and terminal checks add none. A limit does not
prove naive status.

Each Apply uses agent Python functions to retrieve eligible cards and profile the
current kernel. It checks prerequisites against source, ranks measured bottlenecks,
and applies top-1, excluding prior selections. An unchanged result without SKILL.md
or the apply limit ends the loop. The result is the latest kernel; zero applies
returns the final Decompose kernel.

Workflow enables verification before each handoff by default. Exhausted stage
failures stop the workflow. Even with verification disabled, Decompose handoffs
reject missing kernels, changed problems, changed kernels without cards, and
unchanged kernels with cards. Naive status still requires a source audit.
Memory selection with eligible candidates currently requires CUDA counter capture.
See [workflow.md](src/klineage/message/workflow.md) for every handoff.

To build a skill memory, use `init_memory`:

```python
from pathlib import Path
from klineage.action import init_memory

skill_paths = init_memory(
    problem="problems/definitions/gemm.json",
    repo="https://github.com/NVIDIA/cutlass.git",
    expert_kernel="include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong.hpp",
    max_decompose_step=15,
    enable_verifier=True,
    workdir=Path("agent-workspace/extract"),
    timeout=3600,
    max_retries=3,
    memory_dir=Path("memory"),
)
```

It runs Init and Decompose on CUDA, Hygon HIP, or AscendC with the same stop rules.
Each successful step saves its card to `memory_dir/<unique-skill-directory>/SKILL.md`.
Existing cards remain intact, including repeated skill IDs. The returned tuple contains this run's paths
in removal order, suitable for Apply's memory input. Later failures preserve already saved cards.
External verification defaults to disabled; the example enables it explicitly.
Init and Decompose still perform their own checks.

## Individual actions

`Action(prompt, workdir, enable_verifier=True, timeout=3600, max_retries=3)` is the
base class. `run()` accepts no arguments and returns no payload. Domain constructors
prepare prompts; Codex writes the artifacts. Lowercase functions wrap `run()`.

```python
from pathlib import Path
from klineage.action import Apply

step = Path("agent-workspace/experiment/decompose/0")
action = Apply(
    step, step / "SKILL.md",
    workdir=Path("agent-workspace/apply"),
    enable_verifier=True,
)
action.run()
# Read agent-workspace/apply/kernel.json.
```

Use a step containing SKILL.md. Omit the skill and pass `memory=Path("memory")`
or a sequence of cards/paths to let Apply select an optimization. Its optional
`exclude_skills` contains previously applied IDs.

The function catalog in workdir/AGENTS.md documents profiling, retrieval,
evaluation, repository staging, backend selection, and Kernel/SkillCard operations.
`profile` captures counters; `retrieve` filters scope and exclusions. The agent
still checks source prerequisites and ranks candidates.

`@agent_function` from `klineage.agent_api` registers the original function's
signature and docstring without wrapping it. For class methods, place it below
`@classmethod`. These remain ordinary Python calls. CodexRunner refreshes the
catalog before each call, preserving other instructions and updating an active
nonempty AGENTS.override.md as well, following
[Codex instruction discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

Verification uses a separate Verify action returning exactly `true` or `false`;
it does not verify itself. Each attempt keeps a unique Codex trace. Failed runs
retry generation and verification up to `max_retries` times after the first
attempt, with failure feedback and artifact repair in the same workdir.
Interrupts propagate immediately. Disabling verification runs generation once.

## Problems and artifacts

The [five paper workloads](problems/README.md) use FlashInfer Trace:
`definitions/*.json`, matching `workloads/*.jsonl`, and referenced
safetensors. Each definition currently has one workload. Python problems defining
`make_inputs()` and `torch_ref()` remain supported.
Portable Python inputs may use `make_inputs(seed, device)`; the harness passes
supported parameters and verifies that tensors use the selected backend device.

`ProblemSpec` has exactly `name`, `definition`, `workload`, `language`, `platform`.
The ordered definition inputs/outputs and workload axis bindings determine the
ABI. `ABIValue` has `name`, `dtype`, `shape`, `description`.

`kernel.json` contains exactly name, problem, source_files, and
validation. The embedded definition supplies the reference; workload descriptors
locate input data. Keep those input files accessible.

The registered `Kernel.from_sources` builds a callable instance.
Use `kernel(*inputs)` or pass it to `CallableKernelEvaluator(runtime, timer=timer)`.
Build settings come from config.toml; compiled handles stay in memory.
`load_kernel()` restores sources without compiling. Call `build()` before direct
execution; the subprocess harness builds restored kernels in its worker.

`klineage.backend` resolves language/platform, compiler, device, raw adapter ABI,
and runtime skill. A candidate and its timing reference must use the same backend.
Python bundles retain their language and resolve the accelerator from platform.
PyTorch HIP uses the `cuda` device namespace; Ascend uses `npu`.
Ascend tensor bindings support ND, NCHW, NHWC, and NCDHW base formats. Convert
blocked, reordered, or unknown formats to ND before defining the workload.

The harness keeps builds and workload artifacts in `artifacts.py`, correctness
and process entry points in `eval.py`, and backend measurements in `timing.py`.
`klineage.harness.evaluate` remains the action-facing evaluator.

The native skills [cuda](src/klineage/skills/cuda/SKILL.md),
[hip](src/klineage/skills/hip/SKILL.md), and [ascendc](src/klineage/skills/ascendc/SKILL.md)
define source bundles and bindings. All use `config.toml` and `solution/`, with
`build.language` and `entry_point="source::symbol"` selecting the implementation.
The [bench skill](src/klineage/skills/bench/SKILL.md) documents harness measurements.
CodexRunner links these skills into each workdir's
`.agents/skills/`; treat these packaged instructions as read-only. They are separate
from the optimization SkillCards extracted by Decompose.
Python bundles retain the callable configured by config.toml under solution/.

## Verification and measurements

Init performs its own correctness, performance, and mechanism checks. External
verifiers independently compile, compare against the problem reference, measure
with the selected backend timer, and audit mechanisms. Changed source clears
prior validation. Top-K checks selected values and indices while
allowing arbitrary output order and tied indices.

Performance fidelity requires matching backend timing policies and
`reference_ms / candidate_ms >= 0.99` overall and in every trial. Init repeats the
candidate/expert comparison independently. Decompose checks predecessor correctness
and applies the performance gate to forward reconstruction against the input kernel.
Verification results apply to the measured workload, expert instance and hardware.

CUDA timing uses CUPTI's span from the first correlated device activity's start to
the last activity's end, including kernels, copies, memsets, and internal gaps.
HIP and NPU events measure the current stream interval, including dispatch gaps;
their timing boundaries differ. Cold L2 defaults to enabled on CUDA and disabled
on HIP/Ascend, where an explicit cold-L2 request is rejected.

NCU profiles diagnose bottlenecks. Their raw CSV, report, parsed rows and interpretation
remain accessible; NCU does not replace CUPTI performance measurements.

SKILL.md frontmatter contains skill_id, intent, preconditions, and scope.
Intent briefly describes the removed optimization. The independent Markdown body
contains Overview, Precondition, Scope, and Code Change Snippet sections. SkillCard.to_dict()
also carries body. The [GEMM staging example](src/klineage/prompts/decompose_gemm.j4)
is included for CUDA or unresolved directory inputs and explicitly scoped to CUDA.
Other backends retain their own memory hierarchy and execution-unit requirements.
Preconditions describe types, layout and parallel ownership, storage placement,
pipeline structure, and required hardware features before applying the optimization.
State necessary dependencies with reasons; keep instance dimensions, launch sizes,
stage counts, and unrelated retained mechanisms in Overview / Example configuration.
Retrieval and verification check them against actual source_files.
Measurements remain under evaluations/, outside the card.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
KLINEAGE_CUDA_TESTS=1 PYTHONPATH=src python -m unittest discover -s tests -q
```

The CUDA suite executes the cuda skill's pybind11 example and caller-stream checks.
