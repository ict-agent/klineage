# Action Primitive

Actions transform Artifacts through an agent session. **Init** adapts an expert
implementation, **Decompose** extracts an optimization into a reusable skill,
and **Apply** generates or improves a kernel using those skills.

<figure class="diagram" tabindex="0" aria-label="Init and repeated Decompose rounds extract skill memory for Apply to reuse on a target kernel.">
  <img src="../assets/diagrams/actions.svg"
       alt="Init and repeated Decompose rounds extract skill memory for Apply to reuse on a target kernel."
       width="800" height="400" loading="lazy">
</figure>

Each call runs one round and returns an Artifact. A controller can pass that
result to another action, repeat a round or stop. For generation from a problem
directory, [klineage-generation](../guides/cli.md#generate-a-kernel) prepares the
input and runs Apply for you.

## Choose an action

| Action | Purpose | Result |
| --- | --- | --- |
| `Init` | Adapt an expert implementation to the problem and its workloads | Kernel |
| `Decompose` | Remove one optimization and explain how to restore it | Kernel and, when changed, a SkillCard |
| `Apply` | Generate or improve an implementation, optionally using memory | Kernel |

Init gives the agent read-only access to the expert repository. Decompose turns
implementation knowledge into a recipe. Apply can use those recipes, optimize
without memory, or start from an empty submission.

Every kernel keeps the complete problem and workload set. The evaluation service
checks every workload. Apply asks for an improvement on at least one workload
while preserving the others' performance gates; if it finds none, it returns the
unchanged implementation. Decompose can return a slower kernel after removing
an optimization.

## Pass results between actions

Set up model access and an evaluation service using
[Getting started](../getting-started.md). The service must contain the matching
problem. Replace the expert repository and source paths below with your own:

```python
from pathlib import Path

from klineage.action import Apply, Decompose, Init
from klineage.artifact import Kernel
from klineage.memory import SkillCard, save_skill
from klineage.server import Client

client = Client(problem_id="gemm")  # Reads KLINEAGE_SERVER.
options = {"evaluation": client, "timeout": 600, "max_retries": 0}

initialized = Init(
    client.problem(),
    repo=Path("expert-repo"),
    expert_kernel=Path("src/kernel.cu"),
    workdir=Path("runs/init"),
    **options,
).run()

decomposed = Decompose(
    initialized, workdir=Path("runs/decompose"), **options,
).run()

memory = Path("memory")
memory.mkdir(exist_ok=True)
if any(node.name == "skillcard" for node in decomposed.tree.children):
    card = decomposed.part("skillcard", SkillCard)
    save_skill(card, memory / "extracted" / "SKILL.md")

applied = Apply(
    decomposed,
    memory=memory,
    workdir=Path("runs/apply"),
    **options,
).run()
kernel = applied.part("kernel", Kernel)
```

Give each round a fresh work directory. Apply and Decompose accept a Kernel, a
previous action result, serialized artifact JSON, or a directory containing a
Kernel or an action's numbered snapshots.

## Read an action result

The returned Artifact contains a `kernel` part, an optional `skillcard`, and any
runner records. Use `.part("kernel", Kernel)` to obtain the backend-specific
Kernel. Its metadata holds validation, workload timings and diagnostics.

During a round, evaluation saves complete versions under `kernels/0`,
`kernels/1`, and so on. The action returns the last version after checking it
against the service record. Every workload must pass. A failed latest version
is rejected; to select an earlier accepted job, fetch it as a new final version:

```python
client.result(best_job_id, work=Path("runs/apply"))
```

`best_job_id` must come from a previous accepted evaluation. The scratch
`kernel/` directory can differ from the saved result. See
[Evaluation guide](../guides/evaluation.md#submit-and-recover-a-result) for the
result files and recovery workflow.

The service checks correctness and performance. Review the source change and
recipe to assess whether Decompose removed the intended mechanism or Apply
achieved the requested improvement.

## Stop or retry

An unchanged result keeps the original implementation and produces no skill
card. For Decompose, this signals that the agent found no further optimization
to remove. `same_implementation(before, after)` compares the problem, sources
and compile flags while ignoring measurements.

`timeout` limits each agent attempt. Timed prompts include a UTC deadline so the
agent can reserve time for validation and its final response. `max_retries` permits additional attempts
after failures, resuming a recorded session when available. If all attempts fail,
the action raises an error. Controllers that need a total time budget must track
it across rounds and evaluation requests.

## Reuse skill memory

A `SkillCard` stores a Markdown recipe in `SKILL.md`, with four YAML fields:
`skill_id`, `intent`, `preconditions` and `scope`. Keep cards in a memory directory
and use `save_skill` and `load_skill` to write or read them.

Apply receives matching cards read-only. Retrieval filters by operator, language
and platform, preserving memory order; `*` matches any value. The agent checks
each recipe's prerequisites against the current kernel before applying it.

See the [Action API](../reference/actions.md) for action options and the
[Skill memory API](../reference/memory.md) for card formats and retrieval.
