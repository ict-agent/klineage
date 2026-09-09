# Action messages

Domain actions render task parameters into Jinja prompts and write into their `workdir`.
Action and Verify also accept plain prompts.
The workflow passes that directory's absolute path to the next action.
Consume artifacts only after `run()` succeeds; file existence alone is insufficient.
With verification enabled, success also requires `Verify` to return `true`.

| Contract | Input | Downstream artifact |
| --- | --- | --- |
| [Action](action.md) | Prompt and workdir | Action-specific files |
| [Init](init.md) | Problem definition and expert repository | `kernel.json` |
| [Decompose](decompose.md) | Kernel directory | `kernel.json`, `SKILL.md` for a removal |
| [Profile](profile.md) | Kernel directory | `kernel.json` and NCU capture files |
| [Retrieve](retrieve.md) | Profiled kernel and SkillCards | Selected `SKILL.md`, or no selection |
| [Apply](apply.md) | Kernel and one SkillCard | `kernel.json` |
| [CodeGen](code_gen.md) | Inline Kernel and one SkillCard | `kernel.json` |
| [Verify](verify.md) | Producer prompt, response and artifacts | Boolean; permitted evidence updates |
| [Workflow](workflow.md) | Problem and expert repository | Last profile directory |

Paths below are relative to the action's `workdir`, unless marked absolute.
Treat upstream directories as read-only. Preserve referenced directories and input
files: a handoff may retain absolute paths into earlier stages or the dataset.

## Shared files

```text
<workdir>/
  kernel.json                 # Kernel state, when this action produces one
  submission/                 # Generated source bundle, when code changes
    config.toml
    solution/                 # CUDA, HIP, AscendC, or Python sources
  evaluations/                # Inspection, validation or profiling evidence
  build/                      # Local compilation cache, when needed
  .klineage/codex-runs/<ActionClass>-<id>/
    prompt.txt                # Exact rendered prompt
    trace.jsonl               # Codex event stream
    final_message.txt         # Final model response
    stderr.log                # Runner diagnostics
```

Artifact filenames are fixed; optional directories appear only when used.
Kernel source_files contains the complete source bundle. Generated bundles use
submission/. The subprocess harness rebuilds from source_files, so unchanged
kernels need no copied submission directory.
Builds reuse `build/source-<sha256>/` for identical source maps; modified snapshots
are rejected. Measurements retain separate records for each evaluation.
Final responses remain in runner records; directory consumers read the named artifacts.
No separate `message.json`, `candidate.json`, or `report.md` is required.

## `kernel.json`

Write with `save_kernel(kernel, workdir)`; read with `load_kernel(workdir)` from
`klineage.harness.artifacts`. The file contains `Kernel.to_dict()`:

| Field | Format and meaning |
| --- | --- |
| `name` | Kernel name |
| `problem` | Exactly `name`, `definition`, `workload`, `language`, `platform` |
| `source_files` | Every relative bundle path mapped to its exact UTF-8 text |
| `validation` | `ValidationResult.to_dict()` or `null` |

`problem.definition` is a FlashInfer Trace Definition, including ordered
`inputs`/`outputs` and executable `reference.run`. `problem.workload` is the inner
Workload object (`uuid`, `axes`, `inputs`), without its surrounding Trace record.
Preserve order, dtypes, axis bindings and input descriptors. File-backed workload
inputs carry absolute paths, resolved during inspection.
`Kernel.fingerprint` is computed; it is not a serialized Kernel field.

Validation fields are `compile_passed`, `correctness_passed`, `profile_passed`,
`latency_ms`, and `reference_latency_ms`. Latencies are measured medians in ms:
`latency_ms` is the candidate's; `reference_latency_ms` is the supplied reference's
in the same evaluation, or `null` when unmeasured. Both must be finite and positive
when present. Successful timing requires candidate latency. `accepted` is
computed from the three gates. Changed code clears
validation. Paired gate samples, ratios, and rejection reasons are recorded in
eval-0001-stderr.log under evaluations/evaluate-*, outside kernel.json.
Harness failures appear in process stderr. NCU diagnostics remain in
the Profile action's evaluations/ncu-* directory and are not Kernel fields.

## Runtime skills

CodexRunner configures PYTHONPATH for the installed package and exposes message
contracts at `.klineage/message/`. It exposes packaged instructions at
`.agents/skills/{cuda,hip,ascendc,bench}/` in each workdir.
Treat these links as read-only.
The native skills [cuda](../skills/cuda/SKILL.md), [hip](../skills/hip/SKILL.md), and
[ascendc](../skills/ascendc/SKILL.md) own their bundle formats and bindings;
the [bench skill](../skills/bench/SKILL.md) owns harness usage and evidence paths.
These runtime skills are separate from optimization SkillCards emitted as
SKILL.md. Optimization cards use YAML frontmatter and Markdown instructions;
their in-memory and inline JSON representations retain four metadata fields and body.

## Runtime Kernel

`Kernel.from_sources(source_files, problem, build_root=Path("build"))` builds a
callable instance. `kernel(*inputs)` runs it; the callable evaluator accepts that
instance directly. Build language, entry_point and output_style come from config.toml.
These derived attributes and the compiled function are not serialized.
`load_kernel` / `Kernel.from_dict` restore source artifacts; call `build()` before
direct execution. The subprocess harness builds restored kernels in its worker.
