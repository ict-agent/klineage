# Action messages

Actions exchange information through files. A producer writes artifacts into its
`workdir`; the caller passes that directory to the next action, which reads the
named files. Each action's Markdown contract defines its accepted inputs, output
directory layout, artifact meanings, and handoff rules.

| Contract | Accepted input | Output |
| --- | --- | --- |
| [Action](action.md) | Prompt and workdir | Runner records and action-specific artifacts |
| [Init](init.md) | Problem definition and expert repository | `kernel.json` and standalone source bundle |
| [Decompose](decompose.md) | Kernel directory | `kernel.json`; `SKILL.md` for removals |
| [Apply](apply.md) | Kernel and optional memory directory | `kernel.json` after an optimization round |
| [Verify](verify.md) | Producer's generation record and artifacts | `true`/`false`; permitted artifact updates |

[Workflow](workflow.md) describes CLI orchestration, stage directories, and stop rules.

Consume artifacts only after `run()` succeeds; file existence alone is insufficient.
With verification enabled, success also requires `Verify` to return `true`.

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
Measurements retain separate records for each evaluation. Final responses remain
in runner records; downstream actions consume the files named by their contracts.
No separate `message.json`, `candidate.json`, or `report.md` is required.

## `kernel.json`

Use `save_kernel` and `load_kernel` for persistence.
The file contains `Kernel.to_dict()`:

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
Compiled functions are not serialized. `load_kernel` restores sources without
compiling; call `build()` before direct execution. The harness builds restored
kernels in its worker.

Validation fields are `compile_passed`, `correctness_passed`, `profile_passed`,
`latency_ms`, and `reference_latency_ms`. Latencies are measured medians in ms:
`latency_ms` is the candidate's; `reference_latency_ms` is the supplied reference's
in the same evaluation, or `null` when unmeasured. Both must be finite and positive
when present. Successful timing requires candidate latency. `accepted` is
computed from the three gates. Changed code clears
validation. Paired gate samples, ratios, and rejection reasons are recorded in
eval-0001-stderr.log under evaluations/evaluate-*, outside kernel.json.
Harness failures appear in process stderr. Counter diagnostics remain under
evaluations/ncu-* in the calling agent's workdir and are not Kernel fields.

## Implementation references

CodexRunner exposes these contracts at `.klineage/message/` and packaged skills at
`.agents/skills/{cuda,hip,ascendc,bench}/`; keep those links read-only.
[CUDA](../skills/cuda/SKILL.md), [HIP](../skills/hip/SKILL.md), and
[AscendC](../skills/ascendc/SKILL.md) define native bundles and bindings.
[Bench](../skills/bench/SKILL.md) defines evaluation procedures and evidence.
Read workdir/AGENTS.md for registered Python functions and their import paths.
