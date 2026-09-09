# Profile

[Shared message contract](README.md).

Hardware-counter capture currently supports CUDA through NCU. Hygon and Ascend
return an unsupported-capability error; event latency is not a counter profile.

## Input

| Template parameter | Format |
| --- | --- |
| `kernel` | Absolute directory containing `kernel.json`, or inline Kernel |
| `options` | `set`, `sections`, `kernel_filter`, `timeout_seconds` |

Default options:

```json
{"set": "detailed", "sections": [], "kernel_filter": null, "timeout_seconds": 180}
```

## Output

```text
<workdir>/
  kernel.json
  evaluations/ncu-<id>/
    metrics.csv                # Raw NCU metric rows
    profile.ncu-rep            # NCU report
    profile-result.json        # Parsed capture and tool/device metadata
    profile-request.json       # Kernel and problem used for this capture
    profile-process.json       # NCU command, options, and process outcome
  build/                       # Compilation cache, when needed
```

profile-result.json contains tool, tool_version, device, metrics, report_path,
raw_path, and collected_at. Metric rows contain kernel, launch_id, section, metric,
unit, and value. Read actual paths from raw_path and report_path.

Write the unchanged input to kernel.json. The request's serialized Kernel binds
this capture to its source_files and problem; compare fingerprints before reuse.
Profile does not produce a new submission directory.

Final response:

```json
{"summary": "Interpretation of the captured measurements", "metric_indices": [0]}
```

Indices are zero-based positions in the capture result's `metrics`; cite actual rows.
Verification checks the artifacts and interpretation without modifying them.

## Handoff

Pass `workdir` to Retrieve. Retain the capture files.
