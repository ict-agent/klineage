# Extension guide

Add an Artifact class when you need a new file format. Add a backend when you
need to build and measure kernels on another device. Both use the same annotated
Node trees described in [Artifact](../concepts/artifacts.md).

## Define an artifact class

Give the class a type identifier and a Node layout. Add content checks in
`validate()` when file structure alone is not enough.

This example defines a latency report, loads it from a directory and restores
it from JSON. It runs with the base KLineage package and needs no device:

```python
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.artifact import Artifact, Node, load_artifact, save_artifact
from klineage.serialization import raw_text


class LatencyReport(Artifact):
    """A report containing one positive latency."""

    artifact_kind = "report.latency"
    schema = Node.directory(".", (
        Node.file(
            "result.json",
            description="A finite positive latency in milliseconds.",
            one_shot='{"latency_ms": 1.0}',
        ),
    ))

    def validate(self):
        super().validate()
        result = json.loads(raw_text(self.node("result.json")))
        if not isinstance(result, dict) or set(result) != {"latency_ms"}:
            raise ValueError("result.json: expected only latency_ms")
        latency = result["latency_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("result.json/latency_ms: expected a finite positive number")


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    report_dir = root / "report"
    report_dir.mkdir()
    (report_dir / "result.json").write_text(
        '{"latency_ms": 1.0}', encoding="utf-8"
    )

    report = load_artifact(report_dir, LatencyReport)
    data_path = save_artifact(report, root / "report-data.json")
    restored = load_artifact(data_path, LatencyReport)
    assert restored == report
```

The base `validate()` checks `schema` with `Node.check`. Classes with a layout
method, such as `Problem.problem_spec()`, call `check` on the returned tree instead.
Descriptions and examples explain a format; `validate()` enforces its contents.

Pass the concrete class when loading a directory or JSON so its checks run.
`Action(..., output_type=LatencyReport)` can collect the same format from an
agent workspace. No Artifact format registration is needed.

## Register a JSON handler

Artifact subclasses already share a serialization handler. For another Python
type, define how to serialize and restore it, then register that handler:

```python
from dataclasses import dataclass

from klineage.serialization import (
    SerializationHandler,
    from_json,
    raw_text,
    register_handler,
    to_json,
)


@dataclass(frozen=True)
class BuildLabel:
    name: str


class BuildLabelHandler(SerializationHandler):
    def default(self, value):
        return {"name": value.name}

    def restore(self, data, object_type):
        return object_type(**data)

    def raw_text(self, value):
        return value.name


register_handler(BuildLabel, BuildLabelHandler())
label = BuildLabel("cuda")
assert from_json(to_json(label), BuildLabel) == label
assert raw_text(label) == "cuda"
```

`default()` supplies JSON-compatible data; `restore()` reconstructs the requested
type and can check its contents. Optional `raw_text()` and `raw_bytes()` methods
provide text and byte representations through the same interface.

Register each type once per process. Subclasses use the nearest registered
handler in their inheritance chain. See the
[serialization API](../reference/artifacts.md#serialization) for the complete contract.

## Add a device implementation

A backend needs a Kernel class, device information and a timing implementation.
Keep them in this layout, adding other modules as needed:

```text
klineage/backend/example/
  __init__.py         # Public Kernel and Backend classes
  kernel.py           # File layout, builder and device information
  timing.py           # Latency measurement
  skills/SKILL.md      # Kernel authoring guidance for the agent
```

An external package can use the same organization. Keep imports usable without
device libraries; load the vendor runtime inside build, timing and information
calls.

### Define the Kernel

Subclass `Kernel` and choose a unique `artifact_kind`. Implement:

| Method | Responsibility |
| --- | --- |
| `kernel_spec()` | Return the complete Node layout, including metadata and submission files |
| `build_kernel()` | Build the source and return its host callable |

Keep the original problem directory reference in `metadata.json`. The source
entry owns its launch parameters and bindings. Report invalid configuration or
source layout with the failing file or field path.

Reuse `Kernel.load_kernel()` for directory loading and the existing handlers for
serialization. The builder receives a separate compiler-output directory,
`include_paths`, and optional `strides` and `workload` bindings. A workload selects
one case without replacing the Kernel's reference to the whole problem.
See the [Kernel API](../reference/artifacts.md#klineage.artifact.Kernel) for signatures.

The shared `BundleLoader` supports PyTorch tensor inputs and outputs. If you use
it, supply a compiler callback and supported entry suffixes. A different tensor
runtime also needs integration with the evaluation harness.

??? note "Shared builder behavior"

    Native builds without external include paths cache submitted sources, headers,
    build options and build root. External include paths use a fresh module
    identity, so changes to those headers are not hidden by the process-local cache.
    Compiler output is staged separately from submitted files.

    Workload bindings resolve concrete shapes. Direct calls can infer variable
    input axes from tensors; unresolved output-only axes raise an error.

### Implement timing and device information

The Backend connects the Kernel to its timing module and reports runtime
information. This skeleton assumes `ExampleKernel` and `timing.py` are implemented:

```python
from pathlib import Path

from klineage.backend import Backend
from . import timing


class ExampleBackend(Backend):
    kind = "example"
    language = "example"
    device_type = "example"
    platform_prefixes = ("example",)
    kernel_type = ExampleKernel
    skill_root = Path(__file__).with_name("skills")
    timing = timing
    bench = staticmethod(timing.bench)

    def info(self):
        raise NotImplementedError("Query the vendor runtime")
```

Use distinct identifiers and platform prefixes. `info()` must report `available`
for discovery and `platform` for problem loading. Include the architecture,
runtime versions and toolchain information needed by the builder.

The timing module provides:

| Export | Purpose |
| --- | --- |
| `NAME` | Measurement provider recorded in each result |
| `DEFAULT_POLICY` | Supported default `TimingPolicy` |
| `bench(fn, *, args=(), policy=None)` | Measure the loaded callable |
| `check_timing(details, policy)` | Check measurement evidence |

Return a `TimingResult` with `policy.trials` groups of `policy.repeat` finite,
positive millisecond samples and the median of trial medians. Store the policy
in `details["policy"]`, plus evidence such as the provider version or stream
scope. Reject unsupported measurement policies explicitly.

CUDA provides a working example in
[CUDA implementation](../concepts/backends.md#cuda-implementation). HIP and Ascend
use current-stream events and reject cold-L2 requests.

### Register the backend

Export the Kernel and Backend from the package, then register an instance in
startup code. An external package can provide `example_backend/registration.py`:

```python
from klineage.backend import register_backend
from .kernel import ExampleBackend

register_backend(ExampleBackend())
```

Registration applies to the current process. Arrange it wherever the backend is
used:

| Process | Setup |
| --- | --- |
| Controller | Import the registration module before loading kernels or actions |
| Evaluation service and workers | Pass `--plugin example_backend.registration` |
| Agent Python scripts | Include the registration import in backend skill examples |

Install the package in each environment. For a built-in backend, add its identifier
to `BackendKind` and its instance to `klineage.backend.registry.REGISTERED_BACKENDS`.
Built-in registration runs automatically.

The service uses the same Kernel and Backend interfaces; it needs no extra backend
methods. Configure its interpreter, devices and dependencies using the
[Evaluation guide](evaluation.md).

### Check the implementation

Start with imports, layout validation, JSON round trips and backend selection
without device access. On the target hardware, check compilation, numerical
outputs, current-stream behavior and paired timing through the service.

Cover several workload shapes, fixed inputs, external-header changes and
unsupported timing policies. CPU tests establish interface behavior; hardware
runs establish execution support. Existing examples are in
`tests/test_backend_ports.py`.
