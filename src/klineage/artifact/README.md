# Artifacts

| Module | Responsibility |
| --- | --- |
| [kernel.py](kernel.py) | Kernel state, validation results, and JSON persistence |
| [source.py](source.py) | Source-tree reading and immutable snapshots |
| [bundle.py](bundle.py) | Native builds, Python loading, and import isolation |
| [tensor.py](tensor.py) | Tensor ABI checks, output allocation, and call adaptation |
| [problem.py](problem.py) | Trace definitions, workloads, inputs, and reference loading |
| [repository.py](repository.py) | Repository staging |

`Kernel.build()` uses `bundle`, which uses `source`, `tensor`, and the backend.
The evaluation harness consumes these artifacts. Process execution lives in
`harness/process.py`; shared path and JSON helpers live in `utils.py`.
