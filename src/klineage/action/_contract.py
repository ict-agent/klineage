"""Private problem inspection adapter shared by init and apply."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from klineage._utils import nonempty
from klineage.contract import RAW_CUDA_ABI, KernelABI, ProblemSpec
from klineage.kernel import TargetContext


def _contract(
    problem_path: Path,
    description: Mapping[str, Any],
) -> tuple[ProblemSpec, KernelABI, TargetContext]:
    try:
        name = description["problem_name"]
        raw_abi = description["abi"]
        platform = description["platform"]
    except KeyError as exc:
        raise ValueError(f"problem inspection omitted {exc.args[0]!r}") from exc
    if not isinstance(name, str) or not isinstance(platform, str):
        raise TypeError("problem inspection name and platform must be strings")
    if not isinstance(raw_abi, Mapping):
        raise TypeError("problem inspection ABI must be an object")

    source = problem_path.read_text(encoding="utf-8")
    relative = f"{problem_path.parent.name}/{problem_path.name}"
    statement = f"""Authoritative input problem ({relative}):

```python
{source}
```

Implement exactly those semantics with this standalone raw CUDA ABI:
    {RAW_CUDA_ABI}
`inputs[i]` and `outputs[i]` follow kernel_abi declaration order. Shapes and
dtypes are fixed by kernel_abi. Launch on the supplied stream, do not
synchronize it, and return the CUDA launch status. Do not call PyTorch, cuBLAS,
another framework/library implementation, a subprocess, or a precomputed
result.
""".strip()
    operator = nonempty(description.get("operator", name), "problem operator")
    problem = ProblemSpec(name=name, statement=statement, parameters={"operator": operator})
    abi = KernelABI.from_dict(raw_abi)
    context = TargetContext(
        case=operator,
        language="cuda",
        platform=platform,
    )
    return problem, abi, context
