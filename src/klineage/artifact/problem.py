"""FlashInfer Trace artifacts and Python problem loading."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import uuid
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from klineage.backend import get_backend
from klineage.constants import MODULE_HASH_LENGTH

if TYPE_CHECKING:
    import torch


#: Trace dataset directory containing problem definitions.
DEFINITIONS = "definitions"
#: Trace dataset directory containing workload records.
WORKLOADS = "workloads"
#: Safetensors metadata key for captured tensor strides.
STRIDES_METADATA = "klineage.strides"


class InputSource(StrEnum):
    RANDOM = "random"
    SAFETENSORS = "safetensors"
    SCALAR = "scalar"


def load_trace(path: Path) -> ModuleType:
    root = next(
        (parent.parent for parent in path.parents if parent.name == DEFINITIONS), None
    )
    if root is None:
        raise ValueError("Trace definition must be under a definitions directory")
    definition = json.loads(path.read_text())
    relative = path.relative_to(root / DEFINITIONS).with_suffix(".jsonl")
    records = [
        json.loads(line)
        for line in (root / WORKLOADS / relative).read_text().splitlines()
        if line.strip()
    ]
    if len(records) != 1:
        raise ValueError("A problem definition must have exactly one workload")
    record = records[0]
    if record["definition"] != definition["name"]:
        raise ValueError("Workload references a different definition")
    workload = record["workload"]
    if set(workload["inputs"]) != set(definition["inputs"]):
        raise ValueError("Workload inputs must match definition inputs")

    # Artifact consumers run in other directories; retain absolute input paths.
    for descriptor in workload["inputs"].values():
        if descriptor["type"] == InputSource.SAFETENSORS:
            descriptor["path"] = str((root / descriptor["path"]).resolve(strict=True))

    return trace_module(definition, workload)


def trace_module(
    definition: Mapping[str, Any], workload: Mapping[str, Any]
) -> ModuleType:
    """Execute the reference embedded in a problem contract."""
    module = ModuleType(definition["name"])
    exec(  # noqa: S102 - Trace references are executable Python.
        compile(definition["reference"], f"<{definition['name']}>", "exec"),
        module.__dict__,
    )
    if not callable(getattr(module, "run", None)):
        raise TypeError("Trace reference must define callable run()")
    module.__dict__.update(
        PROBLEM_NAME=definition["name"],
        OPERATOR=definition["op_type"],
        definition=definition,
        workload=workload,
        torch_ref=module.run,
    )
    return module


def trace_inputs(
    definition: Mapping[str, Any],
    workload: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    from klineage.contract import axis_size

    # Stage NPU inputs on CPU; generator and safetensors device support vary by release.
    target = str(device)
    on_npu = target.split(":", 1)[0] == "npu"
    if on_npu:
        get_backend("ascendc").torch()
    source_device = "cpu" if on_npu else target
    generator = torch.Generator(device=source_device).manual_seed(seed)
    inputs = {}
    for name, spec in definition["inputs"].items():
        if spec["shape"] is None:
            raise ValueError("The evaluator requires tensor inputs")
        shape = tuple(
            axis_size(axis, definition["axes"], workload["axes"])
            for axis in spec["shape"]
        )
        dtype = getattr(torch, spec["dtype"])
        descriptor = workload["inputs"][name]
        source = InputSource(descriptor["type"])
        if source == InputSource.RANDOM:
            if not dtype.is_floating_point:
                raise ValueError(
                    "Random integer inputs require an explicit safetensors source"
                )
            value = torch.randn(
                shape, dtype=dtype, device=source_device, generator=generator
            )
            if on_npu:
                value = value.to(device)
        elif source == InputSource.SAFETENSORS:
            with safe_open(
                descriptor["path"], framework="pt", device=source_device
            ) as tensors:
                value = tensors.get_tensor(descriptor["tensor_key"])
                if on_npu:
                    value = value.to(device)
                layouts = json.loads(
                    (tensors.metadata() or {}).get(STRIDES_METADATA, "{}")
                )
                stride = layouts.get(descriptor["tensor_key"])
                if stride is not None and tuple(stride) != value.stride():
                    # Safetensors stores contiguous values; restore captured layout.
                    restored = torch.empty_strided(
                        value.shape, stride, dtype=value.dtype, device=value.device
                    )
                    restored.copy_(value)
                    value = restored
        else:
            raise ValueError("The evaluator requires tensor inputs")
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Workload input {name!r} does not match its shape/dtype")
        inputs[name] = value
    return inputs


def trace_definition(
    name: str,
    operator: str,
    source: str,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> dict[str, Any]:
    axes, tensors = {}, {}
    for role, values in (("inputs", inputs), ("outputs", outputs)):
        tensors[role] = {}
        for tensor_name, value in values.items():
            shape = []
            for index, size in enumerate(value.shape):
                axis = f"{role}_{tensor_name}_{index}"
                axes[axis] = {"type": "const", "value": size}
                shape.append(axis)
            tensors[role][tensor_name] = {
                "shape": shape,
                "dtype": str(value.dtype).removeprefix("torch."),
                "description": f"Tensor with element strides {list(value.stride())}.",
            }

    # Trace references expose run; existing problem modules expose torch_ref.
    args = ", ".join(inputs)
    reference = f"{source.rstrip()}\n\ndef run({args}):\n    return torch_ref({args})\n"
    compile(reference, "<trace reference>", "exec")
    return {
        "name": name,
        "op_type": operator,
        "axes": axes,
        **tensors,
        "reference": reference,
    }


def trace_workload(inputs: Mapping[str, Any], path: Path) -> dict[str, Any]:
    from safetensors.torch import save_file

    # Preserve actual inputs; make_inputs may encode a nonrandom distribution.
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in inputs.items()
        },
        str(path),
        metadata={
            STRIDES_METADATA: json.dumps(
                {name: list(value.stride()) for name, value in inputs.items()}
            )
        },
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "uuid": str(uuid.uuid5(uuid.NAMESPACE_OID, digest)),
        "axes": {},
        "inputs": {
            name: {"type": "safetensors", "path": str(path), "tensor_key": name}
            for name in inputs
        },
    }


def load_problem(value: str | Path) -> tuple[ModuleType, Path]:
    path = Path(value).expanduser().absolute().resolve(strict=True)
    if path.is_file() and path.suffix == ".json":
        return load_trace(path), path
    if not path.is_file() or path.suffix != ".py":
        raise ValueError(
            "problem_path must identify a Trace definition JSON or Python file"
        )
    name = f"_klineage_problem_{hashlib.sha256(str(path).encode()).hexdigest()[:MODULE_HASH_LENGTH]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("make_inputs", "torch_ref"):
        if not callable(getattr(module, symbol, None)):
            raise TypeError(f"problem must define callable {symbol}()")
    return module, path
