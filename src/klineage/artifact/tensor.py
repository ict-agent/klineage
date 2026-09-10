"""Adapt bundle calls to the problem's tensor ABI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from klineage.backend import Backend, get_backend
from klineage.contract import ABIValue, OutputStyle, ProblemSpec, ValueRole

if TYPE_CHECKING:
    import torch

#: Accepted ABI dtype spellings mapped to PyTorch names.
DTYPE_ALIASES = {
    "bool": "bool",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "float32": "float32",
    "float": "float32",
    "float64": "float64",
    "double": "float64",
    "int8": "int8",
    "uint8": "uint8",
    "int16": "int16",
    "short": "int16",
    "int32": "int32",
    "int": "int32",
    "int64": "int64",
    "long": "int64",
}


class BundleCallable:
    def __init__(
        self,
        problem: ProblemSpec,
        function: Any,
        style: OutputStyle,
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
    ):
        self.inputs = tensor_values(problem, ValueRole.INPUTS)
        self.outputs = tensor_values(problem, ValueRole.OUTPUTS)
        self.backend = get_backend(problem.language, problem.platform)
        self.function, self.style = function, style
        self.strides = strides or {}

    def __call__(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.inputs):
            raise ValueError("bundle input count does not match its ABI")
        tensors = tuple(
            check_tensor(
                value,
                spec,
                "input",
                self.strides.get(("input", spec.name)),
                backend=self.backend,
            )
            for value, spec in zip(inputs, self.inputs, strict=True)
        )
        device = tensors[0].device if tensors else self.backend.device()
        if any(value.device != device for value in tensors):
            raise ValueError("bundle inputs must share a device")

        # The evaluator supplies trailing destinations; return-style entries own allocation.
        with self.backend.runtime().device(device):
            if self.style is OutputStyle.DESTINATION:
                outputs = tuple(
                    allocate(spec, device, self.strides.get(("output", spec.name)))
                    for spec in self.outputs
                )
                self.function(*tensors, *outputs)
            else:
                result = self.function(*tensors)
                outputs = (result,) if len(self.outputs) == 1 else result
                if not self.outputs and result is None:
                    outputs = ()
                if (
                    not isinstance(outputs, Sequence)
                    or isinstance(outputs, (str, bytes))
                    or len(outputs) != len(self.outputs)
                ):
                    raise ValueError("bundle output count does not match its ABI")

            for value, spec in zip(outputs, self.outputs, strict=True):
                check_tensor(
                    value,
                    spec,
                    "output",
                    self.strides.get(("output", spec.name)),
                    backend=self.backend,
                )
                if value.device != device:
                    raise ValueError("bundle outputs must use the input device")
        if not outputs:
            return None
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def tensor_values(problem: ProblemSpec, role: ValueRole) -> tuple[ABIValue, ...]:
    values = problem.values(role)
    if any(problem.definition[role][value.name]["shape"] is None for value in values):
        raise TypeError("tensor entries do not support scalar arguments or outputs")
    return values


def check_tensor(
    value: Any,
    spec: ABIValue,
    role: str,
    stride: tuple[int, ...] | None = None,
    *,
    backend: Backend,
) -> torch.Tensor:
    tensor = require_tensor(value, f"ABI {role} {spec.name!r}", backend)
    if tensor.dtype != tensor_dtype(spec.dtype):
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong dtype")
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong shape")
    if stride is not None and tuple(tensor.stride()) != tuple(stride):
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong stride")
    return tensor


def allocate(
    spec: ABIValue,
    device: torch.device,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    import torch

    shape = fixed_shape(spec)
    if stride is None:
        return torch.empty(shape, dtype=tensor_dtype(spec.dtype), device=device)
    return torch.empty_strided(
        shape,
        tuple(int(value) for value in stride),
        dtype=tensor_dtype(spec.dtype),
        device=device,
    )


def fixed_shape(spec: ABIValue) -> tuple[int, ...]:
    if any(isinstance(value, str) for value in spec.shape):
        raise ValueError(f"output {spec.name!r} must have a fixed shape")
    return tuple(int(value) for value in spec.shape)


def tensor_dtype(value: str | None) -> torch.dtype:
    import torch

    if value is None:
        raise ValueError("ABI values require a dtype")
    name = value.lower().removeprefix("torch.")
    try:
        return getattr(torch, DTYPE_ALIASES[name])
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {value!r}") from exc


def require_tensor(value: Any, label: str, backend: Backend) -> torch.Tensor:
    torch = backend.torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != backend.device_type:
        raise ValueError(f"{label} must be on {backend.device_type}")
    backend.validate_tensor(value)
    return value
