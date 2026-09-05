"""Private CUDA evaluator process.

The worker accepts one JSON object on stdin. ``inspect`` derives a fixed
tensor ABI from a problem's ``make_inputs`` and ``torch_ref`` functions.
``evaluate`` builds or imports a submitted kernel, checks it against
``torch_ref``, and measures it with FlashInfer CUPTI.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch.utils.cpp_extension import load as load_torch_extension

from klineage.contract import ABIValue, EvaluatorInterface, KernelABI
from klineage.harness.callable_eval import (
    CallableKernelEvaluator,
    CallInputs,
    ProblemRuntime,
    PythonEntrypointLoader,
)
from klineage.harness.eval import ValidationResult
from klineage.harness.timing import FlashInferCuptiTimer, TimingPolicy
from klineage.kernel import Kernel

_BINDING_SOURCE = Path(__file__).with_name("_cuda_binding.cpp")
_DEFAULT_SEED = 0
_DEFAULT_TIMING = TimingPolicy()
_DTYPE_ALIASES = {
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
}


@dataclass(frozen=True, slots=True)
class _Config:
    problem_path: Path
    build_root: Path
    include_paths: tuple[Path, ...]
    timing: TimingPolicy
    seed: int


class _RawLoader:
    """Build raw CUDA files and dispatch directories to the public loader."""

    def __init__(self, config: _Config) -> None:
        self._config = config
        self._directory_loader = PythonEntrypointLoader()
        self._extensions: dict[str, Any] = {}

    def load(self, kernel: Kernel):
        artifact = kernel.artifact_path
        if artifact is None:
            raise ValueError("kernel.artifact_path is required")
        artifact = artifact.resolve(strict=True)
        if artifact.is_dir():
            return self._directory_loader.load(kernel)
        if artifact.suffix.lower() != ".cu" or not artifact.is_file():
            raise ValueError("raw kernel artifact_path must identify a .cu file")
        return self._load_cuda(kernel, artifact)

    def _load_cuda(self, kernel: Kernel, source: Path):
        abi = kernel.abi
        if abi is None:
            raise ValueError("kernel.abi is required")

        digest = hashlib.sha256()
        for path in (_BINDING_SOURCE, source):
            digest.update(path.read_bytes())
            digest.update(b"\0")
        for path in self._config.include_paths:
            digest.update(str(path).encode())
            digest.update(b"\0")
        name = f"klineage_cuda_{digest.hexdigest()[:20]}"
        extension = self._extensions.get(name)
        if extension is None:
            build_dir = self._config.build_root / name
            build_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            extension = load_torch_extension(
                name=name,
                sources=(str(_BINDING_SOURCE), str(source)),
                extra_include_paths=tuple(
                    str(path) for path in self._config.include_paths
                ),
                extra_cflags=("-O3", "-std=c++17"),
                extra_cuda_cflags=(
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ),
                with_cuda=True,
                build_directory=str(build_dir),
                verbose=False,
            )
            self._extensions[name] = extension

        launch = getattr(extension, "launch", None)
        if not callable(launch):
            raise TypeError("native extension did not expose launch")
        return _RawCallable(abi, launch)


class _RawCallable:
    def __init__(self, abi: KernelABI, launch: Any) -> None:
        self._abi = abi
        self._launch = launch

    def __call__(self, *inputs: torch.Tensor):
        if len(inputs) != len(self._abi.inputs):
            raise ValueError("raw CUDA input count does not match its ABI")
        if not inputs and not self._abi.outputs:
            raise ValueError("the raw CUDA ABI requires at least one tensor")

        tensors = tuple(
            _check_tensor(value, spec, "input")
            for value, spec in zip(inputs, self._abi.inputs, strict=True)
        )
        device = tensors[0].device if tensors else torch.device("cuda")
        outputs = tuple(_allocate(spec, device) for spec in self._abi.outputs)
        self._launch(list(tensors), list(outputs))
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def inspect_problem(
    problem_path: str | Path, *, seed: int = _DEFAULT_SEED
) -> dict[str, Any]:
    """Load one reference.py and return its inferred fixed CUDA contract."""

    module, path = _load_problem(problem_path)
    inputs = _make_inputs(module, seed)
    expected = _reference(module, inputs)
    abi = _infer_abi(inputs, expected)

    return {
        "problem_name": _problem_name(module, path),
        "abi": abi.to_dict(),
        "platform": _platform(),
    }


def evaluate_request(request: Mapping[str, Any]) -> ValidationResult:
    """Evaluate one serialized candidate request."""

    kernel_value = _mapping(request.get("kernel"), "kernel")
    reference_value = request.get("reference")
    if reference_value is not None:
        reference_value = _mapping(reference_value, "reference")
    config = _config(_mapping(request.get("config"), "config"))

    kernel = Kernel.from_dict(kernel_value)
    reference = Kernel.from_dict(reference_value) if reference_value else None
    module, _ = _load_problem(config.problem_path)
    problem_inputs = _make_inputs(module, config.seed)
    inferred = _infer_abi(problem_inputs, _reference(module, problem_inputs))
    _check_abi(kernel, inferred)
    if reference is not None:
        _check_abi(reference, inferred)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the evaluator worker")
    evaluator = CallableKernelEvaluator(
        _runtime(module, config.seed),
        loader=_RawLoader(config),
        timer=FlashInferCuptiTimer(config.timing),
    )
    return evaluator.evaluate(kernel, reference=reference)


def _runtime(module: ModuleType, seed: int) -> ProblemRuntime:
    def make_inputs() -> CallInputs:
        values = _make_inputs(module, seed)
        return CallInputs(args=tuple(values.values()))

    def reference(*args: torch.Tensor):
        return module.torch_ref(*args)

    def check_outputs(actual: Any, expected: Any) -> None:
        torch.testing.assert_close(actual, expected)

    return ProblemRuntime(
        make_inputs=make_inputs,
        reference=reference,
        check_outputs=check_outputs,
    )


def _load_problem(value: str | Path) -> tuple[ModuleType, Path]:
    path = Path(value).expanduser().absolute().resolve(strict=True)
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("problem_path must identify a Python file")
    name = f"_klineage_problem_{hashlib.sha256(str(path).encode()).hexdigest()[:20]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("make_inputs", "torch_ref"):
        if not callable(getattr(module, symbol, None)):
            raise TypeError(f"problem must define callable {symbol}()")
    return module, path


def _make_inputs(module: ModuleType, seed: int) -> Mapping[str, torch.Tensor]:
    parameters = inspect.signature(module.make_inputs).parameters
    raw = (
        module.make_inputs(seed=seed) if "seed" in parameters else module.make_inputs()
    )
    if not isinstance(raw, Mapping):
        raise TypeError("problem.make_inputs() must return a mapping")
    values = dict(raw)
    if not values:
        raise ValueError("problem.make_inputs() must return at least one tensor")
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("problem input names must be non-empty strings")
        _require_cuda_tensor(value, f"problem input {name!r}")
    return values


def _reference(module: ModuleType, inputs: Mapping[str, torch.Tensor]) -> Any:
    with torch.inference_mode():
        return module.torch_ref(*inputs.values())


def _infer_abi(inputs: Mapping[str, torch.Tensor], output: Any) -> KernelABI:
    outputs = _outputs(output)
    return KernelABI(
        inputs=tuple(_abi_value(name, value) for name, value in inputs.items()),
        outputs=tuple(
            _abi_value(name, value)
            for name, value in zip(_output_names(output), outputs, strict=True)
        ),
        interface=EvaluatorInterface(),
    )


def _outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("problem.torch_ref() must return a tensor or tensor sequence")
    outputs = tuple(value)
    if not outputs:
        raise ValueError("problem.torch_ref() must return at least one tensor")
    for index, output in enumerate(outputs):
        _require_cuda_tensor(output, f"problem output {index}")
    return outputs


def _output_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, torch.Tensor):
        return ("output",)
    fields = getattr(value, "_fields", None)
    if isinstance(fields, tuple) and all(isinstance(field, str) for field in fields):
        return fields
    return tuple(f"output_{index}" for index in range(len(value)))


def _abi_value(name: str, tensor: torch.Tensor) -> ABIValue:
    return ABIValue(
        name=name,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        constraints={
            "device": "cuda",
            "contiguous": tensor.is_contiguous(),
            "stride": list(tensor.stride()),
        },
    )


def _check_abi(kernel: Kernel, inferred: KernelABI) -> None:
    abi = kernel.abi
    if abi is None:
        raise ValueError("kernel.abi is required")
    _check_values(abi.inputs, inferred.inputs, "input")
    _check_values(abi.outputs, inferred.outputs, "output")


def _check_values(
    actual: Sequence[ABIValue],
    expected: Sequence[ABIValue],
    role: str,
) -> None:
    if len(actual) != len(expected):
        raise ValueError(f"kernel ABI {role} count does not match the problem")
    for value, fixed in zip(actual, expected, strict=True):
        if (value.name, _dtype(value.dtype), value.shape) != (
            fixed.name,
            _dtype(fixed.dtype),
            fixed.shape,
        ):
            raise ValueError(
                f"kernel ABI {role} {value.name!r} does not match the problem"
            )


def _check_tensor(value: Any, spec: ABIValue, role: str) -> torch.Tensor:
    tensor = _require_cuda_tensor(value, f"ABI {role} {spec.name!r}")
    if tensor.dtype != _dtype(spec.dtype):
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong dtype")
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong shape")
    stride = spec.constraints.get("stride")
    if stride is not None and tuple(tensor.stride()) != tuple(stride):
        raise ValueError(f"ABI {role} {spec.name!r} has the wrong stride")
    return tensor


def _allocate(spec: ABIValue, device: torch.device) -> torch.Tensor:
    shape = _fixed_shape(spec)
    stride = spec.constraints.get("stride")
    if stride is None:
        return torch.empty(shape, dtype=_dtype(spec.dtype), device=device)
    return torch.empty_strided(
        shape,
        tuple(int(value) for value in stride),
        dtype=_dtype(spec.dtype),
        device=device,
    )


def _fixed_shape(spec: ABIValue) -> tuple[int, ...]:
    if any(isinstance(value, str) for value in spec.shape):
        raise ValueError(f"raw CUDA output {spec.name!r} must have a fixed shape")
    return tuple(int(value) for value in spec.shape)


def _dtype(value: str | None) -> torch.dtype:
    if value is None:
        raise ValueError("raw CUDA ABI values require a dtype")
    name = value.lower().removeprefix("torch.")
    try:
        return _DTYPE_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {value!r}") from exc


def _require_cuda_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{label} must be on CUDA")
    return value


def _problem_name(module: ModuleType, path: Path) -> str:
    declared = getattr(module, "PROBLEM_NAME", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return path.parent.name or path.stem


def _platform() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the evaluator worker")
    major, minor = torch.cuda.get_device_capability()
    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("PyTorch does not report its CUDA version")
    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    return f"nvidia-sm{major}{minor}-cuda{cuda_major}"


def _config(value: Mapping[str, Any]) -> _Config:
    problem_path = _path(value, "problem_path", kind="file")
    build_root = _path(value, "build_root", kind="directory")
    include_value = value.get("include_paths", ())
    if not isinstance(include_value, Sequence) or isinstance(
        include_value, (str, bytes)
    ):
        raise TypeError("config.include_paths must be an array")
    include_paths = tuple(_existing_dir(item, "include path") for item in include_value)
    timing_value = value.get("timing", {})
    if not isinstance(timing_value, Mapping):
        raise TypeError("config.timing must be an object")
    timing = TimingPolicy(
        warmup=timing_value.get("warmup", _DEFAULT_TIMING.warmup),
        repeat=timing_value.get("repeat", _DEFAULT_TIMING.repeat),
        trials=timing_value.get("trials", _DEFAULT_TIMING.trials),
        cold_l2=timing_value.get("cold_l2", _DEFAULT_TIMING.cold_l2),
    )
    seed = value.get("seed", _DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("config.seed must be an integer")
    return _Config(
        problem_path=problem_path,
        build_root=build_root,
        include_paths=include_paths,
        timing=timing,
        seed=seed,
    )


def _path(value: Mapping[str, Any], key: str, *, kind: str) -> Path:
    if key not in value:
        raise ValueError(f"config.{key} is required")
    return _resolved_path(value[key], f"config.{key}", kind=kind)


def _resolved_path(value: Any, label: str, *, kind: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path string")
    path = Path(value).expanduser().absolute().resolve(strict=True)
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


def _existing_dir(value: Any, label: str) -> Path:
    return _resolved_path(value, label, kind="directory")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _operation(request: Mapping[str, Any]) -> str:
    value = request.get("operation")
    if value not in {"inspect", "evaluate"}:
        raise ValueError("operation must be 'inspect' or 'evaluate'")
    return value


def _failure(exc: Exception) -> ValidationResult:
    message = str(exc).strip() or type(exc).__name__
    return ValidationResult(
        compile_passed=False,
        correctness_passed=False,
        profile_passed=False,
        details={"worker": {"error_type": type(exc).__name__, "error": message}},
    )


def _read_request() -> Mapping[str, Any]:
    return _mapping(json.load(sys.stdin), "worker request")


def main() -> int:
    operation = ""
    try:
        request = _read_request()
        operation = _operation(request)
        with contextlib.redirect_stdout(sys.stderr):
            if operation == "inspect":
                seed = request.get("seed", _DEFAULT_SEED)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("seed must be an integer")
                result: Any = inspect_problem(request["problem_path"], seed=seed)
            else:
                result = evaluate_request(request).to_dict()
    except Exception as exc:  # noqa: BLE001 - worker failures cross a JSON boundary
        message = str(exc).strip() or type(exc).__name__
        print(f"{type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        if operation == "evaluate":
            result = _failure(exc).to_dict()
        else:
            result = {"error_type": type(exc).__name__, "error": message}
            json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
            sys.stdout.write("\n")
            return 1
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
