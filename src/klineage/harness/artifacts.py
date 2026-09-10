"""Source snapshots, workload artifacts, builds, and process resources."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard

from klineage.backend import Backend, get_backend, platform_backend
from klineage.constants import BUNDLE_CONFIG, BUNDLE_SOLUTION, KERNEL_FILE
from klineage.contract import (
    ABIValue,
    OutputStyle,
    ProblemSpec,
    ValueRole,
    relative_source_path,
)
from klineage.errors import StructuredOutputError
from klineage.tools import agent_function

if TYPE_CHECKING:
    import torch

    from klineage.artifact.kernel import Kernel


_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_FILES = 1024
_GRACE_SECONDS = 5
DEFINITIONS = "definitions"
WORKLOADS = "workloads"
STRIDES_METADATA = "klineage.strides"
IMPORT_LOCK = threading.RLock()
HASH_LENGTH = 20
BUILD_SOURCES = "sources"
BUILD_LOCK = ".klineage-build.lock"
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


@agent_function
def read_source_tree(directory: Path, label: str) -> dict[str, str]:
    """Read a complete source directory into relative-path → exact-text entries.

    Reject symlinks, binary files, and escaping paths. Use label to identify the
    bundle in errors. Pass the returned map to Kernel.from_sources.
    """
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise StructuredOutputError(f"{label} cannot be a symbolic link")
    directory = directory.resolve(strict=True)
    files: dict[str, str] = {}
    total_bytes = 0
    pending = [directory]
    while pending:
        parent = pending.pop()
        for path in sorted(parent.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise StructuredOutputError(f"{label} cannot contain symbolic links")
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file():
                raise StructuredOutputError(
                    f"{label} can contain only regular files and directories"
                )
            if len(files) >= _MAX_SOURCE_FILES:
                raise StructuredOutputError(
                    f"{label} contains more than {_MAX_SOURCE_FILES} files"
                )
            try:
                size = path.stat(follow_symlinks=False).st_size
            except OSError as error:
                raise StructuredOutputError(f"cannot stat {label} file") from error
            if total_bytes + size > _MAX_SOURCE_BYTES:
                raise StructuredOutputError(f"{label} exceeds the 16 MiB total limit")
            try:
                relative = path.relative_to(directory).as_posix()
            except ValueError as error:
                raise StructuredOutputError(
                    f"{label} file is outside its submission directory"
                ) from error
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise StructuredOutputError(f"cannot resolve {label} file") from error
            if not resolved.is_relative_to(directory):
                raise StructuredOutputError(
                    f"{label} file is outside its submission directory"
                )
            try:
                portable_path = relative_source_path(relative, f"{label} path")
                source = path.read_bytes().decode("utf-8")
            except UnicodeDecodeError as error:
                raise StructuredOutputError(
                    f"{label} file {relative!r} is not UTF-8 text"
                ) from error
            except (TypeError, ValueError) as error:
                raise StructuredOutputError(str(error)) from error
            if "\x00" in source:
                raise StructuredOutputError(
                    f"{label} file {relative!r} contains a NUL byte"
                )
            total_bytes += len(source.encode("utf-8"))
            if total_bytes > _MAX_SOURCE_BYTES:
                raise StructuredOutputError(f"{label} exceeds the 16 MiB total limit")
            files[portable_path] = source
    return dict(sorted(files.items()))


def snapshot(work: Path, sources: Mapping[str, str]) -> str:
    sources = {
        relative_source_path(name, "snapshot source"): source
        for name, source in sources.items()
    }
    digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    directory = work / f"source-{digest}"
    if not directory.exists():
        work.mkdir(parents=True, exist_ok=True)
        # Publish complete sources atomically; concurrent builds reuse the winner.
        with TemporaryDirectory(prefix=".source-", dir=work) as temporary:
            staging = Path(temporary)
            for name, source in sources.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            try:
                staging.rename(directory)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise

    if read_source_tree(directory, "source snapshot") != sources:
        raise StructuredOutputError(f"source snapshot was modified: {directory}")
    return str(directory)


@agent_function
def save_kernel(kernel: Kernel, workdir: Path) -> None:
    """Write kernel.json in an existing workdir, replacing its prior metadata.

    Serializes the Kernel's sources, problem, and validation; it does not reread
    edited disk sources or compile. Reconstruct the Kernel after source edits.
    """
    path = workdir / KERNEL_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(kernel.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@agent_function
def load_kernel(workdir: Path) -> Kernel:
    """Restore kernel.json from a directory without compiling or running it.

    Returns a Kernel with embedded sources and problem. Evaluation builds it in
    a worker; call Kernel.build before direct execution in this process.
    """
    from klineage.artifact.kernel import Kernel

    return Kernel.from_dict(
        json.loads((workdir / KERNEL_FILE).read_text(encoding="utf-8"))
    )


def stop_process(process: subprocess.Popen):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(
    command: Sequence[str],
    *,
    payload: str,
    timeout: float,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        # Compilers and profilers may leave descendants after the worker exits.
        stop_process(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from error
    except BaseException:
        stop_process(process)
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


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


@contextmanager
def source_import_scope(root: Path, modules: dict[str, ModuleType]):
    """Prevent ordinary local imports from leaking between two artifacts."""

    with IMPORT_LOCK:
        local_names = local_top_level_names(root)
        displaced = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name in modules or name.split(".", 1)[0] in local_names
        }
        for name in displaced:
            sys.modules.pop(name, None)
        before = set(sys.modules)
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.modules.update(modules)
        sys.path.insert(0, str(root))
        try:
            yield
        finally:
            sys.dont_write_bytecode = previous
            try:
                sys.path.remove(str(root))
            except ValueError:
                pass
            for name in set(sys.modules) - before:
                module = sys.modules.get(name)
                if module_is_under(module, root):
                    modules[name] = module
                    sys.modules.pop(name, None)
            sys.modules.update(displaced)


def local_top_level_names(root: Path) -> set[str]:
    names = {path.stem for path in root.glob("*.py") if path.name != "__init__.py"}
    # Directories without __init__.py are importable namespace packages too.
    names.update(path.name for path in root.iterdir() if path.is_dir())
    return names


def module_is_under(module: ModuleType | None, root: Path) -> TypeGuard[ModuleType]:
    if module is None:
        return False
    filename = getattr(module, "__file__", None)
    locations = (filename,) if filename else getattr(module, "__path__", ())
    try:
        return any(Path(path).resolve().is_relative_to(root) for path in locations)
    except (OSError, RuntimeError, ValueError):
        return False


class BundleLoader:
    def __init__(self, build_root: Path, *, include_paths: Sequence[Path] = ()):
        self.build_root = Path(build_root).resolve()
        self.include_paths = tuple(Path(path).resolve() for path in include_paths)
        self.modules: dict[str, Any] = {}

    def build(
        self,
        kernel: Kernel,
        *,
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
    ) -> Callable[..., Any]:
        function = self.load(kernel)
        if (
            kernel.language == "python"
            and kernel.output_style is OutputStyle.RETURN
            and strides is None
        ):
            return function
        return BundleCallable(kernel.problem, function, kernel.output_style, strides)

    def load(self, kernel: Kernel) -> Callable[..., Any]:
        if BUNDLE_CONFIG in kernel.source_files and kernel.language == "python":
            backend = platform_backend(kernel.problem.platform)
            if backend is not None:
                backend.prepare()
            root = Path(snapshot(self.build_root, kernel.source_files))
            return self.load_python(root, kernel)

        backend = get_backend(kernel.problem.language, kernel.problem.platform)
        if BUNDLE_CONFIG not in kernel.source_files:
            if set(kernel.source_files) != {backend.raw_source}:
                raise ValueError(
                    f"raw {backend.kind} adapters require a single {backend.raw_source} source"
                )
            sources = {
                backend.raw_source: kernel.source_files[backend.raw_source],
                "binding.cpp": backend.binding(),
            }
            native_root = Path(snapshot(self.build_root, sources))
            module = self.load_native(native_root, sources, backend)
            launch = getattr(module, "launch", None)
            if not callable(launch):
                raise TypeError("native extension did not expose launch")
            input_count = len(kernel.problem.values(ValueRole.INPUTS))

            def run(*tensors):
                launch(list(tensors[:input_count]), list(tensors[input_count:]))

            return run
        root = Path(snapshot(self.build_root, kernel.source_files))
        module = self.load_native(root, kernel.source_files, backend)
        return self.entry(module, kernel.symbol)

    def load_python(self, root: Path, kernel: Kernel) -> Callable[..., Any]:
        name = f"_klineage_kernel_{kernel.fingerprint[:HASH_LENGTH]}"
        spec = importlib.util.spec_from_file_location(name, root / kernel.source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Python entry {kernel.source_path!r}")
        module = importlib.util.module_from_spec(spec)
        modules = {name: module}
        directory = root / BUNDLE_SOLUTION
        with source_import_scope(directory, modules):
            spec.loader.exec_module(module)
            function = self.entry(module, kernel.symbol)

        # Delayed imports need the same private modules as initial imports.
        def run(*args):
            with source_import_scope(directory, modules):
                return function(*args)

        return run

    @staticmethod
    def entry(module: Any, symbol: str) -> Callable[..., Any]:
        function = getattr(module, symbol)
        if not callable(function):
            raise TypeError(f"bundle entry point {symbol!r} is not callable")
        return function

    def load_native(
        self, root: Path, source_files: Mapping[str, str], backend: Backend
    ):
        native_files = {
            path: source
            for path, source in source_files.items()
            if path != BUNDLE_CONFIG and Path(path).suffix != ".py"
        }
        sources = [
            path
            for path in sorted(native_files)
            if Path(path).suffix in backend.native_suffixes
        ]
        if not sources:
            raise ValueError(f"{backend.kind} build requires native source files")
        include_paths = [str(root / BUNDLE_SOLUTION), *map(str, self.include_paths)]

        with IMPORT_LOCK:
            options = backend.build_options()
            # Cache complete sources, flags, includes, and architecture together.
            digest = hashlib.sha256(
                json.dumps(
                    [backend.kind, native_files, options, include_paths],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:HASH_LENGTH]
            name = f"klineage_{backend.kind}_{digest}"
            if name in self.modules:
                return self.modules[name]
            directory = self.build_root / name
            directory.mkdir(parents=True, exist_ok=True)
            # Compilers may write beside inputs; frozen snapshots stay read-only.
            with native_sources(directory, native_files) as compile_root:
                sources = [str(compile_root / path) for path in sources]
                include_paths[0] = str(compile_root / BUNDLE_SOLUTION)
                module = backend.compile(
                    name, sources, directory, include_paths, options
                )
            self.modules[name] = module
            return module


@contextmanager
def native_sources(directory: Path, sources: Mapping[str, str]):
    # Lock staging as well as compilation across worker processes.
    with (directory / BUILD_LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = directory / BUILD_SOURCES
        for name, source in sources.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.encode("utf-8"))
        yield root


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


def load_problem(value: str | Path) -> tuple[ModuleType, Path]:
    path = Path(value).expanduser().absolute().resolve(strict=True)
    if path.is_file() and path.suffix == ".json":
        return load_trace(path), path
    if not path.is_file() or path.suffix != ".py":
        raise ValueError(
            "problem_path must identify a Trace definition JSON or Python file"
        )
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


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def make_log_dir(path: Path) -> Path:
    directory = Path(path).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def write_json(path: Path, value: Any):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def existing_dir(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path string")
    path = resolve_path(value)
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path
