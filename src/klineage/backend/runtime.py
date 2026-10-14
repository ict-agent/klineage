"""Build source bundles and adapt calls to the tensor ABI."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib.util
import json
import keyword
import os
import sys
import threading
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeGuard
from uuid import uuid4

from klineage.serialization import raw_text
from klineage.artifact.problem import load_problem
from klineage.artifact.tree import (
    read_source_tree,
    relative_source_path,
    source_files,
)
from klineage.backend import Backend, BackendKind, get_backend
from klineage.constants import (
    BUNDLE_CONFIG,
    BUNDLE_SOLUTION,
    MODULE_HASH_LENGTH,
)
from klineage.errors import StructuredOutputError
from klineage.utils import mapping, nonempty

if TYPE_CHECKING:
    import torch

    from klineage.artifact.kernel import Kernel
    from klineage.artifact.problem import Problem

IMPORT_LOCK = threading.RLock()
DEVICE_ROOT = Path("/dev")


class OutputStyle(StrEnum):
    RETURN = "return"
    DESTINATION = "destination"


def device_nodes(backend: Backend) -> tuple[Path, ...]:
    """Find readable character devices for a backend without changing permissions."""
    return tuple(sorted({
        path
        for pattern in backend.device_patterns
        for path in DEVICE_ROOT.glob(pattern)
        if path.is_char_device() and os.access(path, os.R_OK)
    }))


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
            with suppress(ValueError):
                sys.path.remove(str(root))
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


def load_torch(backend):
    """Load the matching tensor runtime only when execution needs it."""
    import torch

    hip = getattr(torch.version, "hip", None)
    if backend.kind == BackendKind.CUDA and hip:
        raise RuntimeError("CUDA backend cannot use HIP-enabled PyTorch")
    if backend.kind == BackendKind.HYGON and not hip:
        raise RuntimeError("Hygon backend requires HIP-enabled DTK PyTorch")
    if backend.device_type == "npu":
        importlib.import_module("torch_npu")
    return torch


def device_runtime(backend):
    runtime = getattr(load_torch(backend), backend.device_type, None)
    if runtime is None or not runtime.is_available():
        raise RuntimeError(f"{backend.kind} is unavailable to this process")
    return runtime


def current_device(backend):
    runtime = device_runtime(backend)
    return load_torch(backend).device(backend.device_type, runtime.current_device())


def runtime_info(backend, torch):
    runtime = getattr(torch, backend.device_type, None)
    available = runtime is not None and runtime.is_available()
    return {
        "language": backend.language,
        "device_type": backend.device_type,
        "device": f"{backend.device_type}:{runtime.current_device()}"
        if available
        else None,
        "available": available,
        "torch_version": torch.__version__,
        "abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


@dataclass(frozen=True, slots=True)
class BundleEntry:
    language: str
    source_path: str
    symbol: str
    output_style: OutputStyle


def bundle_entry(sources, backend):
    """Read the backend's build contract only when loading executable sources."""
    if not sources:
        raise ValueError("submission/config.toml: missing executable source bundle")
    for name, text in sources.items():
        relative_source_path(name, f"submission/{name}")
        if "\x00" in text:
            raise ValueError(f"submission/{name}: source contains NUL bytes")
    if BUNDLE_CONFIG not in sources:
        raise ValueError("submission/config.toml: missing host entry configuration")

    try:
        config = tomllib.loads(sources[BUNDLE_CONFIG])
        for name in sources:
            if name != BUNDLE_CONFIG and not name.startswith(f"{BUNDLE_SOLUTION}/"):
                raise ValueError(
                    "bundles contain only config.toml and solution/ sources"
                )
        solution = mapping(config[BUNDLE_SOLUTION], "solution metadata")
        for key in ("name", "definition", "author"):
            nonempty(solution[key], f"solution {key}")
        build = mapping(config["build"], "build metadata")
        language = build["language"]
        suffixes = (".py",)
        if language != "python":
            if language != backend.language:
                raise ValueError("build language differs from the device backend")
            suffixes = backend.kernel_type.entry_suffixes
        entry = nonempty(build["entry_point"], "build entry_point")
        parts = entry.split("::")
        if entry != build["entry_point"] or len(parts) != 2:
            raise ValueError("entry_point must be relative/source::symbol")
        path, symbol = parts
        relative_source_path(path, "build entry source")
        if Path(path).suffix not in suffixes:
            raise ValueError("entry source suffix is incompatible with build language")
        if not symbol.isidentifier() or keyword.iskeyword(symbol):
            raise ValueError("entry symbol must be an identifier")
        destination = build.get("destination_passing_style", False)
        if type(destination) is not bool:
            raise TypeError("destination_passing_style must be a boolean")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"submission/{BUNDLE_CONFIG}: {error}") from error
    source_path = f"{BUNDLE_SOLUTION}/{path}"
    if source_path not in sources:
        raise ValueError(f"submission/{source_path}: missing entry source")
    nonempty(sources[source_path], f"submission/{source_path}")
    style = OutputStyle.DESTINATION if destination else OutputStyle.RETURN
    return BundleEntry(language, source_path, symbol, style)


def destination_signature(
    problem: Problem, symbol: str = "kernel", *, language: str | None = None,
) -> str:
    """Describe the destination entry in definition order."""
    if (language or problem.language) == "python":
        parameters = [*problem.definition["inputs"], *problem.definition["outputs"]]
        return f"def {symbol}({', '.join(parameters)}) -> None:"
    parameters = [
        f"const at::Tensor& {name}" for name in problem.definition["inputs"]
    ] + [f"at::Tensor& {name}" for name in problem.definition["outputs"]]
    return f"void {symbol}({', '.join(parameters)});"


def check_candidate_entry(sources, problem: Problem) -> None:
    """Require caller-owned outputs for every service-evaluated candidate."""
    entry = bundle_entry(sources, get_backend(problem.language, problem.platform))
    if entry.output_style is OutputStyle.DESTINATION:
        return
    signature = destination_signature(problem, entry.symbol, language=entry.language)
    raise ValueError(
        "candidates require destination_passing_style = true; "
        f"expected entry: {signature}"
    )


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
            for name, content in sources.items():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
            try:
                staging.rename(directory)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise

    if read_source_tree(directory, "source snapshot") != sources:
        raise StructuredOutputError(f"source snapshot was modified: {directory}")
    return str(directory)


class BundleLoader:
    def __init__(
        self, build_root: Path, *, include_paths: Sequence[Path] = (), compiler=None
    ):
        self.build_root = Path(build_root).resolve()
        self.compiler = compiler
        self.include_paths = tuple(Path(path).resolve() for path in include_paths)

    def build(
        self,
        kernel: Kernel,
        *,
        backend: Backend,
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        problem = load_problem(kernel)
        target = get_backend(problem.language, problem.platform)
        if target.kind != backend.kind:
            raise ValueError("kernel backend differs from its problem target")
        sources = source_files(kernel.find("submission"))
        entry = bundle_entry(sources, backend)
        metadata = json.loads(raw_text(kernel.find("metadata.json")))
        function = self.load_entry(sources, entry, backend, metadata.get("compile_flags", ()))
        if (
            entry.language == "python"
            and entry.output_style is OutputStyle.RETURN
            and strides is None
            and all(
                spec["shape"] is None
                or (not spec["shape"] and spec.get("dtype") is None)
                for role in ("inputs", "outputs")
                for spec in problem.definition[role].values()
            )
        ):
            return function
        return BundleCallable(
            problem, function, entry.output_style, strides, workload=workload
        )

    def load_entry(self, sources, entry, backend, compile_flags=()):
        if isinstance(compile_flags, str) or any(not isinstance(flag, str) for flag in compile_flags):
            raise TypeError("metadata.json/compile_flags: expected strings")
        if entry.language == "python":
            load_torch(backend)
            root = Path(snapshot(self.build_root, sources))
            return self.load_python(root, entry)
        module = self.load_native(sources, compile_flags)
        return self.entry(module, entry.symbol)

    def load_python(self, root: Path, entry: BundleEntry) -> Callable[..., Any]:
        # The immutable source snapshot already supplies a content-derived name.
        name = f"klineage_kernel_{root.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(name, root / entry.source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Python entry {entry.source_path!r}")
        module = importlib.util.module_from_spec(spec)
        modules = {name: module}
        directory = root / BUNDLE_SOLUTION
        with source_import_scope(directory, modules):
            spec.loader.exec_module(module)
            function = self.entry(module, entry.symbol)

        # Delayed imports need the same local modules as initial imports.
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

    def load_native(self, sources, compile_flags=()):
        native = {
            name: text
            for name, text in sources.items()
            if name != BUNDLE_CONFIG and Path(name).suffix != ".py"
        }
        if self.compiler is None:
            raise TypeError("native sources require the kernel compiler")
        return self.compiler(
            native,
            self.build_root,
            include_paths=self.include_paths,
            compile_flags=compile_flags,
        )


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


@dataclass(frozen=True, slots=True)
class OutputState:
    storage: Any
    metadata: tuple[Any, ...]

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> OutputState:
        # Retain the storage so replacement cannot reuse its identity or address.
        storage = tensor.untyped_storage()
        return cls(storage, (
            storage.data_ptr(), storage.nbytes(), tensor.data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tensor.stride(),
            tensor.dtype, tensor.device, tensor.layout,
        ))

    def check(self, tensor: torch.Tensor, name: str) -> None:
        current = self.capture(tensor)
        if (
            self.storage._cdata != current.storage._cdata
            or self.metadata != current.metadata
        ):
            raise ValueError(f"ABI output {name!r} changed its storage or metadata")


class BundleCallable:
    def __init__(
        self,
        problem: Problem,
        function: Any,
        style: OutputStyle,
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
        *,
        workload: Mapping[str, Any] | None = None,
    ):
        self.inputs = tensor_values(problem, "inputs", workload)
        self.outputs = tensor_values(problem, "outputs", workload)
        self.backend = get_backend(problem.language, problem.platform)
        self.function, self.style = function, style
        self.strides = strides or {}

    def __call__(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.inputs):
            raise ValueError("bundle input count does not match its ABI")
        axes = {}
        tensors = tuple(
            check_tensor(
                value,
                name,
                spec,
                "input",
                self.strides.get(("input", name)),
                backend=self.backend,
                axes=axes,
            )
            for value, (name, spec) in zip(inputs, self.inputs.items(), strict=True)
        )
        device = tensors[0].device if tensors else current_device(self.backend)
        if any(value.device != device for value in tensors):
            raise ValueError("bundle inputs must share a device")
        outputs_spec = {
            name: spec | {"shape": list(fixed_shape(name, spec, axes))}
            for name, spec in self.outputs.items()
        }

        # The evaluator supplies trailing destinations; return-style entries own allocation.
        with device_runtime(self.backend).device(device):
            if self.style is OutputStyle.DESTINATION:
                outputs = tuple(
                    allocate(name, spec, device, self.strides.get(("output", name)))
                    for name, spec in outputs_spec.items()
                )
                states = tuple(OutputState.capture(value) for value in outputs)
                result = self.function(*tensors, *outputs)
                if result is not None:
                    raise ValueError("destination entry must return None (C++ void)")
                for state, value, name in zip(states, outputs, outputs_spec, strict=True):
                    state.check(value, name)
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

            for value, (name, spec) in zip(outputs, outputs_spec.items(), strict=True):
                check_tensor(
                    value,
                    name,
                    spec,
                    "output",
                    self.strides.get(("output", name)),
                    backend=self.backend,
                )
                if value.device != device:
                    raise ValueError("bundle outputs must use the input device")
        if not outputs:
            return None
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def tensor_values(
    problem: Problem,
    role: str,
    workload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    values = problem.values(role, workload)
    if any(spec["shape"] is None for spec in values.values()):
        raise TypeError("tensor entries do not support scalar arguments or outputs")
    return values


def check_tensor(
    value: Any,
    name: str,
    spec: Mapping[str, Any],
    role: str,
    stride: tuple[int, ...] | None = None,
    *,
    backend: Backend,
    axes: dict[str, int] | None = None,
) -> torch.Tensor:
    tensor = require_tensor(value, f"ABI {role} {name!r}", backend)
    if tensor.dtype != tensor_dtype(spec["dtype"]):
        raise ValueError(f"ABI {role} {name!r} has the wrong dtype")
    if len(tensor.shape) != len(spec["shape"]):
        raise ValueError(f"ABI {role} {name!r} has the wrong shape")
    for expected, actual in zip(spec["shape"], tensor.shape, strict=True):
        if isinstance(expected, str) and axes is not None:
            if axes.setdefault(expected, actual) != actual:
                raise ValueError(
                    f"ABI {role} {name!r} has conflicting axis {expected!r}"
                )
            continue
        if expected != actual:
            raise ValueError(f"ABI {role} {name!r} has the wrong shape")
    if stride is not None and tuple(tensor.stride()) != tuple(stride):
        raise ValueError(f"ABI {role} {name!r} has the wrong stride")
    return tensor


def allocate(
    name: str,
    spec: Mapping[str, Any],
    device: torch.device,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    import torch

    shape = fixed_shape(name, spec)
    if stride is None:
        return torch.empty(shape, dtype=tensor_dtype(spec["dtype"]), device=device)
    return torch.empty_strided(
        shape,
        tuple(int(value) for value in stride),
        dtype=tensor_dtype(spec["dtype"]),
        device=device,
    )


def fixed_shape(
    name: str, spec: Mapping[str, Any], axes: Mapping[str, int] | None = None
) -> tuple[int, ...]:
    """Resolve output axes from this call's inputs, never from another workload."""
    axes = axes or {}
    shape = []
    for axis in spec["shape"]:
        if isinstance(axis, str):
            if axis not in axes:
                raise ValueError(
                    f"output {name!r}: cannot infer axis {axis!r} from inputs"
                )
            axis = axes[axis]
        shape.append(axis)
    return tuple(shape)


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
    torch = load_torch(backend)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != backend.device_type:
        raise ValueError(f"{label} must be on {backend.device_type}")
    if backend.device_type == "npu":
        from .ascend.kernel import check_storage

        check_storage(value)
    return value


COMPILE_LOCK = threading.RLock()
COMPILED_MODULES = {}
BUILD_LOCK = ".klineage-build.lock"
BUILD_SOURCES = "sources"
BUILD_ERROR_LOG = "build-error.log"
COMMON_SUFFIXES = (".c", ".cc", ".cpp", ".cxx")


def compile_sources(sources, build_root, includes, options, suffixes, compiler):
    """Reuse frozen bundles; rebuild whenever external include paths are supplied."""
    sources = {relative_source_path(name): text for name, text in sources.items()}
    native = sorted(
        name for name in sources if Path(name).suffix in (*suffixes, *COMMON_SUFFIXES)
    )
    if not native:
        raise ValueError("build requires native source files")
    includes = tuple(str(Path(path).resolve()) for path in includes)
    build_root = Path(build_root).resolve()
    # PyTorch caches names globally; mutable external headers need a fresh identity.
    if includes:
        digest = uuid4().hex
    else:
        digest = hashlib.sha256(
            json.dumps([sources, options, includes, str(build_root)], sort_keys=True).encode()
        ).hexdigest()[:MODULE_HASH_LENGTH]
    name = f"klineage_{options['language']}_{digest}"
    directory = build_root / name

    with COMPILE_LOCK:
        if not includes and directory in COMPILED_MODULES:
            return COMPILED_MODULES[directory]
        directory.mkdir(parents=True, exist_ok=True)
        with native_sources(directory, sources) as root:
            paths = [str(root / path) for path in native]
            try:
                module = compiler(
                    name,
                    paths,
                    directory,
                    [str(root / BUNDLE_SOLUTION), *includes],
                    options,
                )
            except Exception as error:
                # Loading can fail before an evaluation worker records diagnostics.
                log = directory / BUILD_ERROR_LOG
                log.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
                error.add_note(f"Full build diagnostics: {log}")
                raise
        if not includes:
            COMPILED_MODULES[directory] = module
        return module


@contextmanager
def native_sources(directory, sources):
    # Protect compiler-writable sources across processes using the same cache.
    with (directory / BUILD_LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        root = directory / BUILD_SOURCES
        for name, source in sources.items():
            target = root / name
            content = source.encode("utf-8")
            # Preserve timestamps so a fresh worker can reuse compiled objects.
            if target.is_file() and target.read_bytes() == content:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        yield root


def bench_events(fn, *, backend, args=(), policy=None):
    """Measure current-stream intervals; allocate timing events before recording."""
    import statistics

    from .timing import EVENT_SCOPE, TimingPolicy, TimingResult, validated_samples

    policy = policy or TimingPolicy(cold_l2=False)
    if policy.cold_l2:
        raise ValueError("cold_l2 is unsupported by HIP/NPU event timing")
    if not callable(fn):
        raise TypeError("fn must be callable")
    runtime, torch = device_runtime(backend), load_torch(backend)
    device = next((value.device for value in args if hasattr(value, "device")), None)
    if device is None:
        device = current_device(backend)
    trials = []
    with runtime.device(device), torch.inference_mode():
        stream = runtime.current_stream(device)
        for _ in range(policy.trials):
            for _ in range(policy.warmup):
                fn(*args)
            runtime.synchronize(device)
            events = [
                (runtime.Event(enable_timing=True), runtime.Event(enable_timing=True))
                for _ in range(policy.repeat)
            ]
            samples = []
            for start, end in events:
                start.record(stream)
                output = fn(*args)
                end.record(stream)
                end.synchronize()
                samples.append(start.elapsed_time(end))
                del output
            trials.append(validated_samples(samples))

    return TimingResult(
        samples_ms=tuple(trials),
        median_ms=float(statistics.median(map(statistics.median, trials))),
        backend_used=backend.timing.NAME,
        details={
            "policy": asdict(policy),
            "scope": EVENT_SCOPE,
            "device_type": backend.device_type,
        },
    )
