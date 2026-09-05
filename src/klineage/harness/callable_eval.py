"""Evaluator for source-directory kernels exposing a Python callable ABI."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from klineage.harness.eval import ValidationResult
from klineage.harness.timing import KernelTimer

if TYPE_CHECKING:
    from klineage.kernel import Kernel


@dataclass(frozen=True, slots=True)
class CallInputs:
    """Positional arguments for one kernel invocation."""

    args: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True, slots=True)
class ProblemRuntime:
    """Problem-owned input, reference, and output-comparison callbacks.

    ``check_outputs`` receives ``(candidate_output, reference_output)``.  It may
    return a boolean, or raise an assertion for a detailed failure.  Returning
    ``None`` is treated as success to support assertion-style checkers. The
    problem reference is always the correctness oracle; a supplied Kernel
    reference is only the performance baseline.
    """

    make_inputs: Callable[[], CallInputs]
    reference: Callable[..., Any]
    check_outputs: Callable[[Any, Any], bool | None]

    def __post_init__(self) -> None:
        for name in ("make_inputs", "reference", "check_outputs"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


class KernelLoader(Protocol):
    """Build, import, and expose the callable declared by a kernel artifact."""

    def load(self, kernel: Kernel) -> Callable[..., Any]: ...


class PythonEntrypointLoader:
    """Load ``python-callable-v1`` from a kernel's source directory.

    The kernel ABI names an entry module and a zero-argument loader function.
    Calling that function may perform a native extension build, but it must
    ultimately return the callable that the evaluator invokes.
    """

    def load(self, kernel: Kernel) -> Callable[..., Any]:
        artifact_path = kernel.artifact_path
        if artifact_path is None:
            raise ValueError("kernel.artifact_path is required for callable evaluation")
        unresolved_root = Path(artifact_path).absolute()
        if unresolved_root.is_symlink():
            raise ValueError("kernel.artifact_path cannot be a symbolic link")
        root = unresolved_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("kernel.artifact_path must be a source directory")

        abi = kernel.abi
        if abi is None:
            raise ValueError("kernel.abi is required for callable evaluation")
        interface = abi.interface
        module_name = interface.module
        loader_name = interface.loader
        entrypoint = _resolve_entrypoint(root, module_name)

        module = _import_source_module(root, entrypoint, kernel)
        try:
            loader = getattr(module, loader_name)
        except AttributeError as exc:
            raise AttributeError(
                f"entry module {module_name!r} has no loader {loader_name!r}"
            ) from exc
        if not callable(loader):
            raise TypeError(f"entrypoint loader {loader_name!r} is not callable")
        # A loader commonly performs the actual extension build and may import
        # helper modules from elsewhere in the submission directory.
        with _IMPORT_LOCK, _source_import_scope(root, ""):
            loaded = loader()
        if not callable(loaded):
            raise TypeError(
                f"entrypoint loader {loader_name!r} did not return a callable"
            )
        return loaded


class CallableKernelEvaluator:
    """Run build/load, oracle correctness, and timing as explicit gates."""

    def __init__(
        self,
        runtime: ProblemRuntime,
        *,
        loader: KernelLoader | None = None,
        timer: KernelTimer,
    ) -> None:
        self.runtime = runtime
        self.loader = loader or PythonEntrypointLoader()
        self.timer = timer

    def evaluate(
        self,
        kernel: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        details: dict[str, Any] = {}

        try:
            abi = kernel.abi
            if abi is None:
                raise ValueError("kernel.abi is required for callable evaluation")
            if reference is not None:
                _require_compatible_contracts(kernel, reference)
            candidate_fn = self.loader.load(kernel)
            timing_reference_fn = (
                self.loader.load(reference)
                if reference is not None
                else self.runtime.reference
            )
        except Exception as exc:  # noqa: BLE001 - failures become gate evidence
            details["build_load"] = _failed_stage(exc)
            details["correctness"] = _skipped_stage("build/load failed")
            details["timing"] = _skipped_stage("build/load failed")
            return ValidationResult(
                compile_passed=False,
                correctness_passed=False,
                profile_passed=False,
                details=details,
            )
        try:
            original_inputs = _require_call_inputs(
                self.runtime.make_inputs(),
                len(abi.inputs),
            )
            candidate_inputs = _clone_inputs(original_inputs)
            oracle_inputs = _clone_inputs(original_inputs)
            expected = self.runtime.reference(
                *oracle_inputs.args,
            )
            actual = candidate_fn(
                *candidate_inputs.args,
            )
            _require_output_contract(expected, len(abi.outputs), "reference")
            _require_output_contract(actual, len(abi.outputs), "candidate")
            verdict = self.runtime.check_outputs(actual, expected)
            if verdict is not None and not bool(verdict):
                details["correctness"] = {
                    "reason": "output checker rejected candidate output",
                }
                details["timing"] = _skipped_stage("correctness failed")
                return ValidationResult(
                    compile_passed=True,
                    correctness_passed=False,
                    profile_passed=False,
                    details=details,
                )
        except Exception as exc:  # noqa: BLE001 - failures become gate evidence
            details["correctness"] = _failed_stage(exc)
            details["timing"] = _skipped_stage("correctness failed")
            return ValidationResult(
                compile_passed=True,
                correctness_passed=False,
                profile_passed=False,
                details=details,
            )
        try:
            timing_inputs = _require_call_inputs(
                self.runtime.make_inputs(),
                len(abi.inputs),
            )
            candidate_timing_inputs = _clone_inputs(timing_inputs)
            reference_timing_inputs = _clone_inputs(timing_inputs)
            candidate_timing = self.timer.measure(
                candidate_fn,
                args=candidate_timing_inputs.args,
            )
            reference_timing = self.timer.measure(
                timing_reference_fn,
                args=reference_timing_inputs.args,
            )
        except Exception as exc:  # noqa: BLE001 - failures become gate evidence
            details["timing"] = _failed_stage(exc)
            return ValidationResult(
                compile_passed=True,
                correctness_passed=True,
                profile_passed=False,
                details=details,
            )

        details["timing"] = {
            "candidate": candidate_timing.to_dict(),
            "reference": reference_timing.to_dict(),
        }
        return ValidationResult(
            compile_passed=True,
            correctness_passed=True,
            profile_passed=True,
            latency_ms=candidate_timing.median_ms,
            reference_latency_ms=reference_timing.median_ms,
            details=details,
        )


def _require_compatible_contracts(candidate: Kernel, reference: Kernel) -> None:
    candidate_abi = candidate.abi
    reference_abi = reference.abi
    if candidate_abi != reference_abi:
        raise ValueError("candidate and reference kernel ABIs do not match")

    candidate_problem = candidate.problem
    reference_problem = reference.problem
    if candidate_problem != reference_problem:
        raise ValueError("candidate and reference problem contracts do not match")


def _require_call_inputs(value: object, expected_count: int) -> CallInputs:
    if not isinstance(value, CallInputs):
        raise TypeError("ProblemRuntime.make_inputs() must return CallInputs")
    if len(value.args) != expected_count:
        raise ValueError(
            f"ProblemRuntime.make_inputs() returned {len(value.args)} positional "
            f"arguments; ABI declares {expected_count}"
        )
    return value


def _require_output_contract(value: object, output_count: int, label: str) -> None:
    if output_count == 0 and value is not None:
        raise ValueError(f"{label} must return None for an ABI with no outputs")
    if output_count > 1 and (
        not isinstance(value, tuple) or len(value) != output_count
    ):
        raise ValueError(
            f"{label} must return a {output_count}-item tuple for this ABI"
        )


def _clone_inputs(inputs: CallInputs) -> CallInputs:
    return CallInputs(args=tuple(_clone_value(value) for value in inputs.args))


def _clone_value(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except (TypeError, RuntimeError):
            pass
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _resolve_entrypoint(root: Path, module_name: str) -> Path:
    relative = Path(module_name)
    if relative.is_absolute():
        raise ValueError("interface.module must be relative to kernel.artifact_path")
    _reject_symlink_components(root, relative)
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("interface.module resolves outside kernel.artifact_path")
    if not resolved.is_file():
        raise ValueError("interface.module must identify a regular Python file")
    return resolved


def _reject_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("interface.module cannot traverse a symbolic link")


_IMPORT_LOCK = threading.RLock()


def _import_source_module(root: Path, entrypoint: Path, kernel: Kernel) -> ModuleType:
    digest = hashlib.sha256(kernel.fingerprint.encode()).hexdigest()[:20]
    synthetic_name = f"_klineage_kernel_{digest}"
    spec = importlib.util.spec_from_file_location(synthetic_name, entrypoint)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for {str(entrypoint)!r}")
    module = importlib.util.module_from_spec(spec)

    with _IMPORT_LOCK, _source_import_scope(root, synthetic_name):
        sys.modules[synthetic_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(synthetic_name, None)
            raise
    return module


@contextmanager
def _source_import_scope(root: Path, synthetic_name: str):
    """Prevent ordinary local imports from leaking between two artifacts."""

    local_names = _local_top_level_names(root)
    displaced = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name.split(".", 1)[0] in local_names
    }
    for name in displaced:
        sys.modules.pop(name, None)
    before = set(sys.modules)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
        for name in set(sys.modules) - before:
            module = sys.modules.get(name)
            if name == synthetic_name or _module_is_under(module, root):
                sys.modules.pop(name, None)
        sys.modules.update(displaced)


def _local_top_level_names(root: Path) -> set[str]:
    names = {path.stem for path in root.glob("*.py") if path.name != "__init__.py"}
    names.update(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    return names


def _module_is_under(module: ModuleType | None, root: Path) -> bool:
    if module is None:
        return False
    filename = getattr(module, "__file__", None)
    if not filename:
        return False
    try:
        return Path(filename).resolve().is_relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return False


def _failed_stage(exc: Exception) -> dict[str, Any]:
    message = str(exc).strip()
    return {
        "error_type": type(exc).__name__,
        "error": message or type(exc).__name__,
    }


def _skipped_stage(reason: str) -> dict[str, Any]:
    return {"skipped": True, "reason": reason}


__all__ = [
    "CallInputs",
    "CallableKernelEvaluator",
    "KernelLoader",
    "ProblemRuntime",
    "PythonEntrypointLoader",
]
