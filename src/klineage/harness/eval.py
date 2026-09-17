"""Kernel correctness, device timing, and isolated runtime requests."""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from klineage.artifact.kernel import ValidationResult, save_kernel
from klineage.artifact.problem import (
    load_problem_module,
    trace_definition,
    trace_inputs,
    trace_module,
    trace_workload,
)
from klineage.artifact.tensor import require_tensor, tensor_dtype, tensor_values
from klineage.backend import Backend, BackendKind, detect_backend, get_backend
from klineage.constants import (
    BUILD_DIRECTORY,
    EVALUATIONS_DIRECTORY,
    NUMERICAL_TOLERANCE,
    VERSIONS_DIRECTORY,
    RunKind,
)
from klineage.contract import ProblemSpec, ValueRole
from klineage.errors import ActionError
from klineage.harness.process import run_process
from klineage.harness.timing import (
    KernelTimer,
    TimingPolicy,
    make_timer,
    ncu_command,
    parse_metrics,
    profile_launch,
    timing_evidence,
    timing_policy,
    verify_performance,
)
from klineage.logging import observed
from klineage.tools import agent_function
from klineage.utils import (
    existing_dir,
    make_log_dir,
    operation_id,
    resolve_path,
    write_json,
)

if TYPE_CHECKING:
    import torch

    from klineage.artifact.kernel import Kernel
    from klineage.harness.profiling import ProfileOptions


WORKER_ENTRY = "from klineage.harness.eval import main; raise SystemExit(main())"
# Default worker deadline; enclosing action deadlines are independent.
_TIMEOUT_SECONDS = 900.0
_SEED = 20260903
_MAX_JOBS = 4
_ERROR_TAIL = 2000
_DEFAULT_SEED = 0


class KernelEvaluator(Protocol):
    def evaluate(
        self, kernel: Kernel, *, reference: Kernel | None = None
    ) -> ValidationResult: ...


@dataclass(frozen=True, slots=True)
class CallInputs:
    """Positional arguments for one kernel invocation."""

    args: tuple[Any, ...] = ()

    def __post_init__(self):
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

    def __post_init__(self):
        for name in ("make_inputs", "reference", "check_outputs"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


class CallableKernelEvaluator:
    """Check a built Kernel against the oracle, then measure it."""

    def __init__(self, runtime: ProblemRuntime, *, timer: KernelTimer):
        self.runtime = runtime
        self.timer = timer

    def evaluate(
        self, kernel: Kernel, *, reference: Kernel | None = None
    ) -> ValidationResult:
        try:
            inputs = kernel.problem.values(ValueRole.INPUTS)
            outputs = kernel.problem.values(ValueRole.OUTPUTS)
            if reference is not None:
                require_compatible_contracts(kernel, reference)
            if kernel.function is None or (
                reference is not None and reference.function is None
            ):
                raise ValueError("kernels must be built before evaluation")
        except Exception as error:  # noqa: BLE001 - Record gates; propagate interrupts.
            print(f"Build failed: {error}", file=sys.stderr)
            return ValidationResult(False, False, False)

        try:
            original = require_call_inputs(self.runtime.make_inputs(), len(inputs))
            expected = self.runtime.reference(*clone_inputs(original).args)
            actual = kernel(*clone_inputs(original).args)
            require_output_contract(expected, len(outputs), "reference")
            require_output_contract(actual, len(outputs), "candidate")
            verdict = self.runtime.check_outputs(actual, expected)
            if verdict is not None and not bool(verdict):
                raise ValueError("output checker rejected candidate output")
        except Exception as error:  # noqa: BLE001 - Record gates; propagate interrupts.
            print(f"Correctness failed: {error}", file=sys.stderr)
            return ValidationResult(True, False, False)

        try:
            original = require_call_inputs(self.runtime.make_inputs(), len(inputs))
            timing = self.timer.measure(kernel, args=clone_inputs(original).args)
            timing_evidence(timing.to_dict(), timing.median_ms)
            print("Timing evidence: " + json.dumps(timing.to_dict()), file=sys.stderr)
            reference_latency_ms = None
            passed = True
            if reference is not None:
                baseline = self.timer.measure(
                    reference, args=clone_inputs(original).args
                )
                reference_latency_ms = baseline.median_ms
                passed = verify_performance(timing, baseline)
        except Exception as error:  # noqa: BLE001 - Record gates; propagate interrupts.
            print(f"Timing failed: {error}", file=sys.stderr)
            return ValidationResult(True, True, False)
        return ValidationResult(
            True,
            True,
            passed,
            latency_ms=timing.median_ms,
            reference_latency_ms=reference_latency_ms,
        )


def require_compatible_contracts(candidate: Kernel, reference: Kernel):
    for role in ValueRole:
        if candidate.problem.values(role) != reference.problem.values(role):
            raise ValueError("candidate and reference signatures do not match")

    candidate_problem = candidate.problem
    reference_problem = reference.problem
    if candidate_problem != reference_problem:
        raise ValueError("candidate and reference problem contracts do not match")


def require_call_inputs(value: object, expected_count: int) -> CallInputs:
    if not isinstance(value, CallInputs):
        raise TypeError("ProblemRuntime.make_inputs() must return CallInputs")
    if len(value.args) != expected_count:
        raise ValueError(
            f"ProblemRuntime.make_inputs() returned {len(value.args)} positional "
            f"arguments; ABI declares {expected_count}"
        )
    return value


def require_output_contract(value: object, output_count: int, label: str):
    if output_count == 0 and value is not None:
        raise ValueError(f"{label} must return None for an ABI with no outputs")
    if output_count > 1 and (
        not isinstance(value, tuple) or len(value) != output_count
    ):
        raise ValueError(
            f"{label} must return a {output_count}-item tuple for this ABI"
        )


def clone_inputs(inputs: CallInputs) -> CallInputs:
    return CallInputs(args=tuple(clone_value(value) for value in inputs.args))


def clone_value(value: Any) -> Any:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return clone_tensor(value)
    clone = getattr(value, "clone", None)
    if callable(clone):
        try:
            return clone()
        except (TypeError, RuntimeError):
            pass
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def clone_tensor(value: torch.Tensor) -> torch.Tensor:
    import torch

    cloned = value.clone()
    if value.layout != torch.strided or cloned.stride() == value.stride():
        return cloned

    # Preserve gaps and overlapping views; plain clone densifies these layouts.
    span = 0
    if value.numel():
        span = 1 + sum(
            (size - 1) * step for size, step in zip(value.shape, value.stride())
        )
    storage = value.as_strided((span,), (1,)).clone()
    return storage.as_strided(value.shape, value.stride())


class WorkerError(RuntimeError):
    """A local evaluator worker failed."""


class EvaluationRuntime:
    def __init__(self, *, timeout: float = _TIMEOUT_SECONDS):
        self.timeout = timeout

    def inspect(
        self,
        problem_path: Path,
        *,
        log_dir: Path,
    ) -> Mapping[str, Any]:
        """Load a problem on the available backend and return its contract."""

        logs = make_log_dir(log_dir)
        request = {
            "operation": "inspect",
            "problem_path": str(resolve_path(problem_path)),
            "workload_path": str(logs / "inputs.safetensors"),
            "seed": _SEED,
        }
        payload = self.invoke(
            request,
            logs,
            "inspect",
        )
        if not isinstance(payload, Mapping):
            raise WorkerError("evaluator worker inspect result must be a JSON object")
        if "error" in payload:
            raise WorkerError(str(payload["error"]))
        write_json(logs / "inspect-result.json", payload)

        return dict(payload)

    def evaluate(
        self,
        kernel: Kernel,
        *,
        include_paths: Sequence[Path],
        build_root: Path,
        log_dir: Path,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        """Run the callable evaluator with the selected backend's timing policy."""

        includes = tuple(resolve_path(path) for path in include_paths)
        host_build_root = Path(build_root).expanduser().absolute()
        host_build_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        build = resolve_path(host_build_root)

        config = {
            "seed": _SEED,
            "timing": timing_policy(kernel_backend(kernel)).to_dict(),
            "correctness": {"rtol": NUMERICAL_TOLERANCE, "atol": NUMERICAL_TOLERANCE},
            "include_paths": [str(path) for path in includes],
            "build_root": str(build),
        }
        logs = make_log_dir(log_dir)
        stem = "eval-0001"
        request = {
            "operation": "evaluate",
            "kernel": self.kernel_payload(kernel),
            "reference": (
                self.kernel_payload(reference) if reference is not None else None
            ),
            "config": config,
        }
        payload = self.invoke(request, logs, stem)
        if not isinstance(payload, Mapping):
            raise WorkerError("evaluator worker result must be a JSON object")

        try:
            result = ValidationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise WorkerError(
                "evaluator worker did not emit a ValidationResult"
            ) from error

        write_json(logs / f"{stem}-result.json", result.to_dict())
        return result

    def profile(
        self,
        kernel: Kernel,
        *,
        build_root: Path,
        log_dir: Path,
        include_paths: Sequence[Path] = (),
        options: ProfileOptions,
    ) -> dict[str, Any]:
        require_profile_backend(kernel_backend(kernel))
        logs = make_log_dir(log_dir)
        try:
            return self.collect_profile(
                kernel, build_root, logs, include_paths, options
            )
        except (OSError, ValueError, TypeError, WorkerError) as exc:
            write_json(logs / "profile-error.json", {"error": str(exc)})
            raise ActionError(f"NCU profiling failed: {exc}; see {logs}") from exc

    def collect_profile(self, kernel, build_root, logs, include_paths, options):
        build = make_log_dir(build_root)
        # Separate reports prevent a failed rerun from reusing stale metrics.
        reports = Path(tempfile.mkdtemp(prefix="ncu-", dir=logs))
        raw, report = reports / "metrics.csv", reports / "profile.ncu-rep"
        output = resolve_path(reports)
        request = {
            "operation": "profile",
            "kernel": self.kernel_payload(kernel),
            "config": {
                "build_root": str(resolve_path(build)),
                "seed": _SEED,
                "include_paths": [str(resolve_path(path)) for path in include_paths],
            },
        }
        try:
            payload = self.invoke(
                request,
                reports,
                "profile",
                command_prefix=ncu_command(
                    options, output / raw.name, output / report.name
                ),
                timeout=options.timeout_seconds,
            )
        except WorkerError as exc:
            if raw.is_file():
                try:
                    parse_metrics(raw.read_text(encoding="utf-8"))
                except ValueError as diagnostic:
                    raise WorkerError(f"{exc}; {diagnostic}") from exc
            raise

        if not isinstance(payload, Mapping) or "error" in payload:
            raise WorkerError(f"invalid profile worker result: {payload}")
        if (
            not isinstance(payload.get("tool_version"), str)
            or not payload["tool_version"].strip()
            or not isinstance(payload.get("device"), Mapping)
        ):
            raise WorkerError("profile worker metadata is incomplete")
        if not raw.is_file():
            raise WorkerError("NCU did not produce raw CSV")
        metrics = parse_metrics(raw.read_text(encoding="utf-8"))
        if not report.is_file() or not report.stat().st_size:
            raise WorkerError("NCU did not produce a report")
        result = {
            "tool": "ncu",
            "tool_version": payload["tool_version"],
            "device": payload["device"],
            "metrics": metrics,
            "report_path": str(report),
            "raw_path": str(raw),
            "collected_at": now(),
        }
        write_json(reports / "profile-result.json", result)
        return result

    def kernel_payload(self, kernel: Kernel) -> dict[str, Any]:
        payload = kernel.to_dict()
        payload.pop("validation")
        return payload

    def invoke(
        self,
        request: Mapping[str, Any],
        log_dir: Path,
        stem: str,
        *,
        command_prefix: Sequence[str] = (),
        timeout: float | None = None,
    ) -> Any:
        write_json(log_dir / f"{stem}-request.json", request)
        stdout_path = log_dir / f"{stem}-stdout.log"
        stderr_path = log_dir / f"{stem}-stderr.log"
        process_path = log_dir / f"{stem}-process.json"
        command = (*command_prefix, sys.executable, "-c", WORKER_ENTRY)
        environment = os.environ.copy()
        environment.setdefault("MAX_JOBS", str(_MAX_JOBS))
        source_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_root, environment.get("PYTHONPATH")) if value
        )
        started = now()
        timeout = self.timeout if timeout is None else timeout
        try:
            result = run_process(
                command,
                payload=json.dumps(request),
                timeout=timeout,
                environment=environment,
            )
        except subprocess.TimeoutExpired as error:
            stdout_path.write_text(text(error.stdout), encoding="utf-8")
            stderr_path.write_text(text(error.stderr), encoding="utf-8")
            write_json(
                process_path,
                {
                    "command": command,
                    "started_at": started,
                    "finished_at": now(),
                    "timed_out": True,
                    "timeout_seconds": timeout,
                },
            )
            raise WorkerError(
                f"evaluator worker timed out after {timeout:g}s; see {stderr_path}"
            ) from error
        except OSError as error:
            raise WorkerError(f"Could not launch evaluator worker: {error}") from error

        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        write_json(
            process_path,
            {
                "command": command,
                "started_at": started,
                "finished_at": now(),
                "returncode": result.returncode,
                "timed_out": False,
                "timeout_seconds": timeout,
            },
        )
        if result.returncode != 0:
            raise WorkerError(
                f"evaluator worker exited with status {result.returncode}: "
                f"{result.stderr[-_ERROR_TAIL:]}"
            )
        # The worker owns stdout, so the reply is the last line it wrote. Reading
        # it that way tolerates a stray write from a compiled kernel while still
        # reporting a reply that no line can parse.
        try:
            return result_payload(result.stdout)
        except ValueError as error:
            raise WorkerError(
                "evaluator worker did not emit JSON; "
                f"stdout leads with {result.stdout.strip()[:200]!r}; see {stdout_path}"
            ) from error


def kernel_backend(kernel: Kernel) -> Backend:
    return get_backend(kernel.problem.language, kernel.problem.platform)


def require_profile_backend(backend: Backend):
    if backend.kind is not BackendKind.CUDA:
        raise ActionError(
            f"{backend.kind.value} hardware-counter profiling is unsupported"
        )


def text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def now() -> str:
    return datetime.now(UTC).isoformat()


def inspect_problem(problem: Path, work: Path) -> ProblemSpec:
    """Inspect a Trace definition or Python reference on the available backend.

    Return its ProblemSpec with ordered tensor signatures, oracle, and workload.
    Write inspection evidence and captured inputs under work/evaluations.
    Preserve workload input files referenced by the returned contract.
    """
    description = EvaluationRuntime().inspect(
        problem, log_dir=work / EVALUATIONS_DIRECTORY
    )
    return ProblemSpec(
        name=description["problem_name"],
        definition=description["definition"],
        workload=description["workload"],
        language=description["language"],
        platform=description["platform"],
    )


def next_version(work: Path) -> int:
    """The number a kernel snapshot taken now would take.

    Versions count evaluations: the first snapshot is 1, and the highest number
    already present sets the next, so a directory left by an earlier run is not
    reused and a gap is filled rather than skipped.
    """

    versions = work / VERSIONS_DIRECTORY
    highest = 0
    if versions.is_dir():
        for entry in versions.iterdir():
            if entry.is_dir() and entry.name.isdigit():
                highest = max(highest, int(entry.name))
    return highest + 1


def snapshot_kernel(kernel: Kernel, work: Path) -> Path:
    """Save one numbered copy of a kernel under work/.klineage/versions.

    Evaluating a candidate is what advances the record, so the copy is taken
    before the run: the version always names the sources that were measured,
    whether the measurement then passes or fails. Snapshots are evidence, so a
    failure to write one must not fail the evaluation it describes.
    """

    version = work / VERSIONS_DIRECTORY / str(next_version(work))
    try:
        version.mkdir(parents=True, exist_ok=True)
        save_kernel(kernel, version)
    except OSError:
        pass
    return version


@agent_function
@observed("evaluate")
def evaluate(
    kernel: Kernel,
    work: Path,
    *,
    reference: Kernel | None = None,
    include_paths: Sequence[Path] = (),
    timeout: float | None = None,
) -> ValidationResult:
    """Build frozen Kernel sources, check correctness, and measure device latency.

    Write build artifacts and evidence under work/build and work/evaluations.
    The problem oracle checks correctness; optional reference supplies only the
    performance baseline and must share the problem contract. include_paths
    supplies build headers. Return ValidationResult; accepted requires every gate.
    Rebuild Kernel.source_files after edits. Omit reference for standalone checks.
    timeout overrides the worker deadline in seconds; it changes neither the
    sampling policy nor the enclosing action deadline. None uses the default.
    Each call also saves the kernel under work/.klineage/versions/<N>, numbered
    by call order.
    """
    snapshot_kernel(kernel, work)
    runtime = EvaluationRuntime(timeout=_TIMEOUT_SECONDS if timeout is None else timeout)
    return runtime.evaluate(
        kernel,
        reference=reference,
        include_paths=include_paths,
        build_root=work / BUILD_DIRECTORY,
        log_dir=work / EVALUATIONS_DIRECTORY / operation_id(RunKind.EVALUATE),
    )


def capture(kernel: Kernel, work: Path, options: ProfileOptions) -> dict[str, Any]:
    return EvaluationRuntime().profile(
        kernel,
        options=options,
        build_root=work / BUILD_DIRECTORY,
        log_dir=work / EVALUATIONS_DIRECTORY,
    )


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    build_root: Path
    include_paths: tuple[Path, ...]
    timing: TimingPolicy
    seed: int


def inspect_reference(
    problem_path: str | Path,
    *,
    workload_path: Path,
    seed: int = _DEFAULT_SEED,
    backend: Backend | None = None,
) -> dict[str, Any]:
    """Load a Trace definition or inspect a Python reference."""

    backend = backend or detect_backend()
    backend.torch()
    module, path = load_problem_module(problem_path)
    inputs = load_inputs(module, seed, backend)
    expected = compute_reference(module, inputs)

    name = problem_name(module, path)
    operator = getattr(module, "OPERATOR", name)
    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("problem OPERATOR must be a non-empty string")
    if path.suffix == ".json":
        platform = backend.platform()
        problem = ProblemSpec(
            name, module.definition, module.workload, backend.language, platform
        )
        check_problem(problem, inputs, expected)
        return {
            "problem_name": name,
            "operator": operator.strip(),
            "definition": module.definition,
            "workload": module.workload,
            "platform": platform,
            "language": backend.language,
        }
    return {
        "problem_name": name,
        "operator": operator.strip(),
        "definition": trace_definition(
            name,
            operator.strip(),
            path.read_text(encoding="utf-8"),
            inputs,
            dict(
                zip(
                    output_names(expected),
                    output_tensors(expected, backend),
                    strict=True,
                )
            ),
        ),
        "workload": trace_workload(inputs, workload_path),
        "platform": backend.platform(),
        "language": backend.language,
    }


def evaluate_request(request: Mapping[str, Any]) -> ValidationResult:
    """Evaluate one serialized candidate request."""
    from klineage.artifact.kernel import Kernel

    kernel_value = require_mapping(request.get("kernel"), "kernel")
    reference_value = request.get("reference")
    if reference_value is not None:
        reference_value = require_mapping(reference_value, "reference")
    kernel = Kernel.from_dict(kernel_value)
    backend = kernel_backend(kernel)
    backend.runtime()
    config = parse_config(require_mapping(request.get("config"), "config"), backend)
    reference = (
        Kernel.from_dict(reference_value) if reference_value is not None else None
    )
    if reference is not None:
        require_compatible_contracts(kernel, reference)
    module = trace_module(kernel.problem.definition, kernel.problem.workload)
    problem_inputs = load_inputs(module, config.seed, backend)
    expected = compute_reference(module, problem_inputs)
    check_problem(kernel.problem, problem_inputs, expected)

    strides = tensor_strides(problem_inputs, expected, kernel.problem)
    try:
        kernel.build(
            config.build_root, include_paths=config.include_paths, strides=strides
        )
        if reference is not None:
            reference.build(
                config.build_root, include_paths=config.include_paths, strides=strides
            )
    except Exception as error:  # noqa: BLE001 - Record builds; propagate interrupts.
        print(f"Build failed: {error}", file=sys.stderr)
        return ValidationResult(False, False, False)
    evaluator = CallableKernelEvaluator(
        problem_runtime(module, config.seed, backend),
        timer=make_timer(backend, config.timing),
    )
    return evaluator.evaluate(kernel, reference=reference)


def profile_request(request: Mapping[str, Any]) -> dict[str, Any]:
    from klineage.artifact.kernel import Kernel

    kernel = Kernel.from_dict(require_mapping(request.get("kernel"), "kernel"))
    backend = kernel_backend(kernel)
    require_profile_backend(backend)
    config = parse_config(require_mapping(request.get("config"), "config"))
    module = trace_module(kernel.problem.definition, kernel.problem.workload)
    inputs = load_inputs(module, config.seed, backend)
    device = next(iter(inputs.values())).device
    expected = compute_reference(module, inputs)
    kernel.build(
        config.build_root,
        include_paths=config.include_paths,
        strides=tensor_strides(inputs, expected, kernel.problem),
    )
    return profile_launch(kernel, args=tuple(inputs.values()), device=device)


def problem_runtime(
    module: ModuleType, seed: int, backend: Backend | None = None
) -> ProblemRuntime:
    inputs = {}

    def make_inputs() -> CallInputs:
        nonlocal inputs
        inputs = load_inputs(module, seed, backend)
        return CallInputs(args=tuple(inputs.values()))

    def reference(*args: torch.Tensor):
        return module.torch_ref(*args)

    def check_outputs(actual: Any, expected: Any):
        if getattr(module, "OPERATOR", None) == "topk":
            check_topk(actual, expected, inputs["values"])
            return
        check_close(actual, expected)
        check = getattr(module, "check_outputs", None)
        if check is not None:
            return check(actual, expected)

    return ProblemRuntime(
        make_inputs=make_inputs,
        reference=reference,
        check_outputs=check_outputs,
    )


def check_close(actual: Any, expected: Any):
    """Check tensor metadata before numerical comparison; never broadcast outputs."""
    import torch

    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ValueError("Output structure differs from the reference")
        for tensor, oracle in zip(actual, expected, strict=True):
            check_close(tensor, oracle)
        return
    if actual is None and expected is None:
        return
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError("Outputs must be tensors")
    if (actual.shape, actual.dtype, actual.device) != (
        expected.shape,
        expected.dtype,
        expected.device,
    ):
        raise ValueError("Output shape, dtype or device differs from the reference")
    if not torch.allclose(
        actual, expected, rtol=NUMERICAL_TOLERANCE, atol=NUMERICAL_TOLERANCE
    ):
        raise AssertionError(
            f"torch.allclose failed: rtol=atol={NUMERICAL_TOLERANCE:g}"
        )


def check_topk(actual: Any, expected: Any, inputs: torch.Tensor):
    import torch

    if not isinstance(actual, Sequence) or len(actual) != 2:
        raise ValueError("Top-K must return values and indices")
    values, indices = actual
    for tensor, oracle in zip(actual, expected, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Top-K outputs must be tensors")
        if (tensor.shape, tensor.dtype, tensor.device) != (
            oracle.shape,
            oracle.dtype,
            oracle.device,
        ):
            raise ValueError(
                "Top-K output shape, dtype or device differs from the reference"
            )
    if ((indices < 0) | (indices >= inputs.shape[-1])).any():
        raise ValueError("Top-K indices are out of bounds")
    ordered = indices.sort(dim=-1).values
    if (ordered[..., 1:] == ordered[..., :-1]).any():
        raise ValueError("Top-K contains duplicate indices")
    # Selection order and tied indices are unspecified; gathered values must agree.
    check_close(values, inputs.gather(-1, indices))
    check_close(values.sort(dim=-1).values, expected[0].sort(dim=-1).values)


def load_inputs(
    module: ModuleType, seed: int, backend: Backend | None = None
) -> Mapping[str, torch.Tensor]:
    backend = backend or detect_backend()
    if hasattr(module, "definition"):
        raw = trace_inputs(
            module.definition, module.workload, seed=seed, device=backend.device()
        )
    else:
        parameters = inspect.signature(module.make_inputs).parameters
        arguments = {}
        if "seed" in parameters:
            arguments["seed"] = seed
        if "device" in parameters:
            arguments["device"] = backend.device()
        raw = module.make_inputs(**arguments)
    if not isinstance(raw, Mapping):
        raise TypeError("problem.make_inputs() must return a mapping")
    values = dict(raw)
    if not values:
        raise ValueError("problem.make_inputs() must return at least one tensor")
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("problem input names must be non-empty strings")
        require_tensor(value, f"problem input {name!r}", backend)
    return values


def compute_reference(module: ModuleType, inputs: Mapping[str, torch.Tensor]) -> Any:
    import torch

    with torch.inference_mode():
        return module.torch_ref(*inputs.values())


def output_tensors(
    value: Any, backend: Backend | None = None
) -> tuple[torch.Tensor, ...]:
    import torch

    if not isinstance(value, (torch.Tensor, Sequence)) or isinstance(
        value, (str, bytes)
    ):
        raise TypeError("problem.torch_ref() must return a tensor or tensor sequence")
    outputs = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not outputs:
        raise ValueError("problem.torch_ref() must return at least one tensor")
    for index, output in enumerate(outputs):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"problem output {index} must be a tensor")
        if backend is not None:
            require_tensor(output, f"problem output {index}", backend)
    return outputs


def output_names(value: Any) -> tuple[str, ...]:
    import torch

    if isinstance(value, torch.Tensor):
        return ("output",)
    fields = getattr(value, "_fields", None)
    if isinstance(fields, tuple) and all(isinstance(field, str) for field in fields):
        return fields
    return tuple(f"output_{index}" for index in range(len(value)))


def tensor_strides(
    inputs: Mapping[str, torch.Tensor],
    output: Any,
    problem: ProblemSpec,
) -> dict[tuple[str, str], tuple[int, ...]]:
    result = {("input", name): tuple(value.stride()) for name, value in inputs.items()}
    result.update(
        {
            ("output", spec.name): tuple(value.stride())
            for spec, value in zip(
                problem.values(ValueRole.OUTPUTS), output_tensors(output), strict=True
            )
        }
    )
    return result


def check_problem(
    problem: ProblemSpec,
    inputs: Mapping[str, torch.Tensor],
    output: Any,
):
    backend = get_backend(problem.language, problem.platform)
    if tuple(inputs) != tuple(problem.definition["inputs"]):
        raise ValueError("problem input names or order do not match definition")
    for role, tensors in (
        (ValueRole.INPUTS, tuple(inputs.values())),
        (ValueRole.OUTPUTS, output_tensors(output, backend)),
    ):
        values = tensor_values(problem, role)
        if len(values) != len(tensors):
            raise ValueError(f"problem {role} count does not match definition")
        for spec, tensor in zip(values, tensors, strict=True):
            if (
                tensor.dtype != tensor_dtype(spec.dtype)
                or tuple(tensor.shape) != spec.shape
            ):
                raise ValueError(
                    f"problem {role} {spec.name!r} does not match definition"
                )


def problem_name(module: ModuleType, path: Path) -> str:
    declared = getattr(module, "PROBLEM_NAME", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return path.parent.name or path.stem


def parse_config(
    value: Mapping[str, Any], backend: Backend | None = None
) -> WorkerConfig:
    if "build_root" not in value:
        raise ValueError("config.build_root is required")
    correctness = {"rtol": NUMERICAL_TOLERANCE, "atol": NUMERICAL_TOLERANCE}
    if value.get("correctness", correctness) != correctness:
        raise ValueError("config.correctness differs from the evaluator tolerances")
    build_root = existing_dir(value["build_root"], "config.build_root")
    include_value = value.get("include_paths", ())
    if not isinstance(include_value, Sequence) or isinstance(
        include_value, (str, bytes)
    ):
        raise TypeError("config.include_paths must be an array")
    include_paths = tuple(existing_dir(item, "include path") for item in include_value)
    timing_value = value.get("timing", {})
    if not isinstance(timing_value, Mapping):
        raise TypeError("config.timing must be an object")
    defaults = timing_policy(backend).to_dict() if backend else TimingPolicy().to_dict()
    timing = TimingPolicy(**{**defaults, **timing_value})
    seed = value.get("seed", _DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("config.seed must be an integer")
    return WorkerConfig(
        build_root=build_root,
        include_paths=include_paths,
        timing=timing,
        seed=seed,
    )


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def request_operation(request: Mapping[str, Any]) -> str:
    value = request.get("operation")
    if value not in {"inspect", "evaluate", "profile"}:
        raise ValueError("operation must be 'inspect', 'evaluate' or 'profile'")
    return value


def result_payload(stdout: str) -> Any:
    """Parse the worker's reply from its stdout.

    The reply is the last line written. Anything ahead of it came from the code
    under evaluation, whose output the worker cannot fully capture. Raise
    ValueError when no trailing line parses, so the caller can report the head.
    """

    for line in reversed(stdout.splitlines()):
        if line.strip():
            try:
                return json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("no JSON reply") from error
    raise ValueError("empty worker stdout")


@contextlib.contextmanager
def _descriptor_stdout(stream):
    """Point descriptor 1 at `stream` for the block, then put it back.

    `contextlib.redirect_stdout` only rebinds `sys.stdout`, so output written
    through a file descriptor ignores it. The block restores the descriptor even
    when the body raises, which is what lets `main` still write its reply.
    """

    saved = os.dup(1)
    try:
        os.dup2(stream.fileno(), 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


def _emit(result: Any, *, sort_keys: bool = False) -> None:
    """Write the result frame as the final line of stdout.

    Insertion order is preserved by default: a definition's input order is part
    of its contract, and sorting it would silently reorder the ABI.
    """

    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=sort_keys)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    operation = ""
    # The result frame is this process's only stdout writer. Evaluating a kernel
    # runs code it compiled, and output from that code reaches descriptor 1
    # whatever the interpreter's stdout object is, so the descriptor goes to
    # stderr for the duration. `sys.stdout` is restored before the reply, which
    # therefore still lands on the descriptor the caller reads.
    try:
        request = require_mapping(json.load(sys.stdin), "worker request")
        operation = request_operation(request)
        with _descriptor_stdout(sys.stderr), contextlib.redirect_stdout(sys.stderr):
            if operation == "inspect":
                seed = request.get("seed", _DEFAULT_SEED)
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise TypeError("seed must be an integer")
                result: Any = inspect_reference(
                    request["problem_path"],
                    workload_path=Path(request["workload_path"]),
                    seed=seed,
                )
            elif operation == "profile":
                result = profile_request(request)
            else:
                result = evaluate_request(request).to_dict()
    except Exception as exc:  # noqa: BLE001 - worker failures cross a JSON boundary
        message = str(exc).strip() or type(exc).__name__
        print(f"{type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        if operation == "evaluate":
            result = ValidationResult(False, False, False).to_dict()
        else:
            _emit({"error_type": type(exc).__name__, "error": message}, sort_keys=True)
            return 1
    _emit(result)
    return 0


__all__ = [
    "CallInputs",
    "CallableKernelEvaluator",
    "KernelEvaluator",
    "ProblemRuntime",
    "ValidationResult",
    "capture",
    "evaluate",
    "inspect_problem",
]
