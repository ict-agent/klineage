"""Kernel correctness, device timing, and isolated runtime requests."""

from __future__ import annotations

import contextlib
import copy
import ctypes
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from klineage.artifact.kernel import ValidationResult
from klineage.artifact.problem import (
    Problem,
    load_problem,
    trace_inputs,
    trace_module,
)
from klineage.backend import Backend, detect_backend, get_backend
from klineage.backend.runtime import (
    current_device,
    device_runtime,
    require_tensor,
    tensor_dtype,
    tensor_values,
)
from klineage.backend.timing import bench
from klineage.constants import (
    NUMERICAL_TOLERANCE,
)
from klineage.harness.logging import now
from klineage.harness.process import run_process
from klineage.harness.timing import (
    TimingPolicy,
    timing_evidence,
    verify_performance,
)
from klineage.serialization import restore, to_json
from klineage.utils import (
    existing_dir,
    make_log_dir,
    mapping,
    resolve_path,
    write_json,
)

if TYPE_CHECKING:
    import torch

    from klineage.artifact.kernel import Kernel


WORKER_ENTRY = "from klineage.harness.eval import main; raise SystemExit(main())"
# Default evaluation deadline; enclosing action deadlines are independent.
_TIMEOUT_SECONDS = 900.0
_SEED = 20260903
_MAX_JOBS = 4
_ERROR_TAIL = 2000


@dataclass(frozen=True, slots=True)
class CallInputs:
    """Positional arguments for one kernel invocation.

    Attributes:
        args: Input values in the problem contract's declared order.
    """

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

    Attributes:
        make_inputs: Create positional inputs for each correctness or timing phase.
        reference: Compute expected outputs from the problem's input arguments.
        check_outputs: Compare candidate and oracle outputs.
    """

    make_inputs: Callable[[], CallInputs]
    reference: Callable[..., Any]
    check_outputs: Callable[[Any, Any], bool | None]

    def __post_init__(self):
        for name in ("make_inputs", "reference", "check_outputs"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


class CallableKernelEvaluator:
    """Check a built kernel against the problem oracle, then measure it.

    This evaluator runs in the calling process and does not compile kernels or
    create a sandbox.

    Args:
        runtime: Problem-owned input, oracle and comparison callbacks.
        policy: Timing policy; `None` selects the detected backend's default.
    """

    def __init__(self, runtime: ProblemRuntime, *, policy: TimingPolicy | None = None):
        self.runtime = runtime
        self.policy = policy

    def evaluate(
        self, kernel: Kernel, *, reference: Kernel | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> ValidationResult:
        """Record build readiness, correctness and timing gates.

        Args:
            kernel: Candidate with its callable already built.
            reference: Optional built kernel used only as a timing baseline.
            workload: Selected workload binding the kernel's tensor contract.

        Returns:
            Gate results and measured latency. Ordinary evaluation failures are
            recorded as failed gates; interrupts propagate.
        """
        try:
            problem = load_problem(kernel)
            if workload is None:
                if len(problem.workloads) != 1:
                    raise ValueError("callable evaluation requires one selected workload")
                workload = problem.workloads[0]
            inputs = problem.values("inputs", workload)
            outputs = problem.values("outputs", workload)
            if reference is not None:
                require_compatible_contracts(kernel, reference, workload)
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
            timing = bench(kernel, args=clone_inputs(original).args, policy=self.policy)
            timing_evidence(asdict(timing), timing.median_ms)
            print("Timing evidence: " + to_json(timing, indent=None), file=sys.stderr)
            reference_latency_ms = None
            passed = True
            if reference is not None:
                baseline = bench(
                    reference, args=clone_inputs(original).args, policy=self.policy
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


def require_compatible_contracts(
    candidate: Kernel, reference: Kernel, workload: Mapping[str, Any] | None = None,
):
    candidate_problem, reference_problem = load_problem(candidate), load_problem(reference)
    for role in ("inputs", "outputs"):
        if tuple(candidate_problem.values(role, workload).items()) != tuple(
            reference_problem.values(role, workload).items()
        ):
            raise ValueError("candidate and reference signatures do not match")
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
            (size - 1) * step for size, step in zip(value.shape, value.stride(), strict=True)
        )
    storage = value.as_strided((span,), (1,)).clone()
    return storage.as_strided(value.shape, value.stride())


class WorkerError(RuntimeError):
    """A local evaluator worker failed."""


class EvaluationRuntime:
    """Run metadata queries and evaluation in Python subprocesses.

    Each request records its input, stdout, stderr and process status. The base
    runtime does not sandbox workers; `WorkerRuntime` adds the server's process
    boundary and supports only evaluation and metadata queries.

    Args:
        timeout: Total evaluation budget in seconds, or the deadline for one
            metadata request.
        python: Interpreter executable used for every worker.
        environment: Complete worker environment, or `None` to inherit the host
            environment. The runtime adds its Python path and default build limit.
        worker_plugins: Modules imported in the controller and each worker to
            register additional artifact or backend implementations.
    """

    def __init__(
        self, *, timeout: float = _TIMEOUT_SECONDS,
        python: str | Path = sys.executable,
        environment: Mapping[str, str] | None = None,
        worker_plugins: Sequence[str] = (),
    ):
        self.timeout = timeout
        self.python = str(python)
        self.environment = dict(environment) if environment is not None else None
        self.worker_plugins = tuple(worker_plugins)
        for name in self.worker_plugins:
            importlib.import_module(name)

    def info(self, backend: Backend, *, build_root: Path, log_dir: Path) -> dict[str, Any]:
        """Inspect device and toolchain metadata in the configured worker runtime."""
        build = make_log_dir(build_root)
        logs = make_log_dir(log_dir)
        request = {
            "operation": "info", "backend": backend.language,
            "config": {"build_root": str(resolve_path(build)), "include_paths": []},
        }
        payload = self.invoke(request, logs, "info")
        if not isinstance(payload, Mapping) or payload.get("backend") != str(backend.kind):
            raise WorkerError("worker info returned a different backend")
        device = payload.get("device")
        if not isinstance(device, Mapping) or type(device.get("available")) is not bool:
            raise WorkerError("worker info requires device availability")
        write_json(logs / "info-result.json", payload)
        return dict(device)


    def evaluate(
        self,
        kernel: Kernel,
        *,
        include_paths: Sequence[Path],
        build_root: Path,
        log_dir: Path,
        reference: Kernel | None = None,
        timeout: float | None = None,
    ) -> ValidationResult:
        """Evaluate every workload in a separate worker under one deadline.

        Args:
            kernel: Candidate artifact; an existing host callable is not reused.
            include_paths: Additional compiler include directories.
            build_root: Worker build directory.
            log_dir: Directory for request, process and result records.
            reference: Optional artifact used only as a performance baseline.
            timeout: Optional total budget for this call, capped by the runtime's
                configured timeout. Does not change later calls.

        Returns:
            Aggregate gates and each workload's result. Worker failures and
            expired workloads remain explicit failed entries; no workloads fails.
        """

        budget = self.timeout if timeout is None else min(timeout, self.timeout)
        deadline = time.monotonic() + budget
        includes = tuple(resolve_path(path) for path in include_paths)
        host_build_root = Path(build_root).expanduser().absolute()
        host_build_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        build = resolve_path(host_build_root)

        problem = load_problem(kernel)
        backend = get_backend(problem.language, problem.platform)
        config = {
            "seed": _SEED,
            "timing": asdict(backend.timing.DEFAULT_POLICY),
            "correctness": {"rtol": NUMERICAL_TOLERANCE, "atol": NUMERICAL_TOLERANCE},
            "include_paths": [str(path) for path in includes],
        }
        logs = make_log_dir(log_dir)
        workloads = problem.workloads
        results = {}
        if not workloads:
            (logs / "error.log").write_text("problem has no workloads to evaluate\n")
        for index, workload in enumerate(workloads, start=1):
            # Isolate each build and its evidence; UUIDs are data, never paths.
            case = f"workload-{index:04d}"
            case_logs = make_log_dir(logs / case)
            case_build = make_log_dir(build / case / "build")
            stem = "eval-0001"
            request = {
                "operation": "evaluate", "workload_id": workload["uuid"],
                "kernel": kernel,
                "reference": reference,
                "config": {**config, "build_root": str(case_build)},
            }
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    write_json(case_logs / f"{stem}-request.json", request)
                    raise WorkerError("evaluation deadline expired before workload")
                payload = self.invoke(request, case_logs, stem, timeout=remaining)
                if time.monotonic() >= deadline:
                    raise WorkerError("evaluation deadline expired during workload")
                if not isinstance(payload, Mapping):
                    raise WorkerError("evaluator worker result must be a JSON object")
                result = restore(payload, ValidationResult)
                if result.workloads:
                    raise WorkerError("workload worker must return a leaf result")
            except Exception as error:  # noqa: BLE001 - Retain completed cases.
                (case_logs / "error.log").write_text(f"{type(error).__name__}: {error}\n")
                result = ValidationResult(False, False, False)
            results[workload["uuid"]] = result
            write_json(case_logs / f"{stem}-result.json", result)

        result = ValidationResult.combine(results) if results else ValidationResult(False, False, False)
        write_json(logs / "result.json", result)
        return result

    def invoke(
        self,
        request: Mapping[str, Any],
        log_dir: Path,
        stem: str,
        *,
        timeout: float | None = None,
    ) -> Any:
        timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        write_json(log_dir / f"{stem}-request.json", request)
        stdout_path = log_dir / f"{stem}-stdout.log"
        stderr_path = log_dir / f"{stem}-stderr.log"
        process_path = log_dir / f"{stem}-process.json"
        entry = WORKER_ENTRY
        if self.worker_plugins:
            imports = "\n".join(
                f"importlib.import_module({name!r})" for name in self.worker_plugins
            )
            entry = f"import importlib\n{imports}\n{entry}"
        command = (self.python, "-c", entry)
        environment = dict(os.environ if self.environment is None else self.environment)
        environment.setdefault("MAX_JOBS", str(_MAX_JOBS))
        source_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_root, environment.get("PYTHONPATH")) if value
        )
        started = now()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            timeout = remaining
            result = self.launch(command, request, timeout, environment)
        except subprocess.TimeoutExpired as error:
            stdout_path.write_text(text(error.stdout), encoding="utf-8")
            stderr_path.write_text(text(error.stderr), encoding="utf-8")
            write_json(
                process_path,
                {
                    "command": error.cmd or command,
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
                "command": result.args or command,
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

    def launch(self, command, request, timeout, environment):
        """Start one worker; server runtimes override the process boundary."""
        deadline = time.monotonic() + timeout
        payload = to_json(request, indent=None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        return run_process(
            command, payload=payload, timeout=remaining,
            environment=environment,
        )


def text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    build_root: Path
    include_paths: tuple[Path, ...]
    timing: TimingPolicy
    seed: int


def evaluate_request(request: Mapping[str, Any]) -> ValidationResult:
    """Evaluate one selected workload and return its leaf result."""
    from klineage.artifact.kernel import Kernel

    kernel_payload = mapping(request.get("kernel"), "kernel")
    reference_value = request.get("reference")
    if reference_value is not None:
        reference_value = mapping(reference_value, "reference")
    kernel = restore(kernel_payload, Kernel)
    problem = load_problem(kernel)
    backend = get_backend(problem.language, problem.platform)
    device_runtime(backend)
    config = parse_config(mapping(request.get("config"), "config"), backend)
    reference = (
        restore(reference_value, Kernel) if reference_value is not None else None
    )
    selected = problem.resolve_inputs(request.get("workload_id"))
    if len(selected) != 1:
        raise ValueError("worker evaluation requires one selected workload")
    workload = selected[0]
    if reference is not None:
        require_compatible_contracts(kernel, reference, workload)
    module = trace_module(problem.definition, selected)
    problem_inputs = load_inputs(module, config.seed, backend, workload=workload)
    expected = compute_reference(module, problem_inputs)
    check_problem(problem, problem_inputs, expected, workload)

    strides = tensor_strides(problem_inputs, expected, problem, workload)
    try:
        object.__setattr__(
            kernel,
            "function",
            kernel.build_kernel(
                config.build_root,
                include_paths=config.include_paths,
                strides=strides,
                workload=workload,
            ),
        )
        if reference is not None:
            object.__setattr__(
                reference,
                "function",
                reference.build_kernel(
                    config.build_root,
                    include_paths=config.include_paths,
                    strides=strides,
                    workload=workload,
                ),
            )
    except Exception as error:  # noqa: BLE001 - Record builds; propagate interrupts.
        print(f"Build failed: {error}", file=sys.stderr)
        return ValidationResult(False, False, False)
    evaluator = CallableKernelEvaluator(
        problem_runtime(module, config.seed, backend, workload=workload),
        policy=config.timing,
    )
    return evaluator.evaluate(kernel, reference=reference, workload=workload)


def problem_runtime(
    module: ModuleType, seed: int, backend: Backend | None = None,
    *, workload: Mapping[str, Any] | None = None,
) -> ProblemRuntime:
    inputs = {}

    def make_inputs() -> CallInputs:
        nonlocal inputs
        inputs = load_inputs(module, seed, backend, workload=workload)
        return CallInputs(args=tuple(inputs.values()))

    def reference(*args: torch.Tensor):
        return module.run(*args)

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
    # Include mismatch counts, error magnitudes and indices in worker diagnostics.
    torch.testing.assert_close(
        actual,
        expected,
        rtol=NUMERICAL_TOLERANCE,
        atol=NUMERICAL_TOLERANCE,
        check_stride=False,
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
    module: ModuleType, seed: int, backend: Backend | None = None,
    *, workload: Mapping[str, Any] | None = None,
) -> Mapping[str, torch.Tensor]:
    backend = backend or detect_backend()
    if workload is None:
        if len(module.workloads) != 1:
            raise ValueError("trace inputs require one selected workload")
        workload = module.workloads[0]
    raw = trace_inputs(
        module.definition, workload, seed=seed, device=current_device(backend)
    )
    if not isinstance(raw, Mapping):
        raise TypeError("Trace input loader must return a mapping")
    values = dict(raw)
    if not values:
        raise ValueError("Trace input loader must return at least one tensor")
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("problem input names must be non-empty strings")
        require_tensor(value, f"problem input {name!r}", backend)
    return values


def compute_reference(module: ModuleType, inputs: Mapping[str, torch.Tensor]) -> Any:
    import torch

    with torch.inference_mode():
        return module.run(*inputs.values())


def output_tensors(
    value: Any, backend: Backend | None = None
) -> tuple[torch.Tensor, ...]:
    import torch

    if not isinstance(value, (torch.Tensor, Sequence)) or isinstance(
        value, (str, bytes)
    ):
        raise TypeError("reference run() must return a tensor or tensor sequence")
    outputs = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not outputs:
        raise ValueError("reference run() must return at least one tensor")
    for index, output in enumerate(outputs):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"problem output {index} must be a tensor")
        if backend is not None:
            require_tensor(output, f"problem output {index}", backend)
    return outputs


def tensor_strides(
    inputs: Mapping[str, torch.Tensor],
    output: Any,
    problem: Problem,
    workload: Mapping[str, Any] | None = None,
) -> dict[tuple[str, str], tuple[int, ...]]:
    result = {("input", name): tuple(value.stride()) for name, value in inputs.items()}
    result.update(
        {
            ("output", name): tuple(value.stride())
            for name, value in zip(
                problem.values("outputs", workload), output_tensors(output), strict=True
            )
        }
    )
    return result


def check_problem(
    problem: Problem,
    inputs: Mapping[str, torch.Tensor],
    output: Any,
    workload: Mapping[str, Any] | None = None,
):
    backend = get_backend(problem.language, problem.platform)
    if tuple(inputs) != tuple(problem.definition["inputs"]):
        raise ValueError("problem input names or order do not match definition")
    for role, tensors in (
        ("inputs", tuple(inputs.values())),
        ("outputs", output_tensors(output, backend)),
    ):
        values = tensor_values(problem, role, workload)
        if len(values) != len(tensors):
            raise ValueError(f"problem {role} count does not match definition")
        for (name, spec), tensor in zip(values.items(), tensors, strict=True):
            if (
                tensor.dtype != tensor_dtype(spec["dtype"])
                or tuple(tensor.shape) != tuple(spec["shape"])
            ):
                raise ValueError(
                    f"problem {role} {name!r} does not match definition"
                )


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
    defaults = asdict(backend.timing.DEFAULT_POLICY if backend else TimingPolicy())
    timing = TimingPolicy(**{**defaults, **timing_value})
    seed = value.get("seed", _SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("config.seed must be an integer")
    return WorkerConfig(
        build_root=build_root,
        include_paths=include_paths,
        timing=timing,
        seed=seed,
    )


def request_operation(request: Mapping[str, Any]) -> str:
    value = request.get("operation")
    if value not in {"info", "evaluate"}:
        raise ValueError("operation must be 'info' or 'evaluate'")
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
        # Flush native diagnostics before restoring the JSON reply descriptor.
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


def _emit(result: Any) -> None:
    """Write the result frame as the final line of stdout.

    Insertion order is preserved by default: a definition's input order is part
    of its contract, and sorting it would silently reorder the ABI.
    """

    sys.stdout.write(to_json(result, indent=None) + "\n")
    sys.stdout.flush()
    # Runtime destructors can still write after Python emits the final reply.
    os.dup2(sys.stderr.fileno(), 1)


def main() -> int:
    operation = ""
    # The result frame is this process's only stdout writer. Evaluating a kernel
    # runs code it compiled, and output from that code reaches descriptor 1
    # whatever the interpreter's stdout object is, so the descriptor goes to
    # stderr for the duration. `sys.stdout` is restored before the reply, which
    # therefore still lands on the descriptor the caller reads.
    try:
        request = mapping(json.load(sys.stdin), "worker request")
        operation = request_operation(request)
        with _descriptor_stdout(sys.stderr), contextlib.redirect_stdout(sys.stderr):
            if operation == "info":
                backend = get_backend(request["backend"])
                result: Any = {"backend": str(backend.kind), "device": backend.info()}
            else:
                result = evaluate_request(request)
    except Exception as exc:  # noqa: BLE001 - worker failures cross a JSON boundary
        message = str(exc).strip() or type(exc).__name__
        print(f"{type(exc).__name__}: {message}", file=sys.stderr, flush=True)
        if operation == "evaluate":
            result = ValidationResult(False, False, False)
        else:
            _emit({"error_type": type(exc).__name__, "error": message})
            return 1
    _emit(result)
    return 0


__all__ = [
    "CallInputs",
    "CallableKernelEvaluator",
    "ProblemRuntime",
    "ValidationResult",
]
