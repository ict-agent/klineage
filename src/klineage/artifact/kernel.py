"""Kernel artifact interface and measured validation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import stat
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from klineage.artifact.tree import (
    MAX_INLINE_BYTES,
    Artifact,
    ArtifactKind,
    FileRef,
    Node,
    NodeKind,
    read_source_tree,
)
from klineage.constants import (
    BUILD_DIRECTORY,
    BUNDLE_CONFIG,
    BUNDLE_SOLUTION,
    CURRENT_KERNEL_DIRECTORY,
    KERNEL_METADATA,
    STATE_DIRECTORY,
    SUBMISSION_DIRECTORY,
    VERSIONS_DIRECTORY,
)
from klineage.serialization import raw_bytes, raw_text, restore
from klineage.tools import agent_function
from klineage.utils import DirectoryMode, operation_id, safe_directory, safe_file

VALIDATION_GATES = ("compile_passed", "correctness_passed", "profile_passed")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Compilation, correctness and timing results attached to a kernel.

    Attributes:
        compile_passed: Whether the implementation built successfully.
        correctness_passed: Whether measured outputs passed correctness checks.
        profile_passed: Whether the timing/performance gate passed. This does
            not represent a vendor-profiler report.
        latency_ms: Positive finite candidate latency for a single workload.
            A multi-workload report leaves this unset.
        reference_latency_ms: Optional positive finite reference latency.
        workloads: Individual results keyed by workload UUID. Empty for a
            single worker's result; populated for a complete evaluation report.

    Constructing this value checks field types and latency ranges; it does not
    run evaluation or establish service measurements.
    """

    compile_passed: bool
    correctness_passed: bool
    profile_passed: bool
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    workloads: Mapping[str, ValidationResult] = field(default_factory=dict, hash=False)

    def __post_init__(self):
        for name in VALIDATION_GATES:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.workloads, Mapping):
            raise TypeError("workloads must map UUIDs to validation results")
        workloads = dict(self.workloads)
        for name, result in workloads.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("workload UUID must be nonempty text")
            if not isinstance(result, ValidationResult):
                raise TypeError("workload result must be a ValidationResult")
            if result.workloads:
                raise ValueError("nested workload reports are not allowed")
        object.__setattr__(self, "workloads", MappingProxyType(workloads))

        if workloads:
            for name in VALIDATION_GATES:
                passed = all(getattr(result, name) for result in workloads.values())
                if getattr(self, name) != passed:
                    raise ValueError(f"workload summary differs from {name} results")
            single = next(iter(workloads.values())) if len(workloads) == 1 else None
            for name in ("latency_ms", "reference_latency_ms"):
                expected = getattr(single, name) if single is not None else None
                if getattr(self, name) != expected:
                    raise ValueError(
                        f"workload summary {name} must match the single case or be unset"
                    )
        elif self.profile_passed and self.latency_ms is None:
            raise ValueError("successful timing requires latency_ms")
        for name in ("latency_ms", "reference_latency_ms"):
            latency = getattr(self, name)
            if latency is None:
                continue
            if type(latency) not in (int, float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def accepted(self) -> bool:
        """Whether all three validation gates passed."""
        return self.compile_passed and self.correctness_passed and self.profile_passed

    @classmethod
    def combine(cls, workloads: Mapping[str, ValidationResult]) -> ValidationResult:
        """Summarize nonempty workload results without averaging their latencies."""
        if not isinstance(workloads, Mapping):
            raise TypeError("workloads must map UUIDs to validation results")
        if not workloads:
            raise ValueError("cannot summarize an empty set of workload results")
        if any(not isinstance(value, cls) for value in workloads.values()):
            raise TypeError("workload result must be a ValidationResult")
        single = next(iter(workloads.values())) if len(workloads) == 1 else None
        return cls(
            **{
                name: all(getattr(value, name) for value in workloads.values())
                for name in VALIDATION_GATES
            },
            latency_ms=single.latency_ms if single is not None else None,
            reference_latency_ms=single.reference_latency_ms if single is not None else None,
            workloads=workloads,
        )


KERNEL = Node.directory(
    ".",
    description="\n".join(
        (
            "A kernel source tree with a checked reference to its problem.",
            "Kernel files live under build_root. The framework saves measured versions under versions/<N>/.",
            "The agent submits this directory with klineage.server.Client.evaluate; the service checks kernel_spec and builds it.",
            "After source edits reconstruct the Kernel before evaluating it.",
            "The client automatically saves returned Kernels under versions/<N>; the service owns validation.",
        )
    ),
    children=(
        Node.file(
            KERNEL_METADATA,
            description="\n".join(
                (
                    "Kernel name, problem directory reference {path, name, language, platform}, validation and optional compile_flags.",
                    "Compiled runtime handles are never serialized.",
                    "Validation stores compile_passed, correctness_passed, profile_passed and per-workload results keyed by UUID.",
                    "All workloads must pass. Each workload retains its finite positive latency in milliseconds; multi-workload summaries leave latency fields unset.",
                    "Attach only measured validation; clear it when the implementation changes.",
                    "Preserve native compile_flags through builds, comparisons and handoffs.",
                )
            ),
        ),
        Node.directory(
            "submission",
            description="\n".join(
                (
                    "All bundle paths and their exact source text.",
                    "config.toml and solution/ follow the selected bundle format.",
                    "The host entry source supplies bindings and all launch parameters.",
                )
            ),
        ),
    ),
)


class Kernel(Artifact, ABC):
    """A kernel source artifact that becomes callable after ``load_kernel``.

    Backend subclasses define ``kernel_spec`` and ``build_kernel``. The tree
    contains metadata and submission sources; ``metadata.json`` references the
    original problem directory and its execution target. Constructing or restoring
    a tree does not compile it. ``load_kernel`` builds the entry callable, whose
    source supplies bindings and device launch parameters.

    Args:
        tree: Kernel tree rooted at ``.`` and matching the backend's specification.

    Attributes:
        function (Callable[..., Any] | None): Built entry callable, or ``None``
            until loaded. Not serialized.
        validation (ValidationResult | None): Attached measurements, or ``None``
            if unmeasured.
    """

    __slots__ = ("function",)

    def __init__(self, tree: Node):
        Artifact.__init__(self, self.artifact_kind, tree)

    def __post_init__(self):
        errors = []
        for layout in kernel_layouts(type(self)):
            try:
                layout.check(self.tree)
            except ValueError as error:
                errors.append(str(error))
                continue
            break
        else:
            raise ValueError(f"{self.kind}: no kernel layout matches; " + "; ".join(dict.fromkeys(errors)))
        for name in (KERNEL_METADATA, SUBMISSION_DIRECTORY.name):
            if not isinstance(self.find(name), Node):
                raise ValueError(f"{name}: expected exactly one kernel node")
        Artifact.__post_init__(self)
        object.__setattr__(self, "function", None)

    @classmethod
    def from_tree(cls, kind: str, tree: Node):
        """Restore a concrete Kernel; the abstract interface uses known backends."""
        if cls is Kernel:
            from klineage.backend import backends

            for backend in backends():
                if backend.kernel_type.artifact_kind == kind:
                    return backend.kernel_type.from_tree(kind, tree)
            raise TypeError(f"{kind} is not a Kernel of a known backend")
        return super().from_tree(kind, tree)

    @classmethod
    @abstractmethod
    @agent_function
    def kernel_spec(cls) -> Node:
        """Return this backend's kernel file structure, constraints and examples.

        Subclasses implement this specification; construction and loading use
        it to reject malformed subtrees with path-specific errors.
        """
        raise NotImplementedError

    @classmethod
    @agent_function
    def load_kernel(
        cls, build_root: Path = CURRENT_KERNEL_DIRECTORY, *, include_paths: Sequence[Path] = ()
    ) -> Kernel:
        """Read a kernel directory, check its specification and build its callable.

        Args:
            build_root: Directory containing ``metadata.json`` and ``submission/``.
            include_paths: Additional compiler include directories. They remain
                external dependencies and are not added to the artifact.

        Returns:
            A backend-specific ``Kernel`` callable with the problem's ordered
            input ABI: call it as ``outputs = kernel(*inputs)``. Destination-passing
            outputs are allocated by the adapter. Calling ``Kernel.load_kernel`` selects the backend from
            the referenced problem; subclass calls require a matching target.

        Raises:
            NotADirectoryError: ``build_root`` is not a directory.
            ValueError: The tree, problem reference or target is invalid.
                Structural errors identify the failing subtree.

        Builds use ``workspace/.klineage/build`` beside ``build_root``. JSON representations
        are read separately with ``load_artifact`` and do not carry compiled code.
        """
        from klineage.artifact.problem import load_problem
        from klineage.backend import get_backend

        root = Path(build_root).expanduser().absolute()
        if not root.is_dir():
            raise NotADirectoryError(f"{root}: expected a Kernel directory")
        tree = capture_kernel(root)
        value = Artifact(ArtifactKind.KERNEL, tree) if cls is Kernel else cls(tree)
        problem = load_problem(value)
        try:
            backend = get_backend(problem.language, problem.platform)
        except ValueError as error:
            raise ValueError(f"{KERNEL_METADATA}/problem: {error}") from error
        if cls is Kernel:
            value = backend.kernel_type(tree)
        elif not isinstance(value, backend.kernel_type):
            raise ValueError(
                f"{KERNEL_METADATA}/problem: target differs from {cls.__name__}"
            )
        cache = root.parent / STATE_DIRECTORY / BUILD_DIRECTORY
        function = value.build_kernel(cache, include_paths=include_paths)
        object.__setattr__(value, "function", function)
        return value

    @abstractmethod
    def build_kernel(
        self,
        build_root: Path = BUILD_DIRECTORY,
        *,
        include_paths: Sequence[Path] = (),
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Build this tree and return its entry callable without attaching it.

        Args:
            build_root: Directory for staged sources and build outputs.
            include_paths: External compiler include directories.
            strides: Optional tensor element strides keyed by ``(role, name)``,
                with role ``input`` or ``output``, for the entry adapter.
            workload: Optional workload used to bind the entry's tensor ABI.
                Without a binding, dynamic axes are inferred from call inputs.

        Returns:
            An entry callable accepting the problem's ordered inputs. Subclasses
            own compilation and loading; ``load_kernel`` stores this callable.
        """
        raise NotImplementedError

    @property
    def validation(self):
        """Return stored measurements, or ``None`` when no validation is attached."""
        value = json.loads(raw_text(self.find(KERNEL_METADATA)))["validation"]
        return restore(value, ValidationResult) if value is not None else None

    def __call__(self, *inputs: Any) -> Any:
        """Run with ordered problem inputs and return outputs.

        Destination-passing outputs are allocated by the adapter. Raises
        ``RuntimeError`` if the kernel has not been built.
        """
        if self.function is None:
            raise RuntimeError("Kernel is not built; use its backend's load_kernel()")
        return self.function(*inputs)


def kernel_layouts(kernel_type: type[Kernel]) -> tuple[Node, ...]:
    """Define native, source-free and Python-entry layouts for a Kernel type."""
    native = kernel_type.kernel_spec()
    if not isinstance(native, Node):
        raise TypeError("kernel_spec must return a Node")
    python_entry = native.with_child(Node.directory(SUBMISSION_DIRECTORY.name, (
        Node.file(BUNDLE_CONFIG, description='Python host entry: language="python", entry_point="kernel.py::run".'),
        Node.directory(BUNDLE_SOLUTION, (Node.file("kernel.py", description="Python host entry."),)),
    )))
    return native, KERNEL, python_entry


def capture_kernel(path: Path) -> Node:
    """Read a kernel directory with inline metadata and bounded inline sources."""
    root = Path(path).expanduser().absolute()
    with safe_directory(root):
        pass
    sources = read_source_tree(root / SUBMISSION_DIRECTORY, "submission")

    def read(path: Path, relative: Path) -> Node:
        if path.is_symlink():
            raise ValueError(f"kernel cannot contain symlinks: {path}")
        if path.is_dir():
            with safe_directory(path) as directory:
                names = sorted(os.listdir(directory))
            return Node.directory(
                path.name if relative.parts else ".",
                (read(path / name, relative / name) for name in names),
            )
        with safe_file(path) as stream:
            data = stream.buffer.read()
        if relative == Path(KERNEL_METADATA):
            return Node.file(path.name, data.decode("utf-8"))
        if relative.is_relative_to(SUBMISSION_DIRECTORY):
            name = relative.relative_to(SUBMISSION_DIRECTORY).as_posix()
            content = data.decode("utf-8")
            if sources.get(name) != content:
                raise ValueError(f"submission changed while reading: {relative}")
            return Node.file(path.name, content)
        if len(data) <= MAX_INLINE_BYTES:
            try:
                content = data.decode("utf-8")
                if "\0" not in content:
                    return Node.file(path.name, content)
            except UnicodeDecodeError:
                pass
        return Node.file(path.name, reference=FileRef(
            str(path), hashlib.sha256(data).hexdigest(), len(data),
        ))

    return read(root, Path())


def latest_kernel(work: Path) -> Path:
    """Return the highest numbered kernel directory without following links."""
    root = Path(work).expanduser().absolute() / VERSIONS_DIRECTORY
    versions = []
    with safe_directory(root) as directory:
        for name in os.listdir(directory):
            if not name.isascii() or not name.isdecimal() or str(int(name)) != name:
                continue
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"kernel version must be a real directory: {root / name}")
            versions.append(int(name))
    if not versions:
        raise FileNotFoundError(f"no kernel versions in {root}")
    return root / str(max(versions))


def next_version(work: Path) -> int:
    """Number the next kernel snapshot from zero, without reusing gaps."""
    try:
        return int(latest_kernel(work).name) + 1
    except FileNotFoundError:
        return 0


def snapshot_kernel(kernel: Kernel, work: Path) -> Path:
    """Publish a complete kernel file tree under work/versions/<N>."""
    root = Path(work).expanduser().absolute() / VERSIONS_DIRECTORY
    with safe_directory(root, DirectoryMode.CREATE) as directory:
        # Serialize publishers; only complete trees receive a version number.
        fcntl.flock(directory, fcntl.LOCK_EX)
        number = str(next_version(work))
        temporary = operation_id("kernel")
        os.mkdir(temporary, mode=0o700, dir_fd=directory)
        try:
            pending = [(kernel.tree, root / temporary)]
            while pending:
                node, path = pending.pop()
                if node.kind is NodeKind.FILE:
                    with safe_file(path, "w") as stream:
                        stream.buffer.write(raw_bytes(node))
                    continue
                with safe_directory(path, DirectoryMode.CREATE):
                    pass
                pending.extend((child, path / child.name) for child in node.children)
            os.rename(temporary, number, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            with suppress(FileNotFoundError):
                shutil.rmtree(temporary, dir_fd=directory)
    return root / number
