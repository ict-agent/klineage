"""Backend kernel layouts, compilation, timing and device information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from klineage.constants import BUILD_DIRECTORY

from .timing import TimingPolicy, TimingResult

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel
    from klineage.artifact.tree import Node


class BackendKind(StrEnum):
    """Identifiers for the built-in device backends."""

    CUDA = "cuda"
    HYGON = "hygon"
    ASCEND = "ascend"


class Backend(ABC):
    """Connect a kernel format to device discovery and latency measurement.

    Layout and compilation belong to ``kernel_spec`` and ``build_kernel``.
    Vendor runtime imports belong inside build, timing, and information calls so registry and
    artifact operations can run without device libraries.

    Attributes:
        kind: Unique backend identifier.
        language: Native source language used for target selection.
        device_type: Tensor runtime device name, such as ``cuda`` or ``npu``.
        platform_prefixes: Lowercase platform prefixes accepted by selection.
        kernel_kind: Serialized kernel artifact identifier.
        entry_suffixes: Accepted native entry source suffixes.
        skill_root: Directory containing the backend's generation skills.
        timing: Module providing NAME, DEFAULT_POLICY, bench, and check_timing.
        device_patterns: Relative /dev globs for optional local device discovery.
    """

    kind: ClassVar[BackendKind]
    language: ClassVar[str]
    device_type: ClassVar[str]
    platform_prefixes: ClassVar[tuple[str, ...]]
    kernel_kind: ClassVar[str]
    entry_suffixes: ClassVar[tuple[str, ...]]
    skill_root: ClassVar[Path]
    timing: ClassVar[ModuleType]
    device_patterns: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def kernel_spec(self) -> Node:
        """Return the native kernel layout, constraints and examples."""
        raise NotImplementedError

    @abstractmethod
    def build_kernel(
        self,
        kernel: Kernel,
        build_root: Path = BUILD_DIRECTORY,
        *,
        include_paths: Sequence[Path] = (),
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
        workload: Mapping[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Build a kernel and return its entry callable without attaching it.

        Args:
            kernel: Source artifact for this backend.
            build_root: Directory for staged sources and build outputs.
            include_paths: External compiler include directories.
            strides: Tensor element strides keyed by ``(role, name)``.
            workload: Optional workload binding for the tensor ABI.

        Returns:
            A callable accepting the problem's ordered inputs.
        """
        raise NotImplementedError

    def device_access(self, nodes, environment):
        """Select sandbox device nodes and their runtime visibility."""
        return nodes, dict(environment)

    def agent_devices(self, nodes):
        """Allow direct diagnostics when agent and worker can share devices."""
        return nodes

    @abstractmethod
    def bench(
        self,
        fn: Kernel,
        *,
        args: tuple[Any, ...] = (),
        policy: TimingPolicy | None = None,
    ) -> TimingResult:
        """Measure a loaded kernel without compiling or checking correctness.

        Args:
            fn: Executable kernel for this device backend.
            args: Positional inputs already allocated on the target device.
            policy: Measurement settings; None selects the backend default.

        Returns:
            Millisecond samples, their reduction, and backend timing evidence.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Inspect device and runtime metadata in the calling process.

        Returns:
            Serializable metadata including ``available`` for detection and
            ``platform`` for problem inspection. Other fields describe the
            backend's device, architecture, runtime, and toolchain.

        Raises:
            ImportError: Required runtime libraries cannot be imported.
            RuntimeError: The installed runtime is incompatible.
        """
        raise NotImplementedError
