"""Backend benchmarking and device information."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

from .timing import TimingPolicy, TimingResult

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel


class BackendKind(StrEnum):
    """Identifiers for the built-in device backends."""

    CUDA = "cuda"
    HYGON = "hygon"
    ASCEND = "ascend"


class Backend(ABC):
    """Connect a kernel format to device discovery and latency measurement.

    Source compilation belongs to ``kernel_type.build_kernel``. Vendor runtime
    imports belong inside build, timing, and information calls so registry and
    artifact operations can run without device libraries.

    Attributes:
        kind: Unique backend identifier.
        language: Native source language used for target selection.
        device_type: Tensor runtime device name, such as ``cuda`` or ``npu``.
        platform_prefixes: Lowercase platform prefixes accepted by selection.
        kernel_type: Concrete Kernel defining the tree schema and builder.
        skill_root: Directory containing the backend's generation skills.
        timing: Module providing NAME, DEFAULT_POLICY, bench, and check_timing.
        device_patterns: Relative /dev globs for optional local device discovery.
    """

    kind: ClassVar[BackendKind]
    language: ClassVar[str]
    device_type: ClassVar[str]
    platform_prefixes: ClassVar[tuple[str, ...]]
    kernel_type: ClassVar[type[Kernel]]
    skill_root: ClassVar[Path]
    timing: ClassVar[ModuleType]
    device_patterns: ClassVar[tuple[str, ...]] = ()

    def device_access(self, nodes, environment):
        """Select sandbox device nodes and their runtime visibility."""
        return nodes, dict(environment)

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
