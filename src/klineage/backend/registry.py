"""Backend registration and selection."""

import os
from inspect import isabstract

from .ascend import AscendBackend
from .cuda import CudaBackend
from .hip import HipBackend
from .kernel import Backend
from .timing import TimingPolicy

BACKEND_ENV = "KLINEAGE_BACKEND"
REGISTERED_BACKENDS = [CudaBackend(), HipBackend(), AscendBackend()]


def backends() -> tuple[Backend, ...]:
    """Return registered backend instances in automatic detection order."""
    return tuple(REGISTERED_BACKENDS)


def register_backend(backend: Backend) -> None:
    """Register a backend in the current process.

    Registration must also run in each worker and agent process that uses the
    format. It does not probe device availability or import vendor libraries.

    Args:
        backend: Driver with a concrete Kernel, schema, and timing provider.

    Raises:
        TypeError: The Kernel or timing provider lacks a required interface.
        ValueError: The backend kind, language, or artifact kind is registered.
    """
    from klineage.artifact.kernel import Kernel, kernel_layouts

    cls = getattr(backend, "kernel_type", None)
    if not isinstance(cls, type) or not issubclass(cls, Kernel) or isabstract(cls):
        raise TypeError("backend requires a concrete kernel.py implementation")
    kernel_layouts(cls)
    timing = getattr(backend, "timing", None)
    timing_name = getattr(timing, "NAME", None)
    if (
        not isinstance(timing_name, str)
        or not timing_name.strip()
        or not isinstance(getattr(timing, "DEFAULT_POLICY", None), TimingPolicy)
        or not callable(getattr(timing, "check_timing", None))
    ):
        raise TypeError("backend requires a timing.py provider")
    if any(
        item.kind == backend.kind or item.language == backend.language
        or item.kernel_type.artifact_kind == cls.artifact_kind
        for item in REGISTERED_BACKENDS
    ):
        raise ValueError(f"backend already registered: {backend.kind}")
    REGISTERED_BACKENDS.append(backend)


def platform_backend(platform: str) -> Backend | None:
    """Return the first backend matching a platform prefix, or None."""
    for backend in REGISTERED_BACKENDS:
        if platform.lower().startswith(backend.platform_prefixes):
            return backend
    return None


def get_timing_backend(name: str) -> Backend:
    """Resolve a timing provider NAME, raising ValueError if unregistered."""
    for backend in REGISTERED_BACKENDS:
        if name == backend.timing.NAME:
            return backend
    raise ValueError(f"unsupported timing backend: {name}")


def get_backend(language: str, platform: str = "") -> Backend:
    """Select a registered backend without probing device availability.

    Args:
        language: Native source language, or ``python`` for a Python host entry.
        platform: Platform prefix. Required for Python entries; for native
            languages, a recognized prefix must agree with the language.

    Returns:
        The registered backend instance for the target.

    Raises:
        ValueError: The language is unsupported, a Python target is missing or
            unsupported, or a recognized platform conflicts with the language.
    """
    target = platform_backend(platform)
    if language == "python":
        if target is None:
            raise ValueError("Python bundles require an explicit supported platform")
        return target
    for backend in REGISTERED_BACKENDS:
        if language != backend.language:
            continue
        if target is not None and target is not backend:
            raise ValueError(
                f"language {language!r} conflicts with platform {platform!r}"
            )
        return backend
    raise ValueError(f"unsupported kernel language {language!r}")


def detect_backend() -> Backend:
    """Probe available backends, honoring KLINEAGE_BACKEND when set.

    The environment value selects a platform prefix. Without it, probe in
    registration order and skip drivers raising ImportError or RuntimeError.
    This selects a backend family; device indices and visibility remain under
    the vendor runtime and controller environment.

    Returns:
        The selected backend reporting ``available=True``.

    Raises:
        ValueError: The explicit platform is unsupported.
        RuntimeError: The selected backend or every probed backend is unavailable.
        ImportError: The explicitly selected backend cannot import its runtime.
    """
    requested = os.environ.get(BACKEND_ENV)
    if requested:
        backend = get_backend("python", requested)
        if not backend.info()["available"]:
            raise RuntimeError(f"{backend.kind} is unavailable to this process")
        return backend
    for backend in REGISTERED_BACKENDS:
        try:
            available = backend.info()["available"]
        except (ImportError, RuntimeError):
            continue
        if available:
            return backend
    raise RuntimeError("No supported CUDA, Hygon HIP, or Ascend device is available")
