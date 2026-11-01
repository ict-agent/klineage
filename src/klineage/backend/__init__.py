"""Public backend selection and extension interfaces."""

from .kernel import Backend, BackendKind
from .registry import (
    backends,
    detect_backend,
    get_backend,
    get_timing_backend,
    platform_backend,
    register_backend,
)
from .timing import (
    TimingPolicy,
    TimingResult,
    bench,
)

__all__ = [
    "Backend",
    "BackendKind",
    "backends",
    "detect_backend",
    "get_backend",
    "platform_backend",
    "register_backend",
    "get_timing_backend",
    "TimingPolicy",
    "TimingResult",
    "bench",
]
