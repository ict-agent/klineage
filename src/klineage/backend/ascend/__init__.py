"""ascendc backend public interface."""

from .kernel import ASCEND_ARCH_ENV, AscendBackend, AscendFormat

__all__ = [
    "AscendBackend",
    "ASCEND_ARCH_ENV",
    "AscendFormat",
]
