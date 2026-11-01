"""hip backend public interface."""

from .kernel import HIP_ARCH_ENV, HipBackend

__all__ = ["HipBackend", "HIP_ARCH_ENV"]
