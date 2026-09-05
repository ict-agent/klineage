"""Exceptions raised by lineage actions and their supporting components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klineage.kernel import Kernel


class ActionError(RuntimeError):
    """Base error for lineage actions."""


class StructuredOutputError(ActionError):
    """Raised when an agent does not return the requested JSON object."""


class ValidationGateError(ActionError):
    """Raised when a generated kernel fails the external validation gate."""

    def __init__(
        self,
        message: str,
        *,
        kernel: Kernel | None = None,
    ) -> None:
        self.kernel = kernel
        super().__init__(message)


__all__ = ["ActionError", "StructuredOutputError", "ValidationGateError"]
