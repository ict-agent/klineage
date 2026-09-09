"""Exceptions raised by kernel actions and their supporting components."""


class ActionError(RuntimeError):
    """Base error for kernel actions."""


class StructuredOutputError(ActionError):
    """An agent response or artifact violates its required format."""


class ValidationGateError(ActionError):
    """Raised when a generated kernel fails the external validation gate."""


__all__ = ["ActionError", "StructuredOutputError", "ValidationGateError"]
