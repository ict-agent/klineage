"""Exceptions raised by kernel actions and their supporting components."""


class ActionError(RuntimeError):
    """Base error for kernel actions."""


class StructuredOutputError(ActionError):
    """An agent response or artifact violates its required format."""


__all__ = ["ActionError", "StructuredOutputError"]
