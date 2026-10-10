"""Actions that return typed artifact trees."""

from klineage.action.action import Action, same_implementation
from klineage.action.apply import Apply, apply
from klineage.action.decompose import Decompose, decompose
from klineage.action.init import Init, init

__all__ = [
    "same_implementation",
    "Action",
    "Apply",
    "Decompose",
    "Init",
    "apply",
    "decompose",
    "init",
]
