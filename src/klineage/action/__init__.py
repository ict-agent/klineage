"""Actions that return typed artifact trees."""

from klineage.action.action import Action, same_implementation
from klineage.action.apply import Apply
from klineage.action.decompose import Decompose
from klineage.action.generation import Generation
from klineage.action.init import Init

__all__ = [
    "same_implementation",
    "Action",
    "Apply",
    "Decompose",
    "Generation",
    "Init",
]
