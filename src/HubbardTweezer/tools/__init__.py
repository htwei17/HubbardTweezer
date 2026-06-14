"""Shared utility helpers used across DVR and Hubbard modules."""

from .funcs import duplicate, find_variable_name
from .integrate import integrate_nd, romb3d, simps3d, trapz3d, trapz3dnp

__all__ = [
    "duplicate",
    "find_variable_name",
    "integrate_nd",
    "romb3d",
    "simps3d",
    "trapz3d",
    "trapz3dnp",
]
