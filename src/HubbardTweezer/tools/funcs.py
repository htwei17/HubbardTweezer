from typing import Iterable
from numbers import Number

import numpy as np


def duplicate(var, N: int):
    if var is None:
        return None
    if N == 0:
        return 0
    if not isinstance(var, Iterable):
        var = [var] * N
        var = np.array(var)
    elif len(var) == 1:
        var = np.ones(N) * var[0]
    elif len(var) == N // 2 + N % 2:
        # If the half of the array is given, reflect it
        var = np.concatenate([var, var[N % 2 :][::-1]])
    elif N % len(var) == 0:
        var = np.tile(var, N // len(var))
    elif len(var) != N:
        raise ValueError(
            f"The array '{find_variable_name(var)}' has a length {len(var)} not equal to N={N} or N is not divisible by the array length."
        )
    return var


def find_variable_name(obj, scope=locals()):
    # Find the name of a variable in a given scope
    # Set scope to locals() or globals() to search in the local or global scope
    for name, value in scope.items():
        if value is obj:  # Use 'is' to check if they are the same object
            return name
    return None
