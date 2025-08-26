from typing import Iterable
from numbers import Number

import numpy as np
from scipy.special import erf


def tweezer_potential(x, y, z, wxy, zR, zR0):
    # Tweezer potential function
    """Calculate the tweezer potential based on the given coordinates and waist parameters.
    wxy: Waist parameters for x and y directions.
    zR0: Rayleigh range parameters for anisotrpic x and y directions.
    """
    d0 = 1 + (z / zR0) ** 2 / 2
    dxy = (x / wxy[0]) ** 2 / (1 + (z / zR[0]) ** 2)
    dxy += (y / wxy[1]) ** 2 / (1 + (z / zR[1]) ** 2)
    V = -1 / d0 * np.exp(-2 * dxy)
    return V


def shaking_potential(x, y, z, amp, wxy, zR, zR0):
    # Shaking potential function
    """Calculate the shaking potential based on the given coordinates and waist parameters.
    amp: Shaking amplitude, the triangular shaking is in between (-amp, amp).
    wxy: Waist parameters for x and y directions.
    zR0: Rayleigh range parameters for anisotropic x and y directions.
    """
    d0 = 1 + (z / zR0) ** 2 / 2
    dy = (y / wxy[1]) ** 2 / (1 + (z / zR[1]) ** 2)
    wx = wxy[0] * np.sqrt(1 + (z / zR[0]) ** 2) / np.sqrt(2)
    intgrl = erf((x + amp) / wx) - erf((x - amp) / wx)
    intgrl *= np.sqrt(np.pi) * wx / (4 * amp)
    V = -1 / d0 * np.exp(-2 * dy) * intgrl
    return V


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
