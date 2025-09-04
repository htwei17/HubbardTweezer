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


def shaking_potential(x, y, z, amp, wxy, zR, zR0, balance=False):
    # Shaking potential function
    """Calculate the shaking potential based on the given coordinates and waist parameters.
    amp: Shaking amplitude, the triangular shaking is in between (-amp, amp).
    wxy: Waist parameters for x and y directions.
    zR0: Rayleigh range parameters for anisotropic x and y directions.
    """
    factor = 1
    if balance:
        factor = np.sqrt(8 / np.pi) * amp / (wxy[0] * erf(np.sqrt(2) * amp / wxy[0]))
    d0 = 1 + (z / zR0) ** 2 / 2
    dy = (y / wxy[1]) ** 2 / (1 + (z / zR[1]) ** 2)
    wx = wxy[0] * np.sqrt(1 + (z / zR[0]) ** 2) / np.sqrt(2)
    intgrl = erf((x + amp) / wx) - erf((x - amp) / wx)
    intgrl *= np.sqrt(np.pi) * wx / (4 * amp)
    V = -factor / d0 * np.exp(-2 * dy) * intgrl
    return V
