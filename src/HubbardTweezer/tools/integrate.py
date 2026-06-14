from __future__ import annotations

import numpy as np
from scipy.integrate import romb, simpson

from ..DVR.metadata import DIM

try:
    import torch
except ImportError:
    torch = None


def romb3d(integrand: np.ndarray, dx: list[float, float, float]) -> float:
    for i in range(DIM):
        if dx[i] > 0:
            integrand = romb(integrand, dx[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def simps3d(integrand: np.ndarray, x: list[np.ndarray, np.ndarray, np.ndarray]) -> float:
    for i in range(DIM):
        if x[i].size > 1:
            integrand = simpson(integrand, x[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def trapz3dnp(integrand: np.ndarray, x: list[np.ndarray, np.ndarray, np.ndarray]) -> float:
    for i in range(DIM):
        if x[i].size > 1:
            integrand = np.trapezoid(integrand, x[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def trapz3d(integrand: torch.Tensor, x: list[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if torch is None:
        raise ImportError("torch is required for trapz3d.")
    for i in range(DIM):
        if x[i].shape[0] > 1:
            integrand = torch.trapezoid(integrand, x=x[i], dim=0)
        else:
            integrand = integrand[0]
    return integrand


def integrate_nd(x, dx, integrand, method: str = "trapz"):
    if method == "romb":  # Not recommended as it is not converging well
        return romb3d(integrand, dx)
    return trapz3dnp(integrand, x)
