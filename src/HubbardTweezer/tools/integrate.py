import numpy as np
import torch
from scipy.integrate import romb, simpson

from ..DVR.core import DIM


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
    for i in range(DIM):
        if x[i].shape[0] > 1:
            integrand = torch.trapezoid(integrand, x=x[i], dim=0)
        else:
            integrand = integrand[0]
    return integrand
