import numpy as np
import torch
from scipy.integrate import romb, simps

from ..DVR.core import dim


def romb3d(integrand: np.ndarray, dx: list[float, float, float]) -> float:
    for i in range(dim):
        if dx[i] > 0:
            integrand = romb(integrand, dx[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def simps3d(integrand: np.ndarray, x: list[np.ndarray, np.ndarray, np.ndarray]) -> float:
    for i in range(dim):
        if x[i].size > 1:
            integrand = simps(integrand, x[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def trapz3dnp(integrand: np.ndarray, x: list[np.ndarray, np.ndarray, np.ndarray]) -> float:
    for i in range(dim):
        if x[i].size > 1:
            integrand = np.trapz(integrand, x[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand


def trapz3d(integrand: torch.Tensor, x: list[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    for i in range(dim):
        if x[i].shape[0] > 1:
            integrand = torch.trapezoid(integrand, x=x[i], dim=0)
        else:
            integrand = integrand[0]
    return integrand
