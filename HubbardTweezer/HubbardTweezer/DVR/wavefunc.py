import numpy as np
from opt_einsum import contract

# from numba import njit, guvectorize, int64, float64, complex128

from .const import dim
from .core import get_init


def psi(
    x: list[np.ndarray, np.ndarray, np.ndarray],
    n: np.ndarray,
    dx: np.ndarray,
    W: np.ndarray,
    p: np.ndarray = np.zeros(dim, dtype=int),
) -> np.ndarray:
    init = get_init(n, p)
    # V = np.sum(
    #     W.reshape(*(np.append(n + 1 - init, -1))), axis=1
    # )  # Sum over y, z index to get y=z=0 cross section of the wavefunction
    deltax = dx.copy()
    nd = deltax == 0
    deltax[nd] = 1
    xn = [np.arange(init[i], n[i] + 1, dtype=float) for i in range(dim)]
    x = [x[i] / deltax[i] for i in range(dim)]
    # map object itself is not a list, but we can unpacked it by *V
    V = map(delta, p, x, xn)
    # ufunc of list of different length of arrays are not supported by numpy
    # vd = np.vectorize(delta, signature='(),(m),(n)->(m,n)')
    # V = vd(p, x, xn)
    # V = delta(p, x, xn)
    if W.ndim == 3:
        W = W[..., None]
    psi = 1 / np.sqrt(np.prod(deltax)) * contract("il,jm,kn,lmno", *V, W)
    return psi


def delta(p: int, x: np.ndarray, xn: np.ndarray) -> np.ndarray:
    # Symmetrized sinc DVR basis funciton, x in unit of dx
    Wx = np.sinc(x[:, None] - xn[None])
    if p != 0:
        Wx += p * np.sinc(x[:, None] + xn[None])
        Wx /= np.sqrt(2)
        if p == 1:
            Wx[:, 0] /= np.sqrt(2)
    return Wx


# @guvectorize([(int64, float64[:], float64[:], float64[:, :])], '(),(m),(n)->(m,n)', target='parallel')
# def delta(p: int, x: np.ndarray, xn: np.ndarray, Wx):
#     # Symmetrized sinc DVR basis funciton, x in unit of dx
#     x = x.copy()
#     xn = xn.copy()
#     Wx = np.sinc(x.reshape(-1, 1) - xn.reshape(1, -1))
#     if p != 0:
#         Wx += p * np.sinc(x.reshape(-1, 1) + xn.reshape(1, -1))
#         Wx /= np.sqrt(2)
#         if p == 1:
#             Wx[:, 0] /= np.sqrt(2)
