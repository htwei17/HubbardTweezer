from mimetypes import init
from statistics import mode
from tkinter import N
import numpy as np
import scipy.linalg as la
# import scipy.sparse.linalg as sla
# import scipy.sparse as sp
# from scipy.sparse.linalg import LinearOperator
# import sparse
from opt_einsum import contract
# from einops import rearrange, reduce, repeat
from DVR.core import *

k = 10  # Number of energy levels to track


def N_convergence(N: int, R, avg=1, dim=3, level=1):
    # Convergence of energy vs N, to reproduce Fig. 5a in PRA paper
    E = np.array([]).reshape(0, k)
    dim_factor = np.zeros(3, dtype=int)
    dim_factor[:dim] = 1
    R = R * dim_factor
    x = np.linspace(-1.1 * R[0], 1.1 * R[0], int(1000))
    y = np.array([0])
    z = np.array([0])
    p = []

    for i in N:
        n = i * np.array([1, 1, 1])
        dx = R / n
        n = n * dim_factor
        D = DVR(n, R, avg=avg, sparse=True)
        print('dx= {}w'.format(dx))
        print('R= {}w'.format(R))
        print('n=', n)
        V, W = H_solver(D)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
        p.append(
            psi(n, dx, W.reshape(*(D.n + 1 - D.init), k), x, y, z)[:, 0,
                                                                   0, :level])
    dE = np.diff(E, axis=0)

    return np.array(N), dE, E, x / R[0], p


def R_convergence(N: int, dx):
    # Convergence of energy vs R, to reproduce Fig. 5b in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        Nlist = i * np.array([1, 1, 2]) + np.array([0, 0, 1])
        R = Nlist * dx
        D = DVR(Nlist, R)
        V, W = H_solver(D)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    R = np.array(N) * dx[0] / D.w
    return R, dE, E
