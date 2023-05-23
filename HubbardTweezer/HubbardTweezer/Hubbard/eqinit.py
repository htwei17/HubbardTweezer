import numpy as np
from numbers import Number
from typing import Iterable

from .core import symm_fold
from .lattice import Lattice

dmin = 1.4 # Minimum trap center spacing in unit of wx
# 1.4 wx is roughtly -0.75V0 barrier height

def init_V0(Voff: np.ndarray, lattice: Lattice, nobounds: bool = False):
    v01 = symm_fold(lattice.reflect, Voff)
    if nobounds:
        b1 = list((-np.inf, np.inf) for i in range(lattice.Nindep))
    else:
        b1 = list((0.95, 1.05) for i in range(lattice.Nindep))  # 5% ~ 2.5kHz fluctuation
    return v01, b1


def init_w0(
    lattice: Lattice,
    waists: np.ndarray,
    waist_dir,
    w_dof,
    lb: Number,
    nobounds: bool = False,
):
    if nobounds:
        s2 = (-np.inf, np.inf)
    else:
        s2 = (lb, 1.2)
    if waist_dir == None:
        v02 = np.array([])
        b2 = []
    else:
        # v02 = np.ones(2 * self.lattice.Nindep)
        v02 = symm_fold(lattice.reflect, waists).flatten()
        b2 = list(s2 for i in range(2 * lattice.Nindep) if w_dof[i])
        v02 = v02[w_dof]
    return v02, b2


def init_aij(
    lattice: Lattice,
    lc: Iterable,
    trap_centers: np.ndarray,
    tc_dof,
    nobounds: bool = False,
):
    if nobounds:
        s3 = (-np.inf, np.inf)
    else:
        s3 = abs(np.min(lc) - dmin) / 2
    v03 = symm_fold(lattice.reflect, trap_centers).flatten()
    b3 = list(
        (v03[i] - s3, v03[i] + s3) for i in range(2 * lattice.Nindep) if tc_dof[i]
    )
    v03 = v03[tc_dof]
    return v03, b3
