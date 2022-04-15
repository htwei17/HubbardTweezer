import imp
from typing import Iterable
from numpy import double, dtype
from opt_einsum import contract
from pyrsistent import get_in
from positify import positify

import sparse
import torch
import autograd
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers

from DVR_full import *
import numpy.linalg as la


class Wannier(DVR):

    def update_lattice(self, lattice: np.ndarray, lc=(ax, ay)):
        # graph : each line represents coordinate (x, y) of one lattice site
        self.lattice = lattice.copy()
        self.graph, self.links = lattice_graph(lattice)
        self.lc = np.array(lc) * 1E-9 / (a0 * self.w)  # In unit of w
        self.Nsite = np.prod(self.lattice)

        dx = np.zeros(self.n.shape)
        dx[self.nd] = self.R0[self.nd] / self.n[self.nd]
        lattice = np.resize(np.pad(lattice, (0, 2), constant_values=1), 3)
        lc = np.resize(self.lc, 3)
        print('lattice: Full lattice sizes: {}'.format(lattice))
        print('lattice: lattice constants: {}w'.format(lc))
        # Let there be R0's wide outside the edge trap center
        R0 = (lattice - 1) * lc / 2 + self.R0
        R0 *= self.nd
        self.update_R0(R0, dx)

    def __init__(
            self,
            N: int,
            R0: np.ndarray,
            lattice: np.ndarray = np.array(
                [2], dtype=int),  # Square lattice dimensions
            lc=(1520, 1690),  # Lattice constant, in unit of nm
            ascatt=200,  # Scattering length, in unit of Bohr radius
            avg=1,
            dim: int = 3,
            model='Gaussian',
            trap=(104.52, 1000),
            symmetry: bool = False,
            absorber: bool = False,
            ab_param=(57.04, 1),
            sparse: bool = True) -> None:

        self.N = N
        self.scatt_len = ascatt
        n = np.zeros(3, dtype=int)
        n[:dim] = N
        super().__init__(n, R0, avg, model, trap, symmetry, absorber, ab_param,
                         sparse)

        self.update_lattice(lattice, lc)

    def Vfun(self, x, y, z):
        V = 0
        # TODO: euqlize trap depths for n>3 traps?
        # NOTE: DO NOT SET coord DIRECTLY! THIS WILL DIRECTLY MODIFY self.graph!
        for coord in self.graph:
            shift = coord * self.lc
            V += super().Vfun(x - shift[0], y - shift[1], z)
        return V


def lattice_graph(size: np.ndarray):
    # Square lattice graph builder
    # graph: each ndarray object in graph is a coordinate pair (x, y)
    #        indicating the posistion of node (trap center)
    # links: each row in links is a pair of node indices s.t.
    #        graph[idx1], graph[idx2] are linked by bounds

    if isinstance(size, Iterable):
        size = np.array(size)

    edge = []
    links = []
    for i in range(size.size):
        edge.append(np.arange(-(size[i] - 1) / 2, (size[i] - 1) / 2 + 1))
    if size.size == 1:
        graph = [np.array([i, 0]) for i in edge[0]]
        links = [np.array([i, i + 1]) for i in range(size[0] - 1)]
    elif size.size == 2:
        graph = []
        links = np.array([]).reshape(0, 2)
        node_idx = 0  # Linear index is column (y) prefered
        for i in range(len(edge[0])):
            for j in range(len(edge[1])):
                graph.append(np.array([edge[0][i], edge[1][j]]))
                if i > 0:
                    links = np.append(links,
                                      np.array([node_idx - size[1],
                                                node_idx])[None],
                                      axis=0)  # Row link
                if j > 0:
                    links = np.append(links,
                                      np.array([node_idx - 1, node_idx])[None],
                                      axis=0)  # Column linke
                node_idx += 1

    return graph, links


def eigen_basis(dvr: Wannier):
    if dvr.symmetry:
        # The code is designed for 1D and 2D lattice, and
        # we always leave z direction to be in even sector.
        # So all sectors are [x x 1] with x collected below.
        # This is because the energy cost to go higher level
        # of z direction is of ~5kHz, way above the bandwidth ~200Hz.
        # TODO: add support for 3D lattice

        dvr.p = np.zeros(dim, dtype=int)
        dvr.p[dvr.nd] = 1  # [1 1 1] sector

        E = np.array([])
        W = []
        parity = np.array([], dtype=int).reshape(0, 2)
        E, W, parity = add_sector([1, 1], dvr, E, W, parity)  #  [1 1] sector
        E, W, parity = add_sector([-1, 1], dvr, E, W, parity)  #  [-1 1] sector

        if dvr.lattice.size > 1 and not any(dvr.lattice == 1):  # 2D lattice
            E, W, parity = add_sector([1, -1], dvr, E, W,
                                      parity)  # [1 -1] sector
            E, W, parity = add_sector([-1, -1], dvr, E, W,
                                      parity)  # [-1 -1] sector

        # Sort everything by energy, only keetp lowest Nsite states
        idx = np.argsort(E)[:dvr.Nsite]
        E = E[idx]
        W = [W[i] for i in idx]
        parity = parity[idx, :]

    else:
        E, W = H_solver(dvr, dvr.Nsite)
        W = [W[:, i] for i in range(dvr.Nsite)]
        parity = np.zeros((dvr.Nsite, 2))

    return E, W, parity


def add_sector(sector: np.ndarray, dvr: Wannier, E, W, parity):
    sec_idx = np.array([True, True, False]) * dvr.nd
    p = dvr.p.copy()
    p[sec_idx] = sector[:np.sum(sec_idx)]
    dvr.update_p(p)

    Em, Wm = H_solver(dvr, dvr.Nsite)
    E = np.append(E, Em)
    W += [Wm[:, i].reshape(dvr.n + 1 - dvr.init) for i in range(dvr.Nsite)]
    # Parity sector marker
    parity = np.append(parity, np.tile(sector, (dvr.Nsite, 1)), axis=0)
    return E, W, parity


def cost_matrix(dvr: Wannier, W, parity):
    R = []

    # For calculation keeping only p_z = 1 sector,
    # Z is always zero. Explained below.
    for i in range(dim - 1):
        if dvr.nd[i]:
            Rx = np.zeros((dvr.Nsite, dvr.Nsite))
            if dvr.symmetry:
                # Get X^pp'_ij matrix rep for Delta^p_i, p the parity.
                # This matrix is nonzero only when p!=p':
                # X^pp'_ij = x_i delta_ij with p_x * p'_x = -1
                # As 1 and -1 sector have a dimension difference,
                # the matrix X is alsways a n-diagonal matrix
                # with n-by-n+1 or n+1-by-n dimension
                # So, Z is always zero, if we only use states in
                # p_z = 1 sectors.
                x = np.arange(1, dvr.n[i] + 1) * dvr.dx[i]
                lenx = len(x)
                idx = np.roll(np.arange(dim, dtype=int), -i)
                for j in range(dvr.Nsite):
                    Wj = np.transpose(W[j], idx)[-lenx:, :, :].conj()
                    for k in range(j + 1, dvr.Nsite):
                        pjk = parity[j, idx[idx < dim -
                                            1]] * parity[k, idx[idx < dim - 1]]
                        if pjk[0] == -1 and pjk[1:] == 1:
                            Wk = np.transpose(W[k], idx)[-lenx:, :, :]
                            Rx[j, k] = contract('ijk,i,ijk', Wj, x, Wk)
                            Rx[k, j] = Rx[j, k].conj()
            else:
                # X = x_i delta_ij for non-symmetrized basis
                x = np.arange(-dvr.n[i], dvr.n[i] + 1) * dvr.dx[i]
                for j in range(dvr.Nsite):
                    Wj = np.transpose(W[j], idx)[-lenx:, :, :].conj()
                    Rx[j, j] = contract('ijk,i,ijk', Wj, x, Wj.conj())
                    for k in range(j + 1, dvr.Nsite):
                        Wk = np.transpose(W[k], idx)[-lenx:, :, :]
                        Rx[j, k] = contract('ijk,i,ijk', Wj, x, Wk)
                        Rx[k, j] = Rx[j, k].conj()
            R.append(Rx)
    return R


def multiTensor(R):
    # Convert list of ndarray to list of Tensor
    return [
        torch.complex(torch.from_numpy(Ri),
                      torch.zeros(Ri.shape, dtype=torch.float64)) for Ri in R
    ]


def cost_func(U, R) -> float:
    # Cost function to optimize
    o = 0
    for i in range(len(R)):
        X = U.conj().T @ R[i] @ U
        Xp = X - torch.diag(torch.diag(X))
        o += torch.trace(torch.matrix_power(Xp, 2))
    return np.real(o)


def optimization(dvr: Wannier, E, W, parity):
    R = multiTensor(cost_matrix(dvr, W, parity))

    manifold = pymanopt.manifolds.Unitaries(dvr.Nsite)

    @pymanopt.function.pytorch(manifold)
    def cost(point: torch.Tensor):
        return cost_func(point, R)

    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    solver = pymanopt.solvers.SteepestDescent()
    solution = solver.solve(problem)

    solution = positify(solution)

    A = solution.conj().T @ (
        E[:, None] * solution
    ) * dvr.V0_SI / dvr.kHz_2p  # TB parameter matrix, in unit of kHz
    return A, solution


def tight_binding(dvr: Wannier):
    E, W, parity = eigen_basis(dvr)
    A = optimization(dvr, E, W, parity)
    mu = np.diag(A)  # Diagonals are mu_i
    t = -(A - np.diag(mu))  # Off-diagonals are t_ij
    return np.real(mu), abs(t)


def inteaction(dvr: Wannier, U, W, parity):
    T = np.zeros(dvr.Nsite * np.ones(4, dtype=int))
    for i in range(dvr.Nsite):
        Wi = W[i].conj()
        for j in range(dvr.Nsite):
            Wj = W[j].conj()
            for k in range(dvr.Nsite):
                Wk = W[k]
                for l in range(dvr.Nsite):
                    Wl = W[l]
                    if dvr.symmetry:
                        p_ijkl = np.concatenate((parity[i, :], parity[j, :],
                                                 parity[k, :], parity[l, :]),
                                                axis=0)
                        pqrs = np.prod(p_ijkl, axis=0)
                        # Cancel n = 0 column for any p = -1 in one direction
                        nlen = dvr.n
                        # Mark which dimension has n=0 basis
                        line0 = np.all(p_ijkl != -1, axis=0)
                        nlen[line0] += 1
                        pref0 = 1 / dvr.dx  # prefactor of n=0 line
                        pref = (1 + pqrs) / 4 * pref0  # prefactor of n>0 lines

                        for d in range(dim):
                            f = pref[d] * np.ones(nlen[d])
                            if line0[d]:
                                f[0] = pref0[d]
                            idx = np.ones(dim, dtype=int)
                            idx[d] = len(f)
                            f = f.reshape(idx)
                            Wl = f * Wl
                        T[i, j, k,
                          l] = contract('ijk,ijk,ijk,ijk', pick(Wi, nlen),
                                        pick(Wj, nlen), pick(Wk, nlen),
                                        pick(Wl, nlen))
                    else:
                        T[i, j, k, l] = contract('ijk,ijk,ijk,ijk', Wi, Wj, Wk,
                                                 Wl)

    u = 4 * np.pi * hb**2 * dvr.scatt_len / dvr.m
    return u * V


def pick(W, nlen):
    return W[-nlen[0]:, -nlen[1]:, -nlen[2]:]


def wannier_func(x, dvr: Wannier, W, U, parity):
    V = np.array([]).reshape(len(x), 0)
    for i in range(parity.shape[0]):
        V = np.append(V,
                      psi(dvr.n[0], dvr.dx[0], W[i], x,
                          parity[i, 0]).reshape(-1, 1),
                      axis=1)
    return V @ U


def parity_transfm(n: int):
    # Parity basis transformation matrix: 1D version

    Up = np.zeros((n + 1, 2 * n + 1))
    Up[0, n] = 1  # n=d
    Up[1:, :n] = np.eye(n)  # Negative half-axis
    Up[1:, (n + 1):] = np.eye(n)  # Positive half-axis
    Um = np.zeros((n, 2 * n + 1))
    Um[:, n:] = -1 * np.eye(n)  # Negative half-axis
    Um[:, :n] = np.eye(n)  # Positive half-axis
    return Up, Um
