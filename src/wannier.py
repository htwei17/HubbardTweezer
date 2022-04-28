from typing import Iterable
from numpy import double, dtype
from opt_einsum import contract
from pyrsistent import get_in
from positify import positify
from scipy.integrate import romb
import itertools
import sparse
import torch
import autograd
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers

from DVR_full import *
import numpy.linalg as la


class Wannier(DVR):

    def update_lattice(self, lattice: np.ndarray, lc=(1520, 1690)):
        # graph : each line represents coordinate (x, y) of one lattice site
        self.lattice = lattice.copy()
        self.graph, self.links = lattice_graph(lattice)
        if self.model == 'Gaussian':
            self.lc = np.array(lc) * 1E-9 / (a0 * self.w)  # In unit of w
        elif self.model == 'sho':
            self.lc = np.array(lc)
        self.Nsite = np.prod(self.lattice)

        dx = np.zeros(self.n.shape)
        dx[self.nd] = self.R0[self.nd] / self.n[self.nd]
        lattice = np.resize(np.pad(lattice, (0, 2), constant_values=1), dim)
        lc = np.resize(self.lc, dim)
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
            atom=6.015122,  # Atom mass, in amu
            laser=780,  # 780nm, laser wavelength
            ascatt=560,  # Scattering length, in unit of Bohr radius
            avg=1,
            band=1,  # Number of bands
            dim: int = 3,
            model='Gaussian',
            trap=(104.52, 1000),
            symmetry: bool = False,
            absorber: bool = False,
            ab_param=(57.04, 1),
            sparse: bool = True) -> None:

        self.N = N
        self.scatt_len = ascatt
        self.dim = dim
        self.bands = band
        n = np.zeros(3, dtype=int)
        n[:dim] = N
        super().__init__(n, R0, avg, model, trap, atom, laser, symmetry,
                         absorber, ab_param, sparse)
        self.update_lattice(lattice, lc)

    def Vfun(self, x, y, z):
        V = 0
        # TODO: euqlize trap depths for n>3 traps?
        # NOTE: DO NOT SET coord DIRECTLY! THIS WILL DIRECTLY MODIFY self.graph!
        if self.model == 'sho' and self.Nsite == 2:
            V += super().Vfun(abs(x) - self.lc[0] / 2, y, z)
        else:
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
    k = dvr.Nsite * dvr.bands
    if dvr.symmetry:
        # The code is designed for 1D and 2D lattice, and
        # we always leave z direction to be in even sector.
        # So all sectors are [x x 1] with x collected below.
        # This is because the energy cost to go higher level
        # of z direction is of ~5kHz, way above the bandwidth ~200Hz.

        p_list = sector(dvr)
        E_sb = np.array([])
        W_sb = []
        p_sb = np.array([], dtype=int).reshape(0, 2)
        for p in p_list:
            E_sb, W_sb, p_sb = add_sector(p, dvr, k, E_sb, W_sb, p_sb)

        # Sort everything by energy, only keetp lowest k states
        idx = np.argsort(E_sb)[:k]
        E_sb = E_sb[idx]
        W_sb = [W_sb[i] for i in idx]
        p_sb = p_sb[idx, :]
    else:
        E_sb, W_sb = H_solver(dvr, k)
        W_sb = [W_sb[:, i] for i in range(k)]
        p_sb = np.zeros((k, 2))

    E = []
    W = []
    parity = []
    for b in range(dvr.bands):
        E.append(E_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite])
        W.append(W_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite])
        parity.append(p_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite, :])

    return E, W, parity


def sector(dvr):
    # BUild all sectors
    # x direction
    if dvr.Nsite == 1:
        p_tuple = [[1]]
    else:
        p_tuple = [[1, -1]]
    # y direction. Rule out geometry=[n, 1] case
    if dvr.lattice.size > 1 and not any(dvr.lattice == 1):
        p_tuple.append([1, -1])
    else:
        p_tuple.append([1])
        # For a general omega_z << omega_x,y case,
        # the lowest several bands are in
        # z=1, z=-1, z=1 sector, etc... alternatively
        # A simplest way to build bands is to simply collect
        # Nband * Nsite lowest energy states
        # z direction
    if dvr.bands == 1:
        p_tuple.append([1])
    elif dvr.bands > 1:
        p_tuple.append([1, -1])
    p_list = list(itertools.product(*p_tuple))
    return p_list


def add_sector(sector: np.ndarray, dvr: Wannier, k, E, W, parity):
    p = dvr.p.copy()
    p[:len(sector)] = sector
    dvr.update_p(p)

    Em, Wm = H_solver(dvr, k)
    E = np.append(E, Em)
    W += [Wm[:, i].reshape(dvr.n + 1 - dvr.init) for i in range(k)]
    # Parity sector marker
    parity = np.append(parity, np.tile(sector, (k, 1)), axis=0)
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
    # Multiband optimization
    A = []
    w = []
    for b in range(dvr.bands):
        t_ij, w_mu = singleband_optimization(dvr, E[b], W[b], parity[b])
        A.append(t_ij)
        w.append(w_mu)
    return A, w


def singleband_optimization(dvr: Wannier, E, W, parity):
    # Singleband Wannier optimization
    R = multiTensor(cost_matrix(dvr, W, parity))

    if dvr.Nsite > 1:
        manifold = pymanopt.manifolds.Unitaries(dvr.Nsite)

        @pymanopt.function.pytorch(manifold)
        def cost(point: torch.Tensor):
            return cost_func(point, R)

        problem = pymanopt.Problem(manifold=manifold, cost=cost)
        solver = pymanopt.solvers.SteepestDescent()
        solution = solver.solve(problem)

        solution = positify(solution)
    elif dvr.Nsite == 1:
        solution = np.ones((1, 1))

    U = lattice_order(dvr, solution)

    A = U.conj().T @ (
        E[:, None] *
        U) * dvr.V0_SI / dvr.kHz_2p  # TB parameter matrix, in unit of kHz
    return A, U


def lattice_order(dvr: Wannier, solution):
    # Order Wannier functions by lattice site label
    return U


def tight_binding(dvr: Wannier):
    E, W, parity = eigen_basis(dvr)
    A, w = optimization(dvr, E, W, parity)
    mu = np.diag(A)  # Diagonals are mu_i
    t = -(A - np.diag(mu))  # Off-diagonals are t_ij
    return np.real(mu), abs(t)


def interaction(dvr: Wannier, U, W, parity: np.ndarray):
    # Interaction between i band and j band
    Uint = np.zeros((dvr.bands, dvr.bands, dvr.Nsite))
    for i in range(dvr.bands):
        for j in range(dvr.bands):
            Uint[i, j, :] = singleband_interaction(dvr, U[i], U[j], W[i], W[j],
                                                   parity[i], parity[j])
    return Uint


def singleband_interaction(dvr: Wannier, Ui, Uj, Wi, Wj, pi: np.ndarray,
                           pj: np.ndarray):
    u = 4 * np.pi * hb**2 * dvr.scatt_len / dvr.m
    # Construct integral of 2-body eigenstates, due to basis quadrature it is reduced to sum 'ijk,ijk,ijk,ijk'
    integrl = np.zeros(dvr.Nsite * np.ones(4, dtype=int))
    intgrl_mat(dvr, Wi, Wj, pi, pj, integrl)  # np.ndarray is global variable
    # Bands are degenerate, only differed by spin index, so they share the same set of Wannier functions
    Uint = contract('ia,jb,kc,ld,ijkl->abcd', Ui.conj(), Uj.conj(), Uj, Ui,
                    integrl)
    Uint_onsite = np.zeros(dvr.Nsite)
    for i in range(dvr.Nsite):
        if dvr.model == 'sho':
            print(
                'Test with analytic calculation on {}-th site'.format(i + 1),
                np.real(Uint[i, i, i, i]) * (np.sqrt(2 * np.pi))**dvr.dim *
                np.prod(dvr.hl))
        Uint_onsite[i] = u * np.real(Uint[i, i, i, i])
    return Uint_onsite

    x = []
    dx = []
    for i in range(dim):
        if dvr.nd[i]:
            x.append(np.linspace(-1.2 * dvr.R0[i], 1.2 * dvr.R0[i], 129))
            dx.append(x[i][1] - x[i][0])
        else:
            x.append(np.array([0]))
            dx.append(0)
    Vi = wannier_func(dvr, Wi, Ui, pi, x)
    Vj = wannier_func(dvr, Wj, Uj, pj, x)
    wannier = abs(Vi)**2 * abs(Vj)**2
    Uint_onsite = intgrl3d(dx, wannier)
    if dvr.model == 'sho':
        print(
            'Test with analytic calculation on {}-th site'.format(i + 1),
            np.real(Uint_onsite) * (np.sqrt(2 * np.pi))**dvr.dim *
            np.prod(dvr.hl))
    return u * Uint_onsite


def wannier_func(dvr, W, U, p, x):
    V = np.array([]).reshape(len(x[0]), len(x[1]), len(x[2]), 0)
    for i in range(p.shape[0]):
        V = np.append(V,
                      psi(dvr.n, dvr.dx, W[i], *x, p[i, :])[..., None],
                      axis=dim)
        print('{}-th Wannier function finished.'.format(i))
    return V @ U


def intgrl3d(dx, integrand):
    for i in range(dim):
        if dx[i] > 0:
            integrand = romb(integrand, dx[i], axis=0)
        else:
            integrand = integrand[0, :]
    return integrand


def intgrl_mat(dvr, Wi, Wj, pi, pj, integrl):
    # Construct integral of 2-body eigenstates, due to basis quadrature it is reduced to sum 'ijk,ijk,ijk,ijk'
    for i in range(dvr.Nsite):
        Wii = Wi[i].conj()
        for j in range(dvr.Nsite):
            Wjj = Wj[j].conj()
            for k in range(dvr.Nsite):
                Wjk = Wj[k]
                for l in range(dvr.Nsite):
                    Wil = Wi[l]
                    if dvr.symmetry:
                        p_ijkl = np.concatenate(
                            (pi[i, :][None], pj[j, :][None], pj[k, :][None],
                             pi[l, :][None]),
                            axis=0)
                        pqrs = np.prod(p_ijkl, axis=0)
                        # Cancel n = 0 column for any p = -1 in one direction
                        nlen = dvr.n.copy()
                        # Mark which dimension has n=0 basis
                        line0 = np.all(p_ijkl != -1, axis=0)
                        nlen[line0] += 1
                        # prefactor of n=0 line
                        pref0 = np.zeros(dim)
                        pref0[dvr.nd] = 1 / dvr.dx[dvr.nd]
                        # prefactor of n>0 lines
                        pref = (1 + pqrs) / 4 * pref0
                        for d in range(dim):
                            if dvr.nd[d]:
                                f = pref[d] * np.ones(Wil.shape[d])
                                if Wil.shape[d] > dvr.n[d]:
                                    f[0] = pref0[d]
                                idx = np.ones(dim, dtype=int)
                                idx[d] = len(f)
                                f = f.reshape(idx)
                                Wil = f * Wil
                        integrl[i, j, k, l] = contract('ijk,ijk,ijk,ijk',
                                                       pick(Wii, nlen),
                                                       pick(Wjj, nlen),
                                                       pick(Wjk, nlen),
                                                       pick(Wil, nlen))
                    else:
                        integrl[i, j, k, l] = contract('ijk,ijk,ijk,ijk', Wii,
                                                       Wjj, Wjk, Wil)


def pick(W, nlen):
    # Truncate matrix to only n>0 columns
    return W[-nlen[0]:, -nlen[1]:, -nlen[2]:]


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
