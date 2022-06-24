from cmath import inf, nan
from typing import Iterable
# from numpy import double, dtype
from opt_einsum import contract
from tools.positify import positify
from scipy.integrate import romb
import itertools
import sparse

import torch
# import autograd
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers

from DVR.core import *
from Hubbard.lattice import *
import numpy.linalg as la


class MLWF(DVR):

    def create_lattice(self,
                       lattice: np.ndarray,
                       lc=(1520, 1690),
                       shape='square'):
        # graph : each line represents coordinate (x, y) of one lattice site

        self.Nsite = np.prod(lattice)
        self.lattice_shape = shape

        # Convert [n] to [n, 1]
        if self.Nsite == 1:
            self.lattice = np.ones(1)
            self.lattice_dim = 1
        elif self.Nsite > 1 and lattice.size == 1:
            self.lattice = np.resize(
                np.pad(lattice, pad_width=(0, 1), constant_values=1), 2)
            self.lattice_dim = 1
        else:
            self.lattice = lattice.copy()
            self.lattice_dim = lattice[lattice > 1].size

        # Convert lc to (lc, lc) or the other if only one number is given
        if not isinstance(lc, Iterable):
            if np.isin(shape, np.array(['triangular', 'honeycomvb',
                                        'kagome'])):
                # For equilateral triangle
                lc = (lc, np.sqrt(3) / 2 * lc)
            else:
                # For squre
                lc = (lc, lc)

        self.trap_centers, self.links, self.reflection = lattice_graph(
            self.lattice, shape)
        self.Nsite = self.trap_centers.shape[0]

        self.Nindep = self.reflection.shape[
            0]  # Independent trap number under reflection symmetry

        if self.model == 'Gaussian':
            self.lc = np.array(lc) * 1E-9 / self.w  # In unit of wx
        elif self.model == 'sho':
            self.lc = np.array(lc)

        dx = self.dx.copy()
        lattice_range = np.max(abs(self.trap_centers), axis=0)
        lattice_range = np.resize(
            np.pad(lattice_range, (0, 2), constant_values=0), dim)
        lc = np.resize(self.lc, dim)
        print(f'lattice: dx is fixed at: {dx}w')
        print(f'lattice: lattice shape is {shape}')
        print(f'lattice: Full lattice sizes: {lattice}')
        print(f'lattice: lattice constants: {lc}w')
        # Let there be R0's wide outside the edge trap center
        R0 = lattice_range * lc + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def update_lattice(self, tc: np.ndarray):
        # Update DVR grids when self.graph is updated

        self.trap_centers = tc.copy()
        dx = self.dx.copy()
        lattice = np.resize(np.pad(self.lattice, (0, 2), constant_values=1),
                            dim)
        lc = np.resize(self.lc, dim)
        print(f'lattice: dx is fixed at: {dx}w')
        print(f'lattice: Full lattice sizes updated to: {lattice}')
        print(f'lattice: lattice constants updated to: {lc}w')
        # Let there be R0's wide outside the edge trap center
        R0 = (lattice - 1) * lc / 2 + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def __init__(
            self,
            N: int,
            lattice: np.ndarray = np.array(
                [2], dtype=int),  # Square lattice dimensions
            lc=(1520, 1690),  # Lattice constant, in unit of nm
            ascatt=1600,  # Scattering length, in unit of Bohr radius
            shape='square',  # Shape of the lattice
            band=1,  # Number of bands
            dim: int = 3,
            *args,
            **kwargs) -> None:

        self.N = N
        self.scatt_len = ascatt * a0
        self.dim = dim
        self.bands = band
        n = np.zeros(3, dtype=int)
        n[:dim] = N
        super().__init__(n, *args, **kwargs)
        # Backup of distance from edge trap center to DVR grid boundaries
        self.R00 = self.R0.copy()
        self.create_lattice(lattice, lc, shape)
        self.Voff = np.ones(self.Nsite)  # Set default trap offset

    def Vfun(self, x, y, z):
        # Get V(x, y, z) for the entire lattice
        V = 0

        if self.model == 'sho' and self.Nsite == 2:
            # Two-site SHO case
            V += super().Vfun(abs(x) - self.lc[0] / 2, y, z)
        else:
            # NOTE: DO NOT SET coord DIRECTLY! THIS WILL DIRECTLY MODIFY self.graph!
            for i in range(self.Nsite):
                shift = self.trap_centers[i] * self.lc
                V += self.Voff[i] * super().Vfun(x - shift[0], y - shift[1], z)
        return V

    def singleband_solution(self, u=False, eig_sol=None):
        # Calculate single band tij matrix and U matrix
        band_bak = self.bands
        self.bands = 1
        if eig_sol != None:
            E, W, p = eig_sol
        E, W, p = eigen_basis(self)
        A, U = optimize(self, E, W, p)
        self.A = A[0]
        if u:
            print('Calculate U.')
            V = interaction(self, U, W, p)
            self.U = V[0, 0]
            self.bands = band_bak
            return self.A, self.U
        else:
            self.bands = band_bak
            self.U = None
            return self.A

    def nn_tunneling(self, A: np.ndarray):
        # Pick up nearest neighbor tunnelings
        if self.Nsite == 1:
            nnt = np.zeros(1)
        elif self.lattice_dim == 1:
            nnt = np.diag(A, k=1)
        else:
            nnt = A[self.links[:, 0], self.links[:, 1]]
        return nnt

    def symm_unfold(self, target: Iterable, info, graph=False):
        # Unfold information to all symmetry sectors
        # No need to output as target is Iterable
        parity = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        for row in range(self.reflection.shape[0]):
            if graph:  # Symmetrize graph node coordinates
                target[self.reflection[row, :]] = parity * info[row][None]
            else:  # Symmetrize trap depth
                target[self.reflection[row, :]] = info[row]

    def symm_fold(self, info):
        # Extract information into symmetrized first sector
        target = info[self.reflection[:, 0]]
        return target

    def xy_links(self, nnt):
        # Distinguish x and y n.n. bonds and target t_x t_y values
        xlinks = abs(self.links[:, 0] - self.links[:, 1]) == 1
        ylinks = np.logical_not(xlinks)
        nntx = np.mean(abs(nnt[xlinks]))  # Find x direction links
        nnty = None
        # Find y direction links, if lattice is 1D this is nan
        if any(ylinks == True):
            nnty = np.mean(abs(nnt[ylinks]))
        return xlinks, ylinks, nntx, nnty


def eigen_basis(dvr: MLWF):
    # Find eigenbasis of symmetry block diagonalized Hamiltonian
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
        p_sb = np.array([], dtype=int).reshape(0, dim)
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

    E = [E_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite] for b in range(dvr.bands)]
    W = [W_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite] for b in range(dvr.bands)]
    parity = [
        p_sb[b * dvr.Nsite:(b + 1) * dvr.Nsite, :] for b in range(dvr.bands)
    ]

    return E, W, parity


def sector(dvr: MLWF):
    # Generate all sector information for 1D and 2D lattice
    # Single site case
    if dvr.Nsite == 1:
        p_tuple = [[1]]
    else:
        # x direction
        p_tuple = [[1, -1]]
        # y direction.
        if dvr.lattice_dim == 2:
            p_tuple.append([1, -1])
        else:
            p_tuple.append([1])
        # For a general omega_z << omega_x,y case,
        # the lowest several bands are in
        # z=1, z=-1, z=1 sector, etc... alternatively
        # A simplest way to build bands is to simply collect
        # Nband * Nsite lowest energy states
        # z direction
    if dvr.bands > 1 and dvr.dim == 3:
        # Only for 3D case there are z=-1 bands
        p_tuple.append([1, -1])
    else:
        p_tuple.append([1])
    p_list = list(itertools.product(*p_tuple))
    return p_list


def add_sector(sector: np.ndarray, dvr: MLWF, k: int, E, W, parity):
    # Add a symmetry sector to the list of eigensolutions
    p = dvr.p.copy()
    p[:len(sector)] = sector
    dvr.update_p(p)

    Em, Wm = H_solver(dvr, k)
    E = np.append(E, Em)
    W += [Wm[:, i].reshape(dvr.n + 1 - dvr.init) for i in range(k)]
    # Parity sector marker
    parity = np.append(parity, np.tile(sector, (k, 1)), axis=0)
    return E, W, parity


def cost_mat(dvr: MLWF, W, parity):
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


def multi_tensor(R):
    # Convert list of ndarray to list of Tensor
    return [
        torch.complex(torch.from_numpy(Ri),
                      torch.zeros(Ri.shape, dtype=torch.float64)) for Ri in R
    ]


def cost_func(U, R) -> float:
    # Cost function to Wannier optimize
    o = 0
    for i in range(len(R)):
        X = U.conj().T @ R[i] @ U
        Xp = X - torch.diag(torch.diag(X))
        o += torch.trace(torch.matrix_power(Xp, 2))
    return np.real(o)


def optimize(dvr: MLWF, E, W, parity):
    # Multiband optimization
    A = []
    w = []
    for b in range(dvr.bands):
        t_ij, w_mu = singleband_optimize(dvr, E[b], W[b], parity[b])
        A.append(t_ij)
        w.append(w_mu)
    return A, w


def singleband_optimize(dvr: MLWF, E, W, parity):
    # Singleband Wannier function optimization
    R = multi_tensor(cost_mat(dvr, W, parity))

    if dvr.Nsite > 1:
        manifold = pymanopt.manifolds.Unitaries(dvr.Nsite)

        @pymanopt.function.pytorch(manifold)
        def cost(point: torch.Tensor) -> float:
            return cost_func(point, R)

        problem = pymanopt.Problem(manifold=manifold, cost=cost, verbosity=1)
        solver = pymanopt.solvers.SteepestDescent(maxiter=3000)
        solution = solver.solve(problem, reuselinesearch=True)

        solution = positify(solution)
    elif dvr.Nsite == 1:
        solution = np.ones((1, 1))

    U = lattice_order(dvr, W, solution, parity)

    A = U.conj().T @ (
        E[:, None] *
        U) * dvr.V0 / dvr.kHz_2p  # TB parameter matrix, in unit of kHz
    return A, U


def lattice_order(dvr: MLWF, W, U, p):
    # Order Wannier functions by lattice site label
    shift = np.zeros((dvr.Nsite, dim))
    for i in range(dvr.Nsite):
        shift[i, :2] = dvr.trap_centers[i] * dvr.lc
        # Small offset to avoid zero values on z=0 in odd sector
        shift[i, 2] = 0.25
    x = [[[shift[i, d]] for d in range(dim)] for i in range(dvr.Nsite)]
    center_val = np.zeros((dvr.Nsite, dvr.Nsite))
    for i in range(dvr.Nsite):
        center_val[i, :] = abs(wannier_func(dvr, W, U, p, x[i]))
    # print(center_val)
    # Find at which trap max of Wannier func is
    order = np.argmax(center_val, axis=1)
    print('Trap site position of Wannier functions:', order)
    print('Order of Wannier function is set to match trap site.')
    return U[:, order]


def tight_binding(dvr: MLWF):
    E, W, parity = eigen_basis(dvr)
    A, w = optimize(dvr, E, W, parity)
    mu = np.diag(A)  # Diagonals are mu_i
    t = -(A - np.diag(mu))  # Off-diagonals are t_ij
    return np.real(mu), abs(t)


def interaction(dvr: MLWF, U: Iterable, W: Iterable, parity: Iterable):
    # Interaction between i band and j band
    Uint = np.zeros((dvr.bands, dvr.bands, dvr.Nsite))
    for i in range(dvr.bands):
        for j in range(dvr.bands):
            Uint[i, j, :] = singleband_interaction(dvr, U[i], U[j], W[i], W[j],
                                                   parity[i], parity[j])
    return Uint


def singleband_interaction(dvr: MLWF, Ui, Uj, Wi, Wj, pi: np.ndarray,
                           pj: np.ndarray):
    u = 4 * np.pi * dvr.hb * dvr.scatt_len / (dvr.m * dvr.kHz_2p * dvr.w**dim
                                              )  # Unit to kHz
    # # Construct interaction integral, assuming DVR quadrature this reduced to sum 'ijk,ijk,ijk,ijk'
    # integrl = np.zeros(dvr.Nsite * np.ones(4, dtype=int))
    # intgrl_mat(dvr, Wi, Wj, pi, pj, integrl)  # np.ndarray is global variable
    # # Bands are degenerate, only differed by spin index, so they share the same set of Wannier functions
    # Uint = contract('ia,jb,kc,ld,ijkl->abcd', Ui.conj(), Uj.conj(), Uj, Ui,
    #                 integrl)
    # Uint_onsite = np.zeros(dvr.Nsite)
    # for i in range(dvr.Nsite):
    #     if dvr.model == 'sho':
    #         print(
    #             f'Test with analytic calculation on {i + 1}-th site',
    #             np.real(Uint[i, i, i, i]) * (np.sqrt(2 * np.pi))**dvr.dim *
    #             np.prod(dvr.hl))
    #     Uint_onsite[i] = u * np.real(Uint[i, i, i, i])
    # return Uint_onsite

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
            f'Test with analytic calculation on {i + 1}-th site',
            np.real(Uint_onsite) * (np.sqrt(2 * np.pi))**dvr.dim *
            np.prod(dvr.hl))
    return u * Uint_onsite


def wannier_func(dvr, W, U, p, x: Iterable) -> np.ndarray:
    V = np.array([]).reshape(len(x[0]), len(x[1]), len(x[2]), 0)
    for i in range(p.shape[0]):
        V = np.append(V, psi(dvr.n, dvr.dx, W[i], *x, p[i, :]), axis=dim)
        # print(f'{i+1}-th Wannier function finished.')
    return V @ U


def intgrl3d(dx, integrand):
    for i in range(dim):
        if dx[i] > 0:
            integrand = romb(integrand, dx[i], axis=0)
        else:
            integrand = integrand[0, :]
    return integrand


# ######## DEPRECATED DUE TO WRONG ASSUMPTION OF QUADRATURE ##########
# def intgrl_mat(dvr, Wi, Wj, pi, pj, integrl):
#     # Construct interaction integral, assuming DVR quadrature this reduced to sum 'ijk,ijk,ijk,ijk'
#     for i in range(dvr.Nsite):
#         Wii = Wi[i].conj()
#         for j in range(dvr.Nsite):
#             Wjj = Wj[j].conj()
#             for k in range(dvr.Nsite):
#                 Wjk = Wj[k]
#                 for l in range(dvr.Nsite):
#                     Wil = Wi[l]
#                     if dvr.symmetry:
#                         p_ijkl = np.concatenate(
#                             (pi[i, :][None], pj[j, :][None], pj[k, :][None],
#                              pi[l, :][None]),
#                             axis=0)
#                         pqrs = np.prod(p_ijkl, axis=0)
#                         # Cancel n = 0 column for any p = -1 in one direction
#                         nlen = dvr.n.copy()
#                         # Mark which dimension has n=0 basis
#                         line0 = np.all(p_ijkl != -1, axis=0)
#                         nlen[line0] += 1
#                         # prefactor of n=0 line
#                         pref0 = np.zeros(dim)
#                         pref0[dvr.nd] = 1 / dvr.dx[dvr.nd]
#                         # prefactor of n>0 lines
#                         pref = (1 + pqrs) / 4 * pref0
#                         for d in range(dim):
#                             if dvr.nd[d]:
#                                 f = pref[d] * np.ones(Wil.shape[d])
#                                 if Wil.shape[d] > dvr.n[d]:
#                                     f[0] = pref0[d]
#                                 idx = np.ones(dim, dtype=int)
#                                 idx[d] = len(f)
#                                 f = f.reshape(idx)
#                                 Wil = f * Wil
#                         integrl[i, j, k, l] = contract('ijk,ijk,ijk,ijk',
#                                                        pick(Wii, nlen),
#                                                        pick(Wjj, nlen),
#                                                        pick(Wjk, nlen),
#                                                        pick(Wil, nlen))
#                     else:
#                         integrl[i, j, k, l] = contract('ijk,ijk,ijk,ijk', Wii,
#                                                        Wjj, Wjk, Wil)

# def pick(W: np.ndarray, nlen) -> np.ndarray:
#     # Truncate matrix to only n>0 columns
#     return W[-nlen[0]:, -nlen[1]:, -nlen[2]:]

# def parity_transfm(n: int):
#     # Parity basis transformation matrix: 1D version

#     Up = np.zeros((n + 1, 2 * n + 1))
#     Up[0, n] = 1  # n=d
#     Up[1:, :n] = np.eye(n)  # Negative half-axis
#     Up[1:, (n + 1):] = np.eye(n)  # Positive half-axis
#     Um = np.zeros((n, 2 * n + 1))
#     Um[:, n:] = -1 * np.eye(n)  # Negative half-axis
#     Um[:, :n] = np.eye(n)  # Positive half-axis
#     return Up, Um
