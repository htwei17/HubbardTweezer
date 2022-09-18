import numpy as np
from typing import Iterable
from opt_einsum import contract
from scipy.integrate import romb
from scipy.spatial.distance import cdist
from time import time
from itertools import permutations, product
import numpy.linalg as la

import torch
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers

from DVR.core import *
from .lattice import *


class MLWF(DVR):
    """Maximally localized Wannier function

    Args:
    ----------
        N (`int`): DVR grid size
        lattice (`np.ndarray[int, int]`): Lattice dimensions
        shape (`str`): Lattice geometry
        ascatt (`float`): Scattering length in unit of a0
        band (`int`): Number of bands
        dim (`int`): System dimension
        ...: See `DVR.__init__` for other arguments

    """

    def create_lattice(self,
                       lattice: np.ndarray,
                       lc: tuple[float, float] = (1520, 1690),
                       shape: str = "square"):
        # graph : each line represents coordinate (x, y) of one lattice site

        self.Nsite = np.prod(lattice)
        self.lattice_shape = shape

        # Convert [n] to [n, 1]
        if self.Nsite == 1:
            self.lattice = np.ones(1)
            self.lattice_dim = 1
        elif self.Nsite > 1 and lattice.size == 1:
            self.lattice = np.resize(
                np.pad(lattice, pad_width=(0, 1), constant_values=1), 2
            )
            self.lattice_dim = 1
        else:
            self.lattice = lattice.copy()
            if shape == 'ring':
                self.lattice_dim = 2
            else:
                self.lattice_dim = lattice[lattice > 1].size

        # Convert lc to (lc, lc) or the other if only one number is given
        if isinstance(lc, Iterable) and len(lc) == 1:
            lc = lc[0]
        if isinstance(lc, Number):
            if shape in ["triangular", "honeycomvb", "kagome"]:
                # For equilateral triangle
                lc = (lc, np.sqrt(3) / 2 * lc)
            else:
                # For squre
                lc = (lc, lc)

        self.tc0, self.links, self.reflection, self.inv_coords = lattice_graph(
            self.lattice, shape, self.ls)
        self.trap_centers = self.tc0.copy()
        self.Nsite = self.trap_centers.shape[0]

        # Independent trap number under reflection symmetry
        self.Nindep = self.reflection.shape[0] if self.ls else self.Nsite

        if self.model == "Gaussian":
            self.lc = np.array(lc) * 1e-9 / self.w  # In unit of wx
        elif self.model == "sho":
            self.lc = np.array(lc)

        dx = self.dx.copy()
        lattice_range = np.max(abs(self.trap_centers), axis=0)
        lattice_range = np.resize(
            np.pad(lattice_range, (0, 2), constant_values=0), dim)
        lc = np.resize(self.lc, dim)
        if self.verbosity:
            print(f"lattice: lattice shape is {shape}")
            print(f"lattice: Full lattice sizes: {lattice}")
            if self.verbosity > 1:
                print(f"lattice: lattice constants: {lc[:self.lattice_dim]}w")
                print(f"lattice: dx fixed to: {dx[self.nd]}w")
        # Let there be R0's wide outside the edge trap center
        R0 = lattice_range * lc + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def update_lattice(self, tc: np.ndarray):
        # Update DVR grids when trap centers are shifted

        self.trap_centers = tc.copy()
        dx = self.dx.copy()
        lattice = np.resize(
            np.pad(self.lattice, (0, 2), constant_values=1), dim)
        lc = np.resize(self.lc, dim)
        if self.verbosity:
            print(
                f"lattice: Full lattice sizes updated to: {lattice[self.nd]}")
            if self.verbosity > 1:
                # Let there be R0's wide outside the edge trap center
                print(f"lattice: lattice constants updated to: {lc}w")
                print(f"lattice: dx fixed to: {dx[self.nd]}w")
        R0 = (lattice - 1) * lc / 2 + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def __init__(
        self,
        N: int,
        lattice: np.ndarray = np.array(
            [2], dtype=int),  # Square lattice dimensions
        lc=(1520, 1690),  # Lattice constant, in unit of nm
        ascatt=1770,  # Scattering length, in unit of Bohr radius, default 1770
        shape="square",  # Shape of the lattice
        band=1,  # Number of bands
        lattice_symmetry: bool = True,  # Whether the lattice has reflection symmetry
        dim: int = 3,
        *args,
        **kwargs,
    ) -> None:

        self.N = N
        self.scatt_len = ascatt * a0
        self.dim = dim
        self.bands = band
        self.ls = lattice_symmetry
        n = np.zeros(3, dtype=int)
        n[:dim] = N
        absorber = kwargs.get('absorber', False)
        if absorber:
            raise TypeError(
                "Absorber is not supported for Wannier Function construction!")

        super().__init__(n, *args, **kwargs)
        # Backup of distance from edge trap center to DVR grid boundaries
        self.R00 = self.R0.copy()
        self.create_lattice(lattice, lc, shape)
        self.Voff = np.ones(self.Nsite)  # Set default trap offset
        # Set waist adjustment factor
        self.wxy0 = self.wxy.copy()
        self.waists = np.ones((self.Nsite, 2))

    def Vfun(self, x, y, z):
        # Get V(x, y, z) for the entire lattice
        V = 0

        if self.model == "sho" and self.Nsite == 2:
            # Two-site SHO case
            V += super().Vfun(abs(x) - self.lc[0] / 2, y, z)
        else:
            # NOTE: DO NOT SET coord DIRECTLY!
            # THIS WILL DIRECTLY MODIFY self.graph!
            for i in range(self.Nsite):
                shift = self.trap_centers[i] * self.lc
                self.update_waist(self.waists[i])
                V += self.Voff[i] * super().Vfun(x - shift[0], y - shift[1], z)
        return V

    def update_waist(self, waists):
        self.wxy = self.wxy0 * waists
        self.zR = np.pi * self.w * self.wxy**2 / self.l
        self.zR0: float = np.prod(self.zR) / la.norm(self.zR)
        self.omega = np.array([*(2 / self.wxy), 1 / self.zR0])
        self.omega *= np.sqrt(self.avg * self.hb *
                              self.V0 / self.m) / self.w

    def singleband_Hubbard(
        self, u=False, x0=None, offset=True, output_unitary=False, eig_sol=None
    ):

        # Calculate single band tij matrix and U matrix
        band_bak = self.bands
        self.bands = 1
        if eig_sol != None:
            E, W, p = eig_sol
        else:
            E, W, p = eigen_basis(self)
        E = E[0]
        W = W[0]
        p = p[0]
        self.A, V = singleband_optimize(self, E, W, p, x0)
        if offset:
            # Shift onsite potential to zero average
            self.A -= np.mean(np.real(np.diag(self.A))) * \
                np.eye(self.A.shape[0])

        if self.verbosity > 1:
            print(f'Energies: {E}')
            print(f'parities: {p}')

        if u:
            if self.verbosity:
                print("Calculate U.")
            self.U = singleband_interaction(self, V, V, W, W, p, p)
            self.bands = band_bak
            if output_unitary:
                return self.A, self.U, V
            else:
                return self.A, self.U
        else:
            self.bands = band_bak
            self.U = None
            if output_unitary:
                return self.A, V
            else:
                return self.A

    def nn_tunneling(self, A: np.ndarray):
        # Pick up nearest neighbor tunnelings
        # Not limited to specific geometry
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
                # NOTE: repeated nodes will be removed
                info[row][self.inv_coords[row]] = 0
                pinfo = parity * info[row][None]
                target[self.reflection[row, :]] = pinfo
            else:  # Symmetrize trap depth
                target[self.reflection[row, :]] = info[row]

    def symm_fold(self, info):
        # Extract information into symmetrized first sector
        target = info[self.reflection[:, 0]]
        return target

    def xy_links(self, nnt):
        # Distinguish x and y n.n. bonds and target t_x t_y values
        # FIXME: Only usable for rectangular & triangular latice.
        #        Ring lattice, etc.?
        if self.lattice_shape in ['square', 'triangular']:
            xlinks = abs(self.links[:, 0] - self.links[:, 1]) == 1
        else:
            xlinks = np.tile(True, self.links.shape[0])
        ylinks = np.logical_not(xlinks)
        nntx = np.mean(abs(nnt[xlinks]))  # Find x direction links
        nnty = None
        # Find y direction links, if lattice is 1D this is nan
        if any(ylinks == True):
            nnty = np.mean(abs(nnt[ylinks]))
        return xlinks, ylinks, nntx, nnty


def eigen_basis(dvr: MLWF) -> tuple[list, list, list]:
    # Find eigenbasis of symmetry block diagonalized Hamiltonian
    k = dvr.Nsite * dvr.bands
    if dvr.dvr_symm:
        p_list = sector(dvr)
        E_sb = np.array([])
        W_sb = []
        p_sb = np.array([], dtype=int).reshape(0, dim)
        for p in p_list:
            # print(f'Solve {p} sector.')
            E_sb, W_sb, p_sb = solve_sector(p, dvr, k, E_sb, W_sb, p_sb)

        # Sort everything by energy, only keetp lowest k states
        idx = np.argsort(E_sb)[:k]
        E_sb = E_sb[idx]
        W_sb = [W_sb[i] for i in idx]
        p_sb = p_sb[idx, :]
    else:
        p_sb = np.zeros((k, dim))
        E_sb, W_sb = dvr.H_solver(k)
        W_sb = [W_sb[:, i].reshape(2 * dvr.n + 1) for i in range(k)]

    E = [E_sb[b * dvr.Nsite: (b + 1) * dvr.Nsite] for b in range(dvr.bands)]
    W = [W_sb[b * dvr.Nsite: (b + 1) * dvr.Nsite] for b in range(dvr.bands)]
    parity = [p_sb[b * dvr.Nsite: (b + 1) * dvr.Nsite, :]
              for b in range(dvr.bands)]

    return E, W, parity


def sector(dvr: MLWF):
    # Generate all sector information for 1D and 2D lattice
    # Single site case
    p = [1, -1] if dvr.ls else [0]  # ls: lattice symmetry
    if dvr.Nsite == 1:
        p_tuple = [[1], [1]]  # x, y direction
    else:
        # x direction
        p_tuple = [p]
        # y direction
        if dvr.lattice_dim == 2:
            p_tuple.append(p)
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
    # Generate all possible combinations of xyz parity
    p_list = list(product(*p_tuple))
    return p_list


def solve_sector(sector: np.ndarray, dvr: MLWF, k: int, E, W, parity):
    # Add a symmetry sector to the list of eigensolutions
    p = dvr.p.copy()
    p[:len(sector)] = sector
    dvr.update_p(p)

    Em, Wm = dvr.H_solver(k)
    E = np.append(E, Em)
    W += [Wm[:, i].reshape(dvr.n + 1 - dvr.init) for i in range(k)]
    # Parity sector marker
    parity = np.append(parity, np.tile(sector, (k, 1)), axis=0)
    return E, W, parity


def Xmat(dvr: MLWF, W, parity):
    # Calculate X_ij = <i|x|j> for single-body eigenbasis |i>
    # and position operator x, y, z
    # NOTE: This is not the same as the X 'opterator', as DVR basis
    #       is not invariant subspace of X. So X depends on DBR basis choice.

    R = []

    # For 2D lattice band building keep single p_z = 1 or -1 sector,
    # Z is always zero. Explained below.
    for i in range(dim - 1):
        if dvr.nd[i]:
            Rx = np.zeros((dvr.Nsite, dvr.Nsite))
            idx = np.roll(np.arange(dim, dtype=int), -i)
            if dvr.ls:
                # Get X^pp'_ij matrix rep for Delta^p_i, p the parity.
                # This matrix is nonzero only when p!=p':
                # X^pp'_ij = x_i delta_ij with p_x * p'_x = -1
                # As 1 and -1 sector have a dimension difference,
                # the matrix X is alsways a n-diagonal matrix
                # with n-by-n+1 or n+1-by-n dimension
                # So, Z is always zero, as we only keep single p_z sector
                x = np.arange(1, dvr.n[i] + 1) * dvr.dx[i]
                lenx = len(x)
                for j in range(dvr.Nsite):
                    # If no absorber, W is real
                    # This cconjugate does not change dtype of real W
                    # It is only used in case of absorber
                    Wj = np.transpose(W[j], idx)[-lenx:, :, :].conj()
                    for k in range(j + 1, dvr.Nsite):
                        pjk = (
                            parity[j, idx[idx < dim - 1]]
                            * parity[k, idx[idx < dim - 1]]
                        )
                        if pjk[0] == -1 and pjk[1:] == 1:
                            Wk = np.transpose(W[k], idx)[-lenx:, :, :]
                            # Unitary transform X from DVR basis to single-body eigenbasis
                            Rx[j, k] = contract("ijk,i,ijk", Wj, x, Wk)
                            # Rx is also real if no absorber
                            # So Rx is real-symmetric or hermitian
                            Rx[k, j] = Rx[j, k].conj()
            else:
                # X = x_i delta_ij for non-symmetrized basis
                x = np.arange(-dvr.n[i], dvr.n[i] + 1) * dvr.dx[i]
                for j in range(dvr.Nsite):
                    Wj = np.transpose(W[j], idx).conj()
                    Rx[j, j] = contract("ijk,i,ijk", Wj, x, Wj.conj())
                    for k in range(j + 1, dvr.Nsite):
                        Wk = np.transpose(W[k], idx)
                        Rx[j, k] = contract("ijk,i,ijk", Wj, x, Wk)
                        Rx[k, j] = Rx[j, k].conj()
            R.append(Rx)
    return R


# ========================== OPTIMIZATION ALGORITHMS ==========================


def cost_func(U: torch.Tensor, R: list) -> torch.Tensor:
    # Cost function to Wannier optimize
    o = 0
    for i in range(len(R)):
        # R is real-symmetric if no absorber
        X = U.conj().T @ R[i] @ U
        Xp = X - torch.diag(torch.diag(X))
        o += torch.trace(torch.matrix_power(Xp, 2))
    # X must be hermitian, so is Xp
    # Xp^2 is then positive and hermitian,
    # its diagonal is real and positive, o >= 0
    # Min is found when X diagonal, which means U diagonalize R
    # SO U can be pure real orthogonal matrix!
    # Q: Can X, Y, Z be diagonalized simultaneously in high dims?
    # A: If the space is conplete then by QM theory it is possible
    #    to diagonalize X, Y, Z simultaneously.
    #    But this is not the case as it's a subspace.
    return o.real


def optimize(dvr: MLWF, E, W, parity, offset=True):
    # Multiband optimization
    A = []
    w = []
    for b in range(dvr.bands):
        t_ij, w_mu = singleband_optimize(dvr, E[b], W[b], parity[b])
        if b == 0:
            # Shift onsite potential to zero average
            # Multi-band can only be shifted globally by 1st band
            if offset:
                zero = np.mean(np.real(np.diag(t_ij)))
            else:
                zero = 0
        A.append(t_ij - zero * np.eye(t_ij.shape[0]))
        w.append(w_mu)
    return A, w


def singleband_optimize(dvr: MLWF, E, W, parity, x0=None, eig1d: bool = True) -> tuple[np.ndarray, np.ndarray]:
    # Singleband Wannier function optimization
    # x0 is the initial guess

    t0 = time()

    if dvr.Nsite > 1:
        R = Xmat(dvr, W, parity)
        if dvr.lattice_dim == 1 and eig1d:
            # If only one R given, the problem is simply diagonalization
            # solution is eigenstates of operator X
            X, solution = la.eigh(R[0])
            # Auto sort eigenvectors by X eigenvalues
            U = solution[:, np.argsort(X)]
        else:
            # In high dimension, X, Y, Z don't commute
            # Convert list of ndarray to list of Tensor
            R = [torch.from_numpy(Ri) for Ri in R]
            solution = riemann_optimize(dvr, x0, R)
            U = site_order(dvr, solution, R)
    else:
        U = np.ones((1, 1))

    A = U.conj().T @ (E[:, None] * U) * dvr.V0 / dvr.kHz_2p
    # TB parameter matrix, in unit of kHz

    t1 = time()
    if dvr.verbosity:
        print(f"Single band optimization time: {t1 - t0}s.")

    return A, U


def riemann_optimize(dvr: MLWF, x0, R: list) -> np.ndarray:
    # It's proven above that U can be purely real
    # TODO: DOUBLE CHECK is all real condition still valid for the subspace?
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(dvr.Nsite)

    @pymanopt.function.pytorch(manifold)
    def _cost_func(point: torch.Tensor) -> torch.Tensor:
        return cost_func(point, R)

    problem = pymanopt.Problem(manifold=manifold, cost=_cost_func)
    optimizer = pymanopt.optimizers.ConjugateGradient(
        max_iterations=1000, min_step_size=1e-12, verbosity=dvr.verbosity if dvr.verbosity <= 2 else 2)
    result = optimizer.run(
        problem, initial_point=x0, reuse_line_searcher=True)
    solution = result.point
    return solution

# =============================================================================


def site_order(dvr: MLWF, U: np.ndarray, R: list[torch.Tensor]) -> np.ndarray:
    # Order Wannier functions by lattice site label
    # FIXME: When traps are very close,
    #        optimized Wannier functions may not be localized on sites

    if dvr.lattice_dim == 1:
        # Find WF center of mass
        x = np.diag(U.T @ R[0].numpy() @ U) / dvr.lc[0]
        order = np.argsort(x)
    elif dvr.lattice_dim > 1:
        # Find WF center of mass
        x = np.array([np.diag(U.T @ R[i].numpy() @ U) / dvr.lc[i]
                     for i in range(dvr.lattice_dim)]).T
        order = nearest_match(dvr, x)
    if dvr.verbosity > 1:
        print("Trap site position of Wannier functions:", order)
        print("Order of Wannier functions is set to match traps.")
    return U[:, order]


def nearest_match(dvr: MLWF, x: np.ndarray) -> np.ndarray:
    # Match Wannier functions to nearest trap sites
    # FIXME: Use a better algorithm to locate more accurate and efficient
    #        An idea is to list all possible permutations
    #        and find the one with least dist.
    #        But this is slow as the number of permutations is factorial

    # i-th row is the distance of i-th site to each WFs
    dist_mat = cdist(dvr.trap_centers, x, metric="euclidean")
    # # dist_mat is a square matrix, result shall be a full rank matrix
    # # if trap centers and WFs are 1-to-1 corresponded.
    # if la.matrix_rank(dist_mat <= 1/2) < dvr.Nsite:
    #     print("WARNING: Wannier functions not localized on sites!")
    # order = np.zeros(dvr.Nsite, dtype=int)
    # for i in range(dvr.Nsite):
    #     # Find unused site index that is closest to i-th WF
    #     order[i] = np.where(dist_mat[i] == np.min(
    #         dist_mat[i, np.delete(np.arange(dvr.Nsite), order[:i])]))[0]

    # Two methods time differs ~1000x
    # Naive method is ~75µs per call
    # Permutation method is ~150ms per call
    # If this func is evalauted ~1000 times, it's a considerable time
    perm = np.array(tuple(permutations(range(dvr.Nsite))))
    odx = np.arange(dvr.Nsite)
    idx = np.argmin(tuple(np.sum(dist_mat[odx, order]**2) for order in perm))
    order = perm[idx]
    return order


# def tight_binding(dvr: MLWF):
#     E, W, parity = eigen_basis(dvr)
#     A, w = optimize(dvr, E, W, parity)
#     mu = np.diag(A)  # Diagonals are mu_i
#     t = -(A - np.diag(mu))  # Off-diagonals are t_ij
#     return np.real(mu), abs(t)


def interaction(dvr: MLWF, U: Iterable, W: Iterable, parity: Iterable):
    # Interaction between i band and j band
    Uint = np.zeros((dvr.bands, dvr.bands, dvr.Nsite))
    for i in range(dvr.bands):
        for j in range(dvr.bands):
            Uint[i, j, :] = singleband_interaction(
                dvr, U[i], U[j], W[i], W[j], parity[i], parity[j]
            )
    return Uint


def singleband_interaction(dvr: MLWF, Ui, Uj, Wi, Wj, pi: np.ndarray, pj: np.ndarray):

    t0 = time()

    u = (
        4 * np.pi * dvr.hb * dvr.scatt_len / (dvr.m * dvr.kHz_2p * dvr.w**dim)
    )  # Unit to kHz
    # # Construct interaction integral,
    # # assuming DVR quadrature this reduced to sum 'ijk,ijk,ijk,ijk'
    # integrl = np.zeros(dvr.Nsite * np.ones(4, dtype=int))
    # intgrl_mat(dvr, Wi, Wj, pi, pj, integrl)  # np.ndarray is global variable
    # # Bands are degenerate, only differed by spin index,
    # $ so they share the same set of Wannier functions
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
    wannier = abs(Vi) ** 2 * abs(Vj) ** 2
    Uint_onsite = intgrl3d(wannier, dx)
    if dvr.model == "sho":
        print(
            f"Test with analytic calculation on {i + 1}-th site",
            np.real(Uint_onsite) * (np.sqrt(2 * np.pi)
                                    ) ** dvr.dim * np.prod(dvr.hl),
        )

    t1 = time()
    if dvr.verbosity:
        print(f"Single band interaction time: {t1 - t0}s.")

    return u * Uint_onsite


def wannier_func(dvr: MLWF, W, U, p: np.ndarray, x: Iterable) -> np.ndarray:
    x = [np.array([x[i]]) if isinstance(x[i], Number) else x[i]
         for i in range(dim)]
    V = np.zeros((*(len(x[i]) for i in range(dim)), p.shape[0]))
    for i in range(p.shape[0]):
        V[:, :, :, i] = psi(dvr.n, dvr.dx, W[i], x, p[i, :])[..., 0]
        # print(f'{i+1}-th Wannier function finished.')
    return V @ U


def intgrl3d(integrand: np.ndarray, dx: list[float, float, float]) -> float:
    for i in range(dim):
        if dx[i] > 0:
            integrand = romb(integrand, dx[i], axis=0)
        else:
            integrand = integrand[0]
    return integrand
