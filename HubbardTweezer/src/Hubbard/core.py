from pyclbr import Function
import numpy as np
from typing import Iterable
from opt_einsum import contract
from time import time
from itertools import product, permutations
import numpy.linalg as la

from .optimize import riemann_optimize
from .lattice import *
from DVR.core import *
from DVR.wavefunc import psi
from tools.integrate import romb3d
from tools.point_match import nearest_match


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
            lc: Number = lc[0]
        if isinstance(lc, Number):
            if shape in ["triangular", "honeycomvb", "kagome"]:
                # For equilateral triangle
                lc: tuple = (lc, np.sqrt(3) / 2 * lc)
            else:
                # For squre
                lc: tuple = (lc, lc)

        self.tc0, self.links, self.reflection, self.inv_coords = lattice_graph(
            self.lattice, shape, self.ls)
        self.trap_centers = self.tc0.copy()
        self.wf_centers = self.tc0.copy()  # Assume WF are localized at trap centers
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
        self, u=False, x0=None, offset=True, eig_sol=None
    ):

        # Calculate single band tij matrix and U matrix
        band_bak = self.bands
        self.bands = 1
        if eig_sol != None:
            E, W, p = eig_sol
        else:
            E, W, p = self.eigen_basis()
        E = E[0]
        W = W[0]
        p = p[0]
        self.A, V = singleband_optimize(self, E, W, p, x0)
        if offset:
            # Shift onsite potential to zero average
            self.A -= np.mean(np.real(np.diag(self.A))) * \
                np.eye(self.A.shape[0])

        if u and self.verbosity:
            print("Calculate U.")
        self.U = singleband_interaction(self, V, V, W, W, p, p) if u else None
        self.bands = band_bak
        return self.A, self.U, V

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

    def xylinks(self):
        # Distinguish x and y n.n. bonds and target t_x t_y values
        # FIXME: Check all possible cases
        if self.lattice_shape in ['square', 'Lieb', 'triangular', 'honeycomb', 'kagome']:
            xlinks = abs(self.links[:, 0] - self.links[:, 1]) == 1
        else:
            xlinks = np.tile(True, self.links.shape[0])
        ylinks = np.logical_not(xlinks)
        return xlinks, ylinks

    def t_target(self, nnt, links, func: Function = np.mean):
        xlinks, ylinks = links
        nntx = func(abs(nnt[xlinks]))  # Find x direction links
        # Find y direction links, if lattice is 1D this is nan
        nnty = func(abs(nnt[ylinks])) if any(ylinks == True) else None
        return nntx, nnty

    # TODO: Integrate multisector solver with DVR
    def solve_sector(self, sector: np.ndarray, k: int, E, W, parity):
        # Add a symmetry sector to the list of eigensolutions
        p = self.p.copy()
        p[:len(sector)] = sector
        self.update_p(p)

        Em, Wm = self.H_solver(k)
        E = np.append(E, Em)
        W += [Wm[:, i].reshape(self.n + 1 - self.init) for i in range(k)]
        # Parity sector marker
        parity = np.append(parity, np.tile(sector, (k, 1)), axis=0)
        return E, W, parity

    def build_sectors(self):
        # Generate all sector information for 1D and 2D lattice
        # Single site case
        p = [1, -1] if self.ls else [0]  # ls: lattice symmetry
        if self.Nsite == 1:
            p_tuple = [[1], [1]]  # x, y direction
        else:
            # x direction
            p_tuple = [p]
            # y direction
            if self.lattice_dim == 2:
                p_tuple.append(p)
            else:
                p_tuple.append([1])
            # For a general omega_z << omega_x,y case,
            # the lowest several bands are in
            # z=1, z=-1, z=1 sector, etc... alternatively
            # A simplest way to build bands is to simply collect
            # Nband * Nsite lowest energy states
            # z direction
        if self.bands > 1 and self.dim == 3:
            # Only for 3D case there are z=-1 bands
            p_tuple.append([1, -1])
        else:
            p_tuple.append([1])
        # Generate all possible combinations of xyz parity
        p_list = list(product(*p_tuple))
        return p_list

    def eigen_basis(self) -> tuple[list, list, list]:
        # Find eigenbasis of symmetry block diagonalized Hamiltonian
        k = self.Nsite * self.bands
        if self.dvr_symm:
            p_list = self.build_sectors()
            E_sb = np.array([])
            W_sb = []
            p_sb = np.array([], dtype=int).reshape(0, dim)
            for p in p_list:
                # print(f'Solve {p} sector.')
                E_sb, W_sb, p_sb = self.solve_sector(p, k, E_sb, W_sb, p_sb)

            # Sort everything by energy, only keetp lowest k states
            idx = np.argsort(E_sb)
            if self.verbosity > 2:
                print(f'Energies: {E_sb[idx]}')
                print(f'parities: {p_sb[idx, :]}')
            idx = idx[:k]
            E_sb = E_sb[idx]
            W_sb = [W_sb[i] for i in idx]
            p_sb = p_sb[idx, :]
        else:
            p_sb = np.zeros((k, dim))
            E_sb, W_sb = self.H_solver(k)
            W_sb = [W_sb[:, i].reshape(2 * self.n + 1) for i in range(k)]

        E = [E_sb[b * self.Nsite: (b + 1) * self.Nsite]
             for b in range(self.bands)]
        W = [W_sb[b * self.Nsite: (b + 1) * self.Nsite]
             for b in range(self.bands)]
        parity = [p_sb[b * self.Nsite: (b + 1) * self.Nsite, :]
                  for b in range(self.bands)]

        return E, W, parity

    def Xmat(self, W, parity):
        # Calculate X_ij = <i|x|j> for single-body eigenbasis |i>
        # and position operator x, y, z
        # NOTE: This is not the same as the X 'opterator', as DVR basis
        #       is not invariant subspace of X. So X depends on DBR basis choice.
        R = []
        # For 2D lattice band building keep single p_z = 1 or -1 sector,
        # Z is always zero. Explained below.
        for i in range(dim - 1):
            if self.nd[i]:
                R.append(self.Xmat_1d(W, parity, i))
        return R

    def Xmat_1d(self, W, parity, i):
        Rx = np.zeros((self.Nsite, self.Nsite))
        idx = np.roll(np.arange(dim, dtype=int), -i)
        if self.ls:
            # Get X^pp'_ij matrix rep for Delta^p_i, p the parity.
            # This matrix is nonzero only when p!=p':
            # X^pp'_ij = x_i delta_ij with p_x * p'_x = -1
            # As 1 and -1 sector have a dimension difference,
            # the matrix X is alsways a n-diagonal matrix
            # with n-by-n+1 or n+1-by-n dimension
            # So, Z is always zero, as we only keep single p_z sector
            x = np.arange(1, self.n[i] + 1) * self.dx[i]
            lenx = len(x)
            for j in range(self.Nsite):
                # If no absorber, W is real
                # This cconjugate does not change dtype of real W
                # It is only used in case of absorber
                Wj = np.transpose(W[j], idx)[-lenx:, :, :].conj()
                for k in range(j + 1, self.Nsite):
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
            x = np.arange(-self.n[i], self.n[i] + 1) * self.dx[i]
            for j in range(self.Nsite):
                Wj = np.transpose(W[j], idx).conj()
                Rx[j, j] = contract("ijk,i,ijk", Wj, x, Wj.conj())
                for k in range(j + 1, self.Nsite):
                    Wk = np.transpose(W[k], idx)
                    Rx[j, k] = contract("ijk,i,ijk", Wj, x, Wk)
                    Rx[k, j] = Rx[j, k].conj()
        return Rx

# ========================== OPTIMIZATION ALGORITHMS ==========================


def singleband_optimize(dvr: MLWF, E, W, parity, x0=None, eig1d: bool = True) -> tuple[np.ndarray, np.ndarray]:
    # Singleband Wannier function optimization
    # x0 is the initial guess

    t0 = time()
    if dvr.Nsite > 1:
        R = dvr.Xmat(W, parity)
        if dvr.lattice_dim == 1 and eig1d:
            # If only one R given, the problem is simply diagonalization
            # solution is eigenstates of operator X
            X, solution = la.eigh(R[0])
            # Auto sort eigenvectors by X eigenvalues
            order = np.argsort(X)
            U = solution[:, order]
            wf_centers = np.array([X[order], np.zeros_like(X[order])]).T
        else:
            # In high dimension, X, Y, Z don't commute
            solution = riemann_optimize(R, x0, dvr.verbosity)
            U = site_order(dvr, solution, R)
            wf_centers = np.array([np.diag(U.conj().T @ R[i] @ U) / dvr.lc[i]
                                   for i in range(dvr.lattice_dim)]).T
    else:
        U = np.ones((1, 1))
        wf_centers = np.zeros((1, 2))

    dvr.wf_centers = wf_centers
    A = U.conj().T @ (E[:, None] * U) * dvr.V0 / dvr.kHz_2p
    # TB parameter matrix, in unit of kHz
    t1 = time()
    if dvr.verbosity:
        print(f"Single band optimization time: {t1 - t0}s.")
    return A, U


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


# =============================================================================


def site_order(dvr: MLWF, U: np.ndarray, R: list[np.ndarray]) -> np.ndarray:
    # Order Wannier functions by lattice site label
    # FIXME: When traps are very close,
    #        optimized Wannier functions may not be localized on sites

    if dvr.lattice_dim == 1:
        # Find WF center of mass
        x = np.diag(U.conj().T @ R[0] @ U) / dvr.lc[0]
        order = np.argsort(x)
    elif dvr.lattice_dim > 1:
        # Find WF center of mass
        x = np.array([np.diag(U.conj().T @ R[i] @ U) / dvr.lc[i]
                     for i in range(dvr.lattice_dim)]).T
        order = nearest_match(dvr.trap_centers, x)
    if dvr.verbosity > 1:
        print("Trap site position of Wannier functions:", order)
        print("Order of Wannier functions is set to match traps.")
    return U[:, order]


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
    x = []
    dx = []
    for i in range(dim):
        if dvr.nd[i]:
            x.append(np.linspace(-1.2 * dvr.R0[i], 1.2 * dvr.R0[i], 129))
            dx.append(x[i][1] - x[i][0])
        else:
            x.append(np.array([0]))
            dx.append(0)
    Vi = wannier_func(x, Ui, dvr, Wi, pi)
    Vj = wannier_func(x, Uj, dvr, Wj, pj)
    wannier = abs(Vi) ** 2 * abs(Vj) ** 2
    Uint_onsite = romb3d(wannier, dx)
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


def wannier_func(x: Iterable, U, dvr: MLWF, W, p: np.ndarray) -> np.ndarray:
    x = [np.array([x[i]]) if isinstance(x[i], Number) else x[i]
         for i in range(dim)]
    V = np.zeros((*(len(x[i]) for i in range(dim)), p.shape[0]))
    for i in range(p.shape[0]):
        V[:, :, :, i] = psi(x, dvr.n, dvr.dx, W[i], p[i, :])[..., 0]
        # print(f'{i+1}-th Wannier function finished.')
    return V @ U


def symm_fold(reflection, info):
    # Extract information into symmetrized first sector
    target = info[reflection[:, 0]]
    return target
