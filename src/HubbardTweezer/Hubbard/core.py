import numpy as np
from numpy.linalg import LinAlgError
from typing import Iterable, Union, Optional, Callable
from numbers import Number
from opt_einsum import contract
from time import time
from itertools import product
import numpy.linalg as la
import scipy.interpolate as interp

from ..DVR import DVR
from ..DVR.const import *
from ..DVR.wavefunc import psi
from ..tools.integrate import romb3d, trapz3dnp
from ..tools.point_match import nearest_match

from .riemann import *
from .lattice import Lattice


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

    lattice: Lattice

    Nintgrl_grid: int = 257
    custom_potential: Callable = None  # Custom potential function, if any
    # Rintgrl: np.ndarray

    wf_centers: np.ndarray
    wf_cost: float

    zero_avgV: bool = True

    def update_lattice(self, tc: np.ndarray):
        # Update DVR grids when trap centers are shifted

        self.lattice.trap_centers = tc.copy()
        dx = self.dx.copy()
        grid = np.resize(np.pad(self.lattice.grid.size, (0, 2), constant_values=1), DIM)
        lc = np.resize(self.lattice.lc, DIM)
        if self.verbosity:
            print(f"Lattice: Full lattice sizes updated to: {grid[self.nd]}")
            if self.verbosity > 1:
                # Let there be R0's wide outside the edge trap center
                print(f"Lattice: lattice constants updated to: {lc}w")
                print(f"Lattice: dx fixed to: {dx[self.nd]}w")
        R0 = (grid - 1) * lc / 2 + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def update_waist(self, waists):
        self.wxy = self.wxy0 * waists
        self.zR = np.pi * self.w * self.wxy**2 / self.l
        self.zR0: float = np.prod(self.zR) / la.norm(self.zR)
        self.omega = np.array([*(2 / self.wxy), 1 / self.zR0])
        self.omega *= np.sqrt(self.avg * self.hb * self.V0 / self.m) / self.w

    def __init__(
        self,
        N: int,
        lattice: Lattice,  # Lattice object containing lattice parameters
        custom_potential: Optional[
            Union[Callable, tuple[Iterable, np.ndarray]]
        ] = None,  # Custom potential function, if any
        ascatt: float = 1770,  # Scattering length, in unit of Bohr radius, default 1770
        band=1,  # Number of bands
        balance_V0: bool = False,  # Balance trap depths V0 for all traps first, useful for two-band calculation
        dim: int = 3,
        *args,
        **kwargs,
    ) -> None:
        self.N = N
        self.scatt_len = ascatt * A0
        self.dim = dim
        self.bands = band

        n = np.zeros(3, dtype=int)
        n[:dim] = N

        absorber = kwargs.get("absorber", False)
        if absorber:
            raise TypeError(
                "Absorber is not supported for Wannier Function construction!"
            )

        self.zero_avgV = kwargs.pop("zero_avgV", True)

        self.Nintgrl_grid = kwargs.get(
            "Nintgrl_grid", 257
        )  # Numerical integration grid point number
        print(f"Wannier: Number of integration grid set to {self.Nintgrl_grid}.")

        model = kwargs.get("model", "Gaussian")
        if model == "custom":
            print("Wannier: Custom potential model is set. Ignore lattice parameters.")
            if custom_potential is not None:
                if isinstance(custom_potential, Iterable):
                    self.custom_potential = interp.RegularGridInterpolator(
                        custom_potential[0], custom_potential[1]
                    )
                elif isinstance(custom_potential, Callable):
                    self.custom_potential = custom_potential
                else:
                    raise TypeError(
                        "Invalid custom potential type. The accepted types are callable or tuple of (grid, values)."
                    )

        super().__init__(n, *args, **kwargs)
        # Backup of distance from edge trap center to DVR grid boundaries
        self.R00 = self.R0.copy()

        self.lattice = lattice
        # Set lattice constants in unit of wx
        if self.model in ["Gaussian", "optical_lattice"]:
            self.lattice.set_lc(
                np.array(self.lattice.lc) * 1e-9 / self.w, self.lattice.shape
            )
        elif self.model == "sho":
            self.lattice.set_lc(np.array(self.lattice.lc), self.lattice.shape)

        self.wf_centers = self.lattice.tc0.copy()

        dx = self.dx.copy()
        lattice_range = np.max(abs(self.lattice.tc0), axis=0)
        lattice_range = np.resize(
            np.pad(lattice_range, (0, 2), constant_values=0), self.dim
        )
        if self.verbosity:
            print(f"Lattice: lattice shape is {self.lattice.shape}")
            print(f"Lattice: Full lattice sizes: {self.lattice.size}")
            if self.verbosity > 1:
                print(f"Lattice: lattice constants: {self.lattice.lc}w")
                print(f"Lattice: dx fixed to: {dx[self.nd]}w")
        # Let there be R0's wide outside the edge trap center
        R0 = lattice_range + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

        self.Voff = np.ones(self.lattice.N)  # Set default trap offset
        # Set waist adjustment factor
        self.wxy0 = self.wxy.copy()
        self.waists = np.ones((self.lattice.N, 2))

        # Balance trap depth first, to make sure traps won't go too uneven
        # to have non-local WF. But this makes U to be more uneven.
        if balance_V0:
            self.balance_trap_depths()

        # Set to cancel onsite potential offset, quantities are of no use
        # They will be overwritten in HubbardEqualizer

    def Vfun(self, x, y, z):
        # Get V(x, y, z) for the entire lattice
        V = 0

        if self.model == "sho" and self.lattice.N == 2:
            # Two-site SHO case
            V += super().Vfun(abs(x) - self.lattice.lc[0] / 2, y, z)
        elif self.model == "optical_lattice":
            # Optical lattice potential in 2D
            V = (
                np.cos(2 * np.pi * x / self.lattice.lc[0])
                + np.cos(2 * np.pi * y / self.lattice.lc[1])
                - 2
            ) / 2
        elif self.model == "custom" and self.custom_potential is not None:
            # Custom potential case
            V += self.custom_potential(x, y, z)
        else:
            # NOTE: DO NOT SET coord DIRECTLY!
            # THIS WILL DIRECTLY MODIFY self.graph!
            for i in range(self.lattice.N):
                shift = self.lattice.trap_centers[i]
                self.update_waist(self.waists[i])
                V += self.Voff[i] * super().Vfun(x - shift[0], y - shift[1], z)
        return V

    def singleband_Hubbard(self, u=False, x0=None, W0=None, band=1, eig_sol=None):
        # Calculate single band tij matrix and U matrix
        band_bak = self.bands
        if band == 1:
            self.bands = 1
        if eig_sol != None:
            E, W, p = eig_sol
        else:
            E, W, p = self.eigen_basis(W0=W0)
        E = E[band - 1]  # Eigen energy
        W = W[band - 1]  # Eigen vector
        p = p[band - 1]  # Sector index
        self.A, WF = self.singleband_WF(E, W, p, x0)
        if self.zero_avgV is True:
            # Shift onsite potential to zero average
            self.zero = np.mean(np.real(np.diag(self.A)[self.lattice.mask]))
        elif isinstance(self.zero_avgV, Number):
            self.zero = self.zero_avgV
        else:
            self.zero = 0
        self.A -= self.zero * np.eye(self.A.shape[0])

        if u and self.verbosity:
            print("Calculate U.")
        self.U = singleband_interaction(self, WF, WF, W, W, p, p) if u else None
        self.bands = band_bak
        return self.A, self.U, WF

    def trap_mat(self):
        # depth of each trap center
        tc = np.zeros((self.lattice.N, DIM))
        vij = np.ones((self.lattice.N, self.lattice.N))
        for i in range(self.lattice.N):
            tc[i, :] = np.append(self.lattice.trap_centers[i], 0)
            for j in range(i):
                vij[i, j] = -DVR.Vfun(self, *(tc[i] - tc[j]))
                vij[j, i] = vij[i, j]  # Potential is symmetric in distance
        return vij

    def balance_trap_depths(self):
        vij = self.trap_mat()
        # Set trap depth target to be the deepest one
        Vtarget = np.max(vij @ np.ones(self.lattice.N))
        try:
            # Balance trap depth
            # Powered to compensate for trap unevenness
            self.Voff = la.solve(vij, Vtarget * np.ones(self.lattice.N)) ** 2
            if self.verbosity:
                print(f"Balance: trap depths balanced to {self.Voff}.")
        except:
            raise LinAlgError("Homogenize: failed to solve for Voff.")

    # TODO: Integrate multisector solver with DVR
    def solve_sector(self, sector: np.ndarray, k: int, E, W, parity, v0):
        # Add a symmetry sector to the list of eigensolutions
        p = self.p.copy()
        p[: len(sector)] = sector
        self.update_p(p)

        Em, Wm = self.H_solver(k, v0)
        E = np.append(E, Em)
        W += [Wm[:, i].reshape(self.n + 1 - self.init) for i in range(k)]
        # Parity sector marker
        parity = np.append(parity, np.tile(sector, (k, 1)), axis=0)
        return E, W, parity

    def build_sectors(self):
        # Generate all sector information for 1D and 2D lattice
        # Single site case
        p = [1, -1] if self.lattice.ls else [0]  # ls: lattice symmetry
        if self.lattice.N == 1:
            p_tuple = [[1], [1]]  # x, y direction
        else:
            # x direction
            p_tuple = [p]
            # y direction
            if self.lattice.dim == 2:
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

    def eigen_basis(
        self, W0: list = None, band_std: str = "symmetry"
    ) -> tuple[list, list, list]:
        # Find eigenbasis of symmetry block diagonalized Hamiltonian
        band_site = self.lattice.N
        k = band_site * self.bands
        if self.dvr_symm:
            p_list = self.build_sectors()
            E_sb = np.array([])
            W_sb = []
            p_sb = np.array([], dtype=int).reshape(0, DIM)
            if W0 is not None:  # Pad W0 to match p_list
                W0.extend([None] * (len(p_list) - len(W0)))
            for pidx in range(len(p_list)):
                p = p_list[pidx]
                # print(f'Solve {p} sector.')
                if W0 is None:
                    W0p = None
                else:
                    W0p = W0[pidx]
                E_sb, W_sb, p_sb = self.solve_sector(p, k + 1, E_sb, W_sb, p_sb, W0p)
                if W0 is not None:
                    W0[pidx] = W_sb[-k - 1]  # Inplace update x,y,z-folded W0

            if band_std == "energy":
                # Sort everything by energy, only keetp lowest k states
                idx = np.argsort(E_sb)[: k + 1]
                E_sb = E_sb[idx]
                W_sb = [W_sb[i] for i in idx[:k]]
                p_sb = p_sb[idx, :]
            elif band_std == "symmetry":
                idx = np.argsort(E_sb)
                E_sb = E_sb[idx]
                W_sb = [W_sb[i] for i in idx]
                p_sb = p_sb[idx, :]
        else:
            p_sb = np.zeros((k, DIM))
            E_sb, W_sb = self.H_solver(k + 1)
            W_sb = [W_sb[:, i].reshape(2 * self.n + 1) for i in range(k)]

        if self.verbosity > 2:
            print(f"Energies: {E_sb}")
            if self.lattice.ls:
                print(f"parities: {[p_sb]}")
        # elif self.verbosity > 1 and E_sb[k - 1] - E_sb[0] > E_sb[k] - E_sb[k - 1]:
        #     print("Wannier warning: band gap is smaller than band width.")

        if band_std == "symmetry" and self.bands == 1:
            # Sector already be limited to z=1
            band_std = "energy"

        if band_std == "energy":
            E_sb = E_sb[:k]
            p_sb = p_sb[:k]
            E = [E_sb[b * band_site : (b + 1) * band_site] for b in range(self.bands)]
            W = [W_sb[b * band_site : (b + 1) * band_site] for b in range(self.bands)]
            parity = [
                p_sb[b * band_site : (b + 1) * band_site, :] for b in range(self.bands)
            ]
        elif band_std == "symmetry" and self.bands == 2:
            # Hand coded pz-even and pz-odd bands
            E_even = np.array([])
            E_odd = np.array([])
            W_even = []
            W_odd = []
            parity_even = np.array([], dtype=int).reshape(0, DIM)
            parity_odd = np.array([], dtype=int).reshape(0, DIM)
            count_even = 0
            count_odd = 0
            for pidx in range(len(p_sb)):
                p = p_sb[pidx]
                if p[2] == 1 and count_even < band_site:
                    E_even = np.append(E_even, E_sb[pidx])
                    W_even.append(W_sb[pidx])
                    parity_even = np.append(parity_even, p[None], axis=0)
                    count_even += 1
                elif p[2] == -1 and count_odd < band_site:
                    E_odd = np.append(E_odd, E_sb[pidx])
                    W_odd.append(W_sb[pidx])
                    parity_odd = np.append(parity_odd, p[None], axis=0)
                    count_odd += 1
            E = [E_even, E_odd]
            W = [W_even, W_odd]
            parity = [parity_even, parity_odd]
        else:
            raise ValueError("Invalid band forming standard.")
        # TODO: add multi-band support for symmetry standard
        return E, W, parity

    def Xmat(self, W, parity):
        # Calculate X_ij = <i|x|j> for single-body eigenbasis |i>
        # and position operator x, y, z
        # NOTE: This is not the same as the X 'opterator', as DVR basis
        #       is not invariant subspace of X. So X depends on DBR basis choice.
        R = []
        # For 2D lattice keeps single p_z = 1 or -1 sector,
        for i in range(DIM):
            if self.nd[i]:  # DVR dimension
                Rx = self.Xmat_1d(W, parity, i)
                if Rx is not None:  # If Rx is zero matrix it's not added
                    R.append(Rx)
        return R

    def Xmat_1d(self, W, parity: np.ndarray, i: int):
        Rx = np.zeros((self.lattice.N, self.lattice.N))
        # Permute the dimension to contract to the 1st
        idx = np.roll(np.arange(DIM, dtype=int), -i)
        if any(parity[:, i] == 0):
            # X = x_i delta_ij for non-symmetrized basis
            x = np.arange(-self.n[i], self.n[i] + 1) * self.dx[i]
            for j in range(self.lattice.N):
                Wj = np.transpose(W[j], idx).conj()
                Rx[j, j] = contract("ijk,i,ijk", Wj, x, Wj.conj())
                for k in range(j + 1, self.lattice.N):
                    Wk = np.transpose(W[k], idx)
                    Rx[j, k] = contract("ijk,i,ijk", Wj, x, Wk)
                    Rx[k, j] = Rx[j, k].conj()
        elif any(parity[:, i] == 1) and any(parity[:, i] == -1):
            # Get X^pp'_ij matrix rep for Delta^p_i, p the parity.
            # For direction a = x, matrix is nonzero only when p_a != p'_a:
            # X^pp'_ij = x_i delta_ij with p_x * p'_x = -1
            # For direction a != x, matrix is nonzero only when p_a = p'_a
            # As 1 and -1 sector have a dimension difference,
            # the matrix X is alsways an n-by-n+1
            # or n+1-by-n matrix with n diagonals
            # PS: Z is zero for singleband, as we only keep single p_z sector
            x = np.arange(1, self.n[i] + 1) * self.dx[i]
            lenx = len(x)
            for j in range(self.lattice.N):
                # If no absorber, W is real
                # This cconjugate does not change dtype of real W
                Wj = np.transpose(W[j], idx)[-lenx:, :, :].conj()
                for k in range(j + 1, self.lattice.N):
                    pjk = parity[j, idx] * parity[k, idx]
                    if pjk[0] == -1 and all(pjk[1:] == 1):
                        # Nonezero when only 1st dim parity differs for j and k states
                        Wk = np.transpose(W[k], idx)[-lenx:, :, :]
                        # Unitary transform X from DVR basis to single-body eigenbasis
                        Rx[j, k] = contract("ijk,i,ijk", Wj, x, Wk)
                        # Rx is also real if no absorber
                        # So Rx is real-symmetric or hermitian
                        Rx[k, j] = Rx[j, k].conj()
        else:
            Rx = None
        return Rx

    def singleband_WF(
        self, E, W, parity, x0=None, eig1d: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        # Singleband Wannier function optimization
        # x0 is the initial guess

        t0 = time()
        if self.lattice.N > 1:
            R = self.Xmat(W, parity)
            if len(R) == 1 and eig1d:
                # If only one R given, the problem is simply diagonalization
                # solution is eigenstates of operator X
                X, solution = la.eigh(R[0])
                # Auto sort eigenvectors by X eigenvalues
                order = np.argsort(X)
                WF = solution[:, order]
                wf_centers = np.array([X[order], np.zeros_like(X)]).T
            else:
                # In high dimension, X, Y, Z don't commute
                solution = riemann_minimize(R, x0, self.verbosity)
                WF = site_sort(self, solution, R)
                wf_centers = np.array(
                    [np.diag(WF.conj().T @ R[i] @ WF) for i in range(self.lattice.dim)]
                ).T
            cost = cost_func(WF, R).item()  # Convert to float
        else:
            WF = np.ones((1, 1))
            wf_centers = np.zeros((1, 2))
            cost = 0

        self.wf_centers = wf_centers
        self.wf_cost = cost
        A = WF.conj().T @ (E[:, None] * WF) * self.V0 / self.kHz_2p
        # TB parameter matrix, in unit of kHz
        t1 = time()
        if self.verbosity:
            print(f"Single band optimization time: {t1 - t0}s.")
        return A, WF

    def multiband_WF(self, E, W, parity, offset=True):
        # Multiband optimization
        A = []
        w = []
        wf_centers = []
        wf_costs = []
        for b in range(self.bands):
            t_ij, w_mu = self.singleband_WF(E[b], W[b], parity[b])
            if b == 0:
                # Shift onsite potential to zero average
                # Multi-band can only be shifted globally by 1st band
                if offset:
                    zero = np.mean(np.real(np.diag(t_ij)))
                else:
                    zero = 0
            A.append(t_ij - zero * np.eye(t_ij.shape[0]))
            w.append(w_mu)
            wf_centers.append(self.wf_centers)
            wf_costs.append(self.wf_cost)
        return A, w, wf_centers, wf_costs


# =============================================================================
# ========================== HELPERS ==========================


def site_sort(mlwf: MLWF, WF: np.ndarray, R: list[np.ndarray]) -> np.ndarray:
    # Order Wannier functions by lattice site label

    if mlwf.lattice.dim == 1:
        # Find WF center of mass
        x = np.diag(WF.conj().T @ R[0] @ WF)
        order = np.argsort(x)
    elif mlwf.lattice.dim > 1:
        # Find WF center of mass
        x = np.array(
            [np.diag(WF.conj().T @ R[i] @ WF) for i in range(mlwf.lattice.dim)]
        ).T
        order = nearest_match(mlwf.lattice.trap_centers, x)
    if mlwf.verbosity > 1:
        print("Trap site position of Wannier functions:", order)
        print("Order of Wannier functions is set to match traps.")
    return WF[:, order]


def interaction(mlwf: MLWF, WF: Iterable, W: Iterable, parity: Iterable, **kwargs):
    # Interaction between i band and j band
    onsite = kwargs.get("onsite", True)
    if onsite:
        Uint = np.zeros((mlwf.bands, mlwf.bands, mlwf.lattice.N))
        for i in range(mlwf.bands):
            for j in range(i, mlwf.bands):
                Uint[i, j, :] = singleband_interaction(
                    mlwf, WF[i], WF[j], W[i], W[j], parity[i], parity[j], **kwargs
                )
                if i != j:
                    Uint[j, i, :] = Uint[i, j, :]
    else:
        Uint = np.zeros(
            (
                mlwf.bands,
                mlwf.bands,
                mlwf.lattice.N,
                mlwf.lattice.N,
                mlwf.lattice.N,
                mlwf.lattice.N,
            )
        )
        for i in range(mlwf.bands):
            for j in range(i, mlwf.bands):
                Uint[i, j] = singleband_interaction(
                    mlwf, WF[i], WF[j], W[i], W[j], parity[i], parity[j], **kwargs
                )
                if i != j:
                    Uint[j, i] = Uint[i, j]
    return Uint


def singleband_interaction(
    mlwf: MLWF,
    WFi,
    WFj,
    Wi,
    Wj,
    pi: np.ndarray,
    pj: np.ndarray,
    method: str = "trapz",
    onsite=True,
) -> np.ndarray:
    # Interactions between single band i and j
    t0 = time()
    u = (
        4 * np.pi * mlwf.hb * mlwf.scatt_len / (mlwf.m * mlwf.kHz_2p * mlwf.w**DIM)
    )  # Unit to kHz
    x = []
    dx = []
    for i in range(DIM):
        if mlwf.nd[i]:  # Think of a way to make numerical integration converge
            x.append(
                np.linspace(-1.2 * mlwf.R0[i], 1.2 * mlwf.R0[i], mlwf.Nintgrl_grid)
            )
            dx.append(x[i][1] - x[i][0])
        else:
            x.append(np.array([0]))
            dx.append(0)
    Vi = wannier_func(x, WFi, mlwf, Wi, pi)
    Vj = Vi if WFi is WFj else wannier_func(x, WFj, mlwf, Wj, pj)
    if onsite:
        integrand = abs(Vi) ** 2 * abs(Vj) ** 2
        Uint = integrate(x, dx, integrand, method)
        if mlwf.model == "sho":
            print(
                f"Test with analytic calculation on {i + 1}-th site",
                np.real(Uint) * (np.sqrt(2 * np.pi)) ** mlwf.dim * np.prod(mlwf.hl),
            )
        t1 = time()
        if mlwf.verbosity:
            print(f"Single band interaction time: {t1 - t0}s.")
    else:
        # The matrix size is huge so do it sequentially
        mlwf.Nintgrl_grid = 129
        Uint = np.zeros([mlwf.lattice.N] * 4)
        for i in range(mlwf.lattice.N):
            for j in range(mlwf.lattice.N):
                for k in range(mlwf.lattice.N):
                    for l in range(mlwf.lattice.N):
                        integrand = (
                            Vi[:, :, :, i].conj()
                            * Vj[:, :, :, j].conj()
                            * Vj[:, :, :, k]
                            * Vi[:, :, :, l]
                        )
                        Uint[i, j, k, l] = integrate(x, dx, integrand, method)
        mlwf.Nintgrl_grid = 257  # Reset
    return u * Uint


def integrate(x, dx, integrand, method):
    if method == "romb":  # Not recommanded as is not converging well
        U = romb3d(integrand, dx)
    else:
        U = trapz3dnp(integrand, x)
    return U


def wannier_func(x: Iterable, WF, dvr: MLWF, W, p: np.ndarray) -> np.ndarray:
    x = [np.array([x[i]]) if isinstance(x[i], Number) else x[i] for i in range(DIM)]
    V = np.zeros((*(len(x[i]) for i in range(DIM)), p.shape[0]))
    for i in range(p.shape[0]):  # Loop over trap sites, p.shape[0] = Ntrap
        V[:, :, :, i] = psi(x, dvr.n, dvr.dx, W[i], p[i, :])[..., 0]
    return V @ WF


def symm_fold(reflection, info):
    # Extract information into symmetrized first sector
    return info[reflection[:, 0]]
