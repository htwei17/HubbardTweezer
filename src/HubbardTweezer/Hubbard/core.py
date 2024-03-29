import numpy as np
from numpy.linalg import LinAlgError
from typing import Iterable
from numbers import Number
from opt_einsum import contract
from time import time
from itertools import product
import numpy.linalg as la

from ..DVR import DVR
from ..DVR.const import *
from ..DVR.wavefunc import psi
from ..tools.integrate import romb3d, trapz3dnp
from ..tools.point_match import nearest_match
from .riemann import riemann_minimize
from .lattice import Lattice
from .ghost import GhostTrap

tri_lattice_list = ["triangular", "honeycomvb", "defecthoneycomb", "kagome", "zigzag"]


class MLWF(DVR):
    """Construct maximally localized Wannier functions (MLWF) for a given lattice
    and calculate Hubbard parameters.

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

    # Number of grid points for numerical integration in each dimension
    Nintgrl_grid: int = 257
    ghost: GhostTrap  # GhostTrap property object
    lattice: Lattice  # Lattice property object

    def create_lattice(
        self,
        shape: str = "square",
        lattice: np.ndarray = np.array([2], dtype=int),
        lc: tuple[float, float] = (1520, 1690),
        nodes: np.ndarray = None,
        links: np.ndarray = None,
    ):
        # Create lattice and set DVR grid and box
        # graph : each line represents coordinate (x, y) of one lattice site

        self.lattice = Lattice(lattice, shape, self.ls, nodes, links)

        self.set_lc(lc, shape)  # Convert lc to (lc, lc) and in unit of wx

        # Assume WF are localized at trap centers, location in unit of wx
        self.tc0 = self.lattice.nodes * self.lc
        self.trap_centers = self.tc0.copy()
        self.wf_centers = self.tc0.copy()

        dx = self.dx.copy()
        lattice_range = np.max(abs(self.tc0), axis=0)
        lattice_range = np.resize(np.pad(lattice_range, (0, 2), constant_values=0), dim)
        if self.verbosity:
            print(f"Lattice: lattice shape is {shape}")
            print(f"Lattice: Full lattice sizes: {lattice}")
            if self.verbosity > 1:
                print(f"Lattice: lattice constants: {self.lc}w")
                print(f"Lattice: dx fixed to: {dx[self.nd]}w")
        # Let there be R0's wide outside the edge trap center
        R0 = lattice_range + self.R00
        R0 *= self.nd
        self.update_R0(R0, dx)

    def set_lc(self, lc, shape):
        if isinstance(lc, Iterable) and len(lc) == 1:
            # Convert (lc, ) back to lc
            lc: Number = lc[0]
        if isinstance(lc, Number):
            # Convert lc to (lcx, lcy)
            # Lattice is isotropic if only one lc is given
            self.isotropic = True
            if shape in tri_lattice_list:
                # For equilateral triangle,
                # y lattice constant is sqrt(3)/2 * x lattice constant
                lc: tuple = (lc, np.sqrt(3) / 2 * lc)
            else:
                # For squre and others,
                # convert lc to (lc, lc)
                lc: tuple = (lc, lc)
        # Check if the lattice is isotropic
        if shape not in tri_lattice_list and lc[0] == lc[1]:
            self.isotropic = True
        print(f"Lattice: lattice shape is {shape}; lattice constants set to: {lc}")

        # Set lattice constants in unit of wx
        if self.model in ["Gaussian", "lattice"]:
            self.lc = np.array(lc) * 1e-9 / self.w
        elif self.model == "sho":
            self.lc = np.array(lc)

    def update_lattice(self, tc: np.ndarray):
        # Update DVR grids when trap centers are shifted

        self.trap_centers = tc.copy()
        dx = self.dx.copy()
        lattice = np.resize(np.pad(self.lattice.size, (0, 2), constant_values=1), dim)
        lc = np.resize(self.lc, dim)
        if self.verbosity:
            print(f"Lattice: Full lattice sizes updated to: {lattice[self.nd]}")
            if self.verbosity > 1:
                # Let there be R0's wide outside the edge trap center
                print(f"Lattice: lattice constants updated to: {lc}w")
                print(f"Lattice: dx fixed to: {dx[self.nd]}w")
        R0 = (lattice - 1) * lc / 2 + self.R00
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
        shape="square",  # Shape of the lattice
        lattice_symmetry: bool = True,  # Whether the lattice has reflection symmetry
        # Lattice dimensions & lattice constant, in unit of nm
        lattice_params: tuple[np.ndarray, tuple] = (
            np.array([2], dtype=int),
            (1520, 1690),
        ),
        # Custom lattice site positions & lattice links
        custom_lattice: tuple[np.ndarray] = (None, None),
        isotropic: bool = False,  # Check if the lattice is isotropic
        ascatt=1770,  # Scattering length, in unit of Bohr radius, default 1770
        band=1,  # Number of bands
        balance_V0: bool = False,  # Equalize trap depths V0 for all traps first, useful for two-band calculation
        dim: int = 3,
        *args,
        **kwargs,
    ) -> None:
        self.N = N
        self.scatt_len = ascatt * a0
        self.dim = dim
        self.bands = band
        self.ls = lattice_symmetry
        self.isotropic = isotropic
        if shape == "zigzag":
            self.ls = False
        n = np.zeros(3, dtype=int)
        n[:dim] = N

        # make sure absortber is not used
        absorber = kwargs.get("absorber", False)
        if absorber:
            raise TypeError(
                "Absorber is not supported for Wannier Function construction!"
            )

        # Numerical integration grid point number
        self.Nintgrl_grid = kwargs.get("Nintgrl_grid", 257)
        print(f"Wannier: Number of integration grid set to {self.Nintgrl_grid}.")

        super().__init__(n, *args, **kwargs)
        # Backup buffer zone size
        # buffer zone = from edge trap center to DVR box edge
        self.R00 = self.R0.copy()
        lsize, lc = lattice_params  # Lattice size & lattice constant
        nodes, links = custom_lattice  # Custom lattice site positions & lattice links
        self.create_lattice(shape, lsize, lc, nodes, links)
        self.Voff = np.ones(self.lattice.N)  # Set default trap offset
        # Set waist adjustment factor
        self.wxy0 = self.wxy.copy()
        self.waists = np.ones((self.lattice.N, 2))

        # Balance trap depth to make sure traps aren't too uneven
        # NOTE: this makes U more uneven.
        if balance_V0:
            self.balance_trap_depths()

        # Set to cancel onsite potential offset, quantities are of no use
        # They will be overwritten in HubbardEqualizer
        self.ghost = GhostTrap(self.lattice, shape)

    def Vfun(self, x, y, z):
        # Get V(x, y, z) for the entire lattice
        V = 0

        if self.model == "sho" and self.lattice.N == 2:
            # Two-site SHO case
            V += super().Vfun(abs(x) - self.lc[0] / 2, y, z)
        elif self.model == "lattice":
            # Rectangular optical lattice potential in 2D
            V = (
                np.cos(2 * np.pi * x / self.lc[0])
                + np.cos(2 * np.pi * y / self.lc[1])
                - 2
            ) / 2
        else:
            # Gaussian trap potential of tweezer array
            # NOTE: DO NOT SET coord DIRECTLY!
            # THIS WILL DIRECTLY MODIFY self.graph!
            for i in range(self.lattice.N):
                shift = self.trap_centers[i]
                self.update_waist(self.waists[i])
                V += self.Voff[i] * super().Vfun(x - shift[0], y - shift[1], z)
        return V

    def singleband_Hubbard(
        self, u=False, x0=None, W0=None, offset=True, band=1, eig_sol=None
    ):
        # Calculate single band tij matrix and U matrix
        band_bak = self.bands
        if band == 1: # If only 1st band is needed, set bands to 1
            self.bands = 1
        if eig_sol != None:  # Unpack pre-calculated eigen solution
            E, W, p = eig_sol
        else:  # Calculate eigen solution
            E, W, p = self.eigen_basis(W0=W0)
        # Select band
        E = E[band - 1]  # Eigen energy
        W = W[band - 1]  # Eigen vector
        p = p[band - 1]  # Sector index
        self.A, V = self.singleband_WF(E, W, p, x0)  # Single band WF & tij matrix
        if offset is True:
            # Shift onsite physical potential to zero average
            self.zero = np.mean(self.ghost.mask_quantity(np.real(np.diag(self.A))))
        elif isinstance(offset, Number):
            self.zero = offset
        else:
            self.zero = 0
        self.A -= self.zero * np.eye(self.A.shape[0])

        if u and self.verbosity:
            print("Calculate U.")
        self.U = singleband_interaction(self, V, V, W, W, p, p) if u else None
        self.bands = band_bak
        return self.A, self.U, V

    def trap_mat(self):
        # Total depth at each trap center
        tc = np.zeros((self.lattice.N, dim))
        vij = np.ones((self.lattice.N, self.lattice.N))
        for i in range(self.lattice.N):
            tc[i, :] = np.append(self.trap_centers[i], 0)
            for j in range(i):
                vij[i, j] = -DVR.Vfun(self, *(tc[i] - tc[j]))
                vij[j, i] = vij[i, j]  # Potential is symmetric in distance
        return vij

    def balance_trap_depths(self):
        # Balance trap depths at each trap center to be equal
        vij = self.trap_mat()
        # Set trap depth target to be the deepest one
        Vtarget = np.max(vij @ np.ones(self.lattice.N))
        try:
            # Balance trap depth by adjusting trap offset
            # to compensate for trap unevenness
            self.Voff = la.solve(vij, Vtarget * np.ones(self.lattice.N)) ** 2
            if self.verbosity:
                print(f"Equalize: trap depths equalzlied to {self.Voff}.")
        except:
            raise LinAlgError("Homogenize: failed to solve for Voff.")

    def nn_tunneling(self, A: np.ndarray, links=None):
        # From tij matrix, pick up nearest neighbor tunnelings
        # by the links information
        if links is None:
            # Default links
            links = self.lattice.links
        if self.lattice.N == 1:
            nnt = np.zeros(1)
        elif self.lattice.dim == 1:
            nnt = np.diag(A, k=1)
        else:
            nnt = A[links[:, 0], links[:, 1]]
        return nnt

    def symm_unfold(self, target: Iterable, info, graph=False):
        # Unfold information to all symmetric sites
        # No need to output as target is Iterable
        if self.ls:
            parity = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
            for row in range(self.lattice.reflect.shape[0]):
                if graph:  # Symmetrize graph node coordinates
                    # NOTE: repeated nodes will be removed
                    info[row][self.lattice.inv_coords[row]] = 0
                    target[self.lattice.reflect[row, :]] = parity * info[row][None]
                else:  # Symmetrize trap depth
                    target[self.lattice.reflect[row, :]] = info[row]
        else:
            target[:] = info

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
        p = [1, -1] if self.ls else [0]  # ls: lattice symmetry
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
        self, W0: list = None, standard: str = "energy"
    ) -> tuple[list, list, list]:
        # Find eigenbasis of symmetry block diagonalized Hamiltonian
        k = self.lattice.N * self.bands
        if self.dvr_symm:
            p_list = self.build_sectors()
            E_sb = np.array([])
            W_sb = []
            p_sb = np.array([], dtype=int).reshape(0, dim)
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

            if standard == "energy":
                # Sort everything by energy, only keetp lowest k states
                idx = np.argsort(E_sb)[: k + 1]
                E_sb = E_sb[idx]
                W_sb = [W_sb[i] for i in idx[:k]]
                p_sb = p_sb[idx, :]
            elif standard == "symmetry":
                # Don't select state right now,
                # keep all k+1 states in each sector
                idx = np.argsort(E_sb)
                E_sb = E_sb[idx]
                W_sb = [W_sb[i] for i in idx]
                p_sb = p_sb[idx, :]
        else:
            p_sb = np.zeros((k, dim))
            E_sb, W_sb = self.H_solver(k + 1)
            W_sb = [W_sb[:, i].reshape(2 * self.n + 1) for i in range(k)]

        if self.verbosity > 2:
            print(f"Energies: {E_sb}")
            if self.ls:
                print(f"parities: {[p_sb]}")
        # elif self.verbosity > 1 and E_sb[k - 1] - E_sb[0] > E_sb[k] - E_sb[k - 1]:
        #     print("Wannier warning: band gap is smaller than band width.")

        if standard == "symmetry" and self.bands == 1:
            standard = "energy"

        if standard == "energy":
            E_sb = E_sb[:k]
            p_sb = p_sb[:k]
            E = [
                E_sb[b * self.lattice.N : (b + 1) * self.lattice.N]
                for b in range(self.bands)
            ]
            W = [
                W_sb[b * self.lattice.N : (b + 1) * self.lattice.N]
                for b in range(self.bands)
            ]
            parity = [
                p_sb[b * self.lattice.N : (b + 1) * self.lattice.N, :]
                for b in range(self.bands)
            ]
        elif standard == "symmetry" and self.bands == 2:
            # Hard coded pz-even and pz-odd bands
            E_even = np.array([])
            E_odd = np.array([])
            W_even = []
            W_odd = []
            parity_even = np.array([], dtype=int).reshape(0, dim)
            parity_odd = np.array([], dtype=int).reshape(0, dim)
            count_even = 0
            count_odd = 0
            for pidx in range(len(p_sb)):
                p = p_sb[pidx]
                if p[2] == 1 and count_even < self.lattice.N:
                    # Even parity
                    E_even = np.append(E_even, E_sb[pidx])
                    W_even.append(W_sb[pidx])
                    parity_even = np.append(parity_even, p[None], axis=0)
                    count_even += 1
                elif p[2] == -1 and count_odd < self.lattice.N:
                    # Odd parity
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
        for i in range(dim):
            if self.nd[i]:
                Rx = self.Xmat_1d(W, parity, i)
                if Rx is not None:
                    R.append(Rx)
        return R

    def Xmat_1d(self, W, parity: np.ndarray, i: int):
        # Calculate X_ij = <i|x|j> for single-body eigenbasis |i>
        Rx = np.zeros((self.lattice.N, self.lattice.N))
        # Permute the dimension to contract to the 1st
        idx = np.roll(np.arange(dim, dtype=int), -i)
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

    # ========================== MLWF OPTIMIZATION ==========================

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
                U = solution[:, order]
                wf_centers = np.array([X[order], np.zeros_like(X)]).T
            else:
                # In high dimension, X, Y, Z don't commute
                solution = riemann_minimize(R, x0, self.verbosity)
                U = site_sort(self, solution, R)
                wf_centers = np.array(
                    [np.diag(U.conj().T @ R[i] @ U) for i in range(self.lattice.dim)]
                ).T
        else:
            U = np.ones((1, 1))
            wf_centers = np.zeros((1, 2))

        self.wf_centers = wf_centers
        A = U.conj().T @ (E[:, None] * U) * self.V0 / self.kHz_2p
        # TB parameter matrix, in unit of kHz
        t1 = time()
        if self.verbosity:
            print(f"Single band optimization time: {t1 - t0}s.")
        return A, U

    def multiband_WF(self, E, W, parity, offset=True):
        # Multiband optimization
        A = []
        w = []
        wf_centers = []
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
        return A, w, wf_centers


# =============================================================================


def site_sort(dvr: MLWF, U: np.ndarray, R: list[np.ndarray]) -> np.ndarray:
    # Sort Wannier functions by lattice site label

    if dvr.lattice.dim == 1:
        # Find WF center of mass
        x = np.diag(U.conj().T @ R[0] @ U)
        order = np.argsort(x)
    elif dvr.lattice.dim > 1:
        # Find WF center of mass
        x = np.array([np.diag(U.conj().T @ R[i] @ U) for i in range(dvr.lattice.dim)]).T
        order = nearest_match(dvr.trap_centers, x)
    if dvr.verbosity > 1:
        print("Trap site position of Wannier functions:", order)
        print("Order of Wannier functions is set to match traps.")
    return U[:, order]


def interaction(dvr: MLWF, U: Iterable, W: Iterable, parity: Iterable, **kwargs):
    # Interaction between i band and j band
    onsite = kwargs.get("onsite", True)
    if onsite:
        Uint = np.zeros((dvr.bands, dvr.bands, dvr.lattice.N))
        for i in range(dvr.bands):
            for j in range(i, dvr.bands):
                Uint[i, j, :] = singleband_interaction(
                    dvr, U[i], U[j], W[i], W[j], parity[i], parity[j], **kwargs
                )
                if i != j:
                    Uint[j, i, :] = Uint[i, j, :]
    else:
        Uint = np.zeros(
            (
                dvr.bands,
                dvr.bands,
                dvr.lattice.N,
                dvr.lattice.N,
                dvr.lattice.N,
                dvr.lattice.N,
            )
        )
        for i in range(dvr.bands):
            for j in range(i, dvr.bands):
                Uint[i, j] = singleband_interaction(
                    dvr, U[i], U[j], W[i], W[j], parity[i], parity[j], **kwargs
                )
                if i != j:
                    Uint[j, i] = Uint[i, j]
    return Uint


def singleband_interaction(
    dvr: MLWF,
    Ui,
    Uj,
    Wi,
    Wj,
    pi: np.ndarray,
    pj: np.ndarray,
    method: str = "trapz",
    onsite=True,
) -> np.ndarray:
    # Density-density interactions between single band i and j
    t0 = time()
    u = (
        4 * np.pi * dvr.hb * dvr.scatt_len / (dvr.m * dvr.kHz_2p * dvr.w**dim)
    )  # Unit to kHz
    x = []
    dx = []
    for i in range(dim):
        # Construct integration spatial grid
        if dvr.nd[i]:  # Think of a way to make numerical integration converge
            x.append(np.linspace(-1.2 * dvr.R0[i], 1.2 * dvr.R0[i], dvr.Nintgrl_grid))
            dx.append(x[i][1] - x[i][0])
        else:
            x.append(np.array([0]))
            dx.append(0)
    Vi = wannier_func(x, Ui, dvr, Wi, pi)
    Vj = Vi if Ui is Uj else wannier_func(x, Uj, dvr, Wj, pj)
    if onsite:
        integrand = abs(Vi) ** 2 * abs(Vj) ** 2
        Uint = integrate(x, dx, integrand, method)
        # if dvr.model == "sho":
        #     print(
        #         f"Test with analytic calculation on {i + 1}-th site",
        #         np.real(Uint) * (np.sqrt(2 * np.pi)) ** dvr.dim * np.prod(dvr.hl),
        #     )
        t1 = time()
        if dvr.verbosity:
            print(f"Single band interaction time: {t1 - t0}s.")
    else:
        # The matrix size is huge so do it sequentially
        Nintgl_grid = dvr.Nintgrl_grid
        dvr.Nintgrl_grid = 129
        Uint = np.zeros([dvr.lattice.N] * 4)
        for i in range(dvr.lattice.N):
            for j in range(dvr.lattice.N):
                for k in range(dvr.lattice.N):
                    for l in range(dvr.lattice.N):
                        integrand = (
                            Vi[:, :, :, i].conj()
                            * Vj[:, :, :, j].conj()
                            * Vj[:, :, :, k]
                            * Vi[:, :, :, l]
                        )
                        Uint[i, j, k, l] = integrate(x, dx, integrand, method)
        dvr.Nintgrl_grid = Nintgl_grid  # Reset
    return u * Uint


def integrate(x, dx, integrand, method):
    if method == "romb":  # Not recommanded as is not converging well
        U = romb3d(integrand, dx)
    else:
        U = trapz3dnp(integrand, x)
    return U


def wannier_func(x: Iterable, U, dvr: MLWF, W, p: np.ndarray) -> np.ndarray:
    x = [np.array([x[i]]) if isinstance(x[i], Number) else x[i] for i in range(dim)]
    V = np.zeros((*(len(x[i]) for i in range(dim)), p.shape[0]))
    for i in range(p.shape[0]):  # Loop over trap sites, p.shape[0] = Ntrap
        V[:, :, :, i] = psi(x, dvr.n, dvr.dx, W[i], p[i, :])[..., 0]
    return V @ U


def symm_fold(reflection, info):
    # Extract information into symmetrized first sector
    return info[reflection[:, 0]]
