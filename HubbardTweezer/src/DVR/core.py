from numbers import Number
from typing import Iterable, Literal, Union
import numpy as np
import sys
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from opt_einsum import contract
from time import time


# Fundamental constants
a0 = 5.29177E-11  # Bohr radius, in unit of meter
# micron = 1000  # Length scale micrn, in unit of nm
# Eha = 6579.68392E12 * 2 * np.pi  # Hartree energy, in unit of Hz
amu = 1.66053907E-27  # atomic mass, in unit of kg
h = 6.62607015E-34  # Planck constant
# l = 780E-9 / a0  # 780nm, light wavelength

dim = 3  # space dimension

# NOTE: 1. Harmonic length propto sqrt(w)
#       2. All length units are in wx, wy are rep by a factor wy/wx.
#          z direction zRx, zRy are also in unit of wx.


def get_init(n: np.ndarray, p: np.ndarray) -> np.ndarray:
    init = -n
    init[p == 1] = 0
    init[p == -1] = 1
    return init


def _kinetic_offdiag(T: np.ndarray) -> np.ndarray:
    # Kinetic energy matrix off-diagonal elements
    non0 = T != 0  # To avoid warning on 0-divide-0
    T[non0] = 2 * np.power(-1., T[non0]) / T[non0]**2
    return T


class DVR:
    """DVR base class

    Args:
    ----------
        n (`np.ndarray[int, int, int]`): DVR grid half-size in each direction.
            The actual grid size is 2n+1.
            If i-th dimension is not calculated, n[i]=0
        R0 (`np.ndarray[float, float]`): Grid halfwidth in each direction
        model (`str`): Trap potential.
            'Gaussian' means tweezer potential;
            'sho' means harmonic potential
        avg (`float`): Factor a in Humiltonian H = T + aV
        trap (`tuple[float, float | tuple[float, float]]`): Trap potential parameters.
            trap[0] is the trap potential strength in unit of kHz
            trap[1] is the waist in unit of nm;
            if trap[1] given as a tuple, then it is (wx, wy)
        atom (`float`): Atom mass in unit of amu
        laser ('float'): Laser wavelength in unit of nm
        zR ('float'): Rayleigh length in unit of nm;
            if not given, zR is calculated by \pi * w^2 / laser
        sparse (`bool`): Whether to use sparse matrix
        symmetry (`bool`): Whether to use symmetry in DVR
        absorber (`bool`): Whether to use absorber
        ab_param (`tuple[float, float]`): Absorber parameters.
            ab_param[0] is the strength of linear imaginary potential in unit of kHz
            ab_param[1] is the width of absorber in unit of wx
    """

    def update_n(self, n: np.ndarray, R0: np.ndarray):
        # Change n by fixed R0
        self.n = n.copy()
        self.init[self.nd] = get_init(self.n[self.nd], self.p[self.nd])
        self.R0 = R0.copy()
        self.dx = np.zeros(n.shape)
        self.nd = n != 0
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]
        self.update_ab()

    def update_R0(self, R0: np.ndarray, dx: np.ndarray):
        # Update R0 by fixed dx
        self.R0 = R0.copy()
        self.dx = dx.copy()
        self.nd = R0 != 0
        self.n[self.nd == 0] = 0
        self.dx[self.nd == 0] = 0
        self.n[self.nd] = (self.R0[self.nd] / self.dx[self.nd]).astype(int)
        self.init[self.nd] = get_init(self.n[self.nd], self.p[self.nd])
        self.update_ab()

    def update_ab(self):
        # Update absorber
        if self.verbosity:
            print('DVR: dx={}w is set.'.format(self.dx[self.nd]))
            print('DVR: n={} is set.'.format(self.n[self.nd]))
            print('DVR: R0={}w is set.'.format(self.R0[self.nd]))
        if self.absorber:
            if self.verbosity:
                print('DVR: Absorber width LI={:g}w'.format(self.LI))
            # if __debug__:
            #     print(self.R0[0])
            #     print(self.dx[0])
            #     a = self.LI / self.dx[self.nd]
            #     print(a[0])
            #     print(np.rint(self.LI / self.dx[self.nd]).astype(int))
            self.n[self.nd] += np.rint(self.LI / self.dx[self.nd]).astype(int)
            self.R[self.nd] = self.n[self.nd] * self.dx[self.nd]
            if self.verbosity > 1:
                print('DVR: n is set to {} by adding absorber.'.format(
                    self.n[self.nd]))
                print('DVR: R={}w is set.'.format(self.R[self.nd]))

    def update_p(self, p):
        # Update parity
        self.p = p
        self.init = get_init(self.n, p)

    def __init__(
            self,
            n: np.ndarray,
            R0: np.ndarray,
            avg: float = 1,
            model: str = 'Gaussian',
            # 2nd entry in array is (wx, wy) in unit of nm
            # if given in single number w it is (w, w)
            trap: tuple[float, Union[float, tuple[float, float]]] = (
                104.52, 1000),
            atom: float = 6.015122,  # Atom mass, in amu. Default Lithium-6
            laser: float = 780,  # 780nm, laser wavelength in unit of nm
            # Rayleigh range input by hand, in unit of nm
            zR: Union[None, float] = None,
            symmetry: bool = False,
            # Parity of each dimension, used when symmetry is True
            parity: Union[None, np.ndarray] = None,
            absorber: bool = False,
            ab_param: tuple[float, float] = (57.04, 1),
            sparse: bool = False,
            verbosity: int = 2  # How much information to print
    ) -> None:
        self.n = n.copy()
        self.R0 = R0.copy()  # Physical region size, In unit of wx
        self.R = R0.copy()  # Total region size, R = R0 + LI

        self.avg = avg
        self.model = model
        if self.avg == 0:
            self.model = "free"

        self.absorber = absorber
        self.dvr_symm = symmetry
        self.nd: np.ndarray = n != 0  # Nonzero dimensions
        self.sparse = sparse
        self.verbosity = verbosity if verbosity >= 0 else 0

        self.dx = np.zeros(n.shape)
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]  # In unit of wx
        if self.absorber:
            self.VI, self.LI = ab_param
        else:
            self.VI = 0
            self.LI = 0
        self.update_ab()
        # if __debug__:
        #     print(self.R)
        #     print(self.R0)

        self.p = np.zeros(dim, dtype=int)
        if self.dvr_symm:
            if parity is None:
                self.p[self.nd] = 1
            else:
                self.p[self.nd] = parity[self.nd].astype(int)
            if self.verbosity:
                axis = np.array(['x', 'y', 'z'])
                print(f'{axis[self.nd]}-reflection symmetry is used.')
        self.init = get_init(self.n, self.p)

        if model == 'Gaussian':
            # Experiment parameters in atomic units
            self.hb = h / (2 * np.pi)  # Reduced Planck constant
            self.m: Literal = atom * amu  # Atom mass, in unit of electron mass
            self.l: Literal = laser * 1E-9  # Laser wavelength, in unit of Bohr radius
            self.kHz: Literal = 1E3  # Make in the frequency unit of kHz
            self.kHz_2p: Literal = 2 * np.pi * 1E3  # Make in the agnular kHz frequency
            self.V0: float = trap[
                0] * self.kHz_2p  # Input V0 is frequency in unit of kHz, convert to angular frequency 2 * pi * kHz

            # Input in unit of nm, converted to m
            if isinstance(trap[1], Iterable) and len(trap[1]) == 1:
                wx: Number = trap[1][0]
                self.wxy: np.ndarray = np.ones(2)
            elif isinstance(trap[1], Iterable):  # Convert to np.array
                wx: Number = trap[1][0]  # In unit of nm
                self.wxy: np.ndarray = np.array(
                    trap[1]) / wx  # wi in unit of wx
            elif isinstance(trap[1], Number):  # Number convert to np.array
                wx: Number = trap[1]  # In unit of nm
                self.wxy: np.ndarray = np.ones(2)
            else:
                wx: Literal = 1000
                self.wxy: np.ndarray = np.ones(2)
            self.w: Literal = wx * 1E-9  # Convert micron to m

            # TO GET A REASONABLE ENERGY SCALE, WE SET V0=1 AS THE ENERGY UNIT HEREAFTER
            self.mtV0 = self.m * self.V0
            # Rayleigh range, a vector of (zRx, zRy), in unit of wx
            self.zR = np.pi * self.w * self.wxy**2 / self.l
            # Rayleigh range input by hand, in unit of wx
            if isinstance(zR, Number):
                self.zR: np.ndarray = zR * np.ones(2) / wx
            elif isinstance(zR, Iterable):
                self.zR: np.ndarray = np.array(zR) / wx
            # "Effective" Rayleigh range
            self.zR0: float = np.prod(self.zR) / la.norm(self.zR)

            # Trap frequencies
            self.omega = np.array([*(np.sqrt(2) / self.wxy), 1 / self.zR0])
            self.omega *= np.sqrt(2 * self.avg * self.hb *
                                  self.V0 / self.m) / self.w
            # Trap harmonic lengths
            self.hl: np.ndarray = np.sqrt(self.hb / (self.m * self.omega))

            if self.verbosity:
                print(
                    f"param_set: trap parameter V0={avg * trap[0]}kHz w={trap[1]}nm")
        elif model == 'sho':
            # Harmonic parameters
            self.hb: Literal = 1.0  # Reduced Planck constant
            self.omega = np.ones(dim)  # Harmonic frequencies
            self.m: Literal = 1.0
            self.w: Literal = 1.0
            self.mtV0 = self.m
            self.V0 = 1.0
            self.kHz: Literal = 1.0
            self.kHz_2p: Literal = 1.0
            self.hl: np.ndarray = np.sqrt(self.hb /
                                          (self.m * self.omega))  # Harmonic lengths

            print(
                f"param_set: trap parameter V0={avg * self.V0} w0={self.w}")

        self.R0 *= self.nd
        self.R *= self.nd
        self.dx *= self.nd
        # To cancel effect of any h.l. multiplication
        self.hl[np.logical_not(self.nd)] = 1

        # Abosorbers
        if absorber:
            self.VI *= self.kHz_2p  # Absorption potential strength in unit of angular kHz frequency
            self.VIdV0 = self.VI / self.V0  # Energy in unit of V0

    def Vfun(self, x, y, z):
        # Potential function
        if self.model == 'Gaussian':
            # Tweezer potential funciton, Eq. 2 in PRA
            d0 = 1 + (z / self.zR0)**2 / 2
            dxy = (x / self.wxy[0])**2 / (1 + (z / self.zR[0])**2)
            dxy += (y / self.wxy[1])**2 / (1 + (z / self.zR[1])**2)
            V = -1 / d0 * np.exp(-2 * dxy)
        elif self.model == 'sho':
            # Harmonic potential function
            V = self.m / 2 * self.omega**2 * (x**2 + y**2 + z**2)
        return V

    def Vabs(self, x, y, z):
        r = np.array([x, y, z]).transpose(1, 2, 3, 0)
        np.set_printoptions(threshold=sys.maxsize)
        d: np.ndarray = abs(r) - self.R0
        d = (d > 0) * d
        L: np.ndarray = self.R - self.R0
        L[L == 0] = np.inf
        # if __debug__:
        #     print(self.R)
        #     print(self.R0)
        #     print(L)
        Vi: np.ndarray = np.sum(d / L, axis=3)
        if Vi.any() != 0.0:
            V = -1j * self.VIdV0 * Vi  # Energy in unit of V0
        return V

    def Vmat(self) -> tuple[np.ndarray, np.ndarray]:
        # Potential energy tensor, index order [x y z x' y' z']
        # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
        #       potential(x, y, z) is a function handle to be processed as potential function for solving
        x = []
        for i in range(dim):
            x.append(np.arange(self.init[i], self.n[i] + 1) *
                     self.dx[i])  # In unit of micron
        X = np.meshgrid(*x, indexing='ij')
        # 3 index tensor V(x, y, z)
        V: np.ndarray = self.avg * self.Vfun(*X)
        if self.absorber:  # add absorber
            V = V.astype(complex) + self.Vabs(*X)
        no = self.n + 1 - self.init
        # V * identity rank-6 tensor
        if not self.sparse:
            V = np.diag(V.reshape(-1))
            V = V.reshape(*no, *no)
        return V, no

    def _Tmat_1d(self, i: int):
        # Kinetic energy matrix for 1-dim
        n = self.n[i]
        # dx is dimensionless by dividing w,
        # to restore unit we need to multiply w
        dx = self.dx[i] * self.w
        p = self.p[i]

        init = get_init(np.array([n]), np.array([p]))[0]

        # Off-diagonal part
        T0 = np.arange(init, n + 1, dtype=float)[None]
        T = _kinetic_offdiag(T0 - T0.T)

        # Diagonal part
        T[np.diag_indices(n + 1 - init)] = np.pi**2 / 3
        if p != 0:
            T += p * _kinetic_offdiag(T0 + T0.T)
            if p == 1:
                T[:, 0] /= np.sqrt(2)
                T[0, :] /= np.sqrt(2)
                T[0, 0] = np.pi**2 / 3
        # get kinetic energy in unit of V0, hb is cancelled as V0 is in unit of angular freq, mtV0 = m * V0
        T *= self.hb / (2 * dx**2 * self.mtV0)
        return T

    def Tmat(self):
        # Kinetic energy tensor, index order [x y z x' y' z']
        # NOTE: 1. here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
        #       2. p=0, d=-1 means no symmetry applied
        delta = []
        T0 = []
        for i in range(dim):
            delta.append(np.eye(self.n[i] + 1 - self.init[i]))  # eg. delta_xx'
            if self.n[i]:
                T0.append(self._Tmat_1d(i))  # append p-sector
            # If the systems is set to have only 1 grid point (N = 0) in this direction, ie. no such dimension
            else:
                T0.append(None)

        if self.sparse:
            for i in range(dim):
                if not isinstance(T0[i], np.ndarray):
                    T0[i] = np.zeros((1, 1))
            return T0
        else:
            # delta_xx' delta_yy' T_zz'
            # delta_xx' T_yy' delta_zz'
            # T_xx' delta_yy' delta_zz'
            T = 0
            for i in range(dim):
                if isinstance(T0[i], np.ndarray):
                    T += contract('ij,kl,mn->ikmjln', *delta[:i], T0[i],
                                  *delta[i + 1:])
            return T

    def H_op(self, T: list, V, no, psi0: np.ndarray):
        # Define Hamiltonian operator for sparse solver

        psi0 = psi0.reshape(*no)
        psi: np.ndarray = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
        # T_xx' delta_yy' delta_zz'
        psi += np.einsum('ij,jkl->ikl', T[0], psi0)
        # delta_xx' T_yy' delta_zz'
        psi += np.einsum('jl,ilk->ijk', T[1], psi0)
        # delta_xx' delta_yy' T_zz'
        psi += np.einsum('ij,klj->kli', T[2], psi0)
        return psi.reshape(-1)

    def H_mat(self):
        # Construct Hamiltonian matrix
        self.p *= self.n != 0
        self.dx *= self.n != 0

        # np.set_printoptions(precision=2, suppress=True)
        if self.verbosity:
            print(
                f"H_mat: n={self.n[self.nd]} dx={self.dx[self.nd]}w p={self.p[self.nd]} {self.model} diagonalization starts.")
        T = self.Tmat()
        V, no = self.Vmat()
        H = T + V
        del T, V
        N = np.prod(no)
        H = H.reshape((N, N))
        if not self.absorber:
            H = (H + H.T.conj()) / 2
        if self.verbosity:
            print(f"H_mat: H matrix memory usage: {H.nbytes / 2**20:.2f} MiB.")
        return H

    def H_solver(self, k: int = -1) -> tuple[np.ndarray, np.ndarray]:
        # Solve Hamiltonian matrix

        if self.sparse:
            if self.verbosity:
                print(
                    "H_op: n={} dx={}w p={} {} sparse diagonalization starts. Lowest {} states are to be calculated."
                    .format(self.n[self.nd], self.dx[self.nd], self.p[self.nd], self.model,
                            k))

            self.p *= self.n != 0
            self.dx *= self.n != 0

            T = self.Tmat()
            V, no = self.Vmat()
            # for i in range(3):
            #     print(T[i])
            # print(V)
            if self.verbosity > 2:
                print("H_op: n={} dx={}w p={} {} operator constructed.".format(
                    self.n[self.nd], self.dx[self.nd], self.p[self.nd], self.model))

            t0 = time()
            def applyH(psi) -> np.ndarray: return self.H_op(T, V, no, psi)
            N = np.product(no)
            H = LinearOperator((N, N), matvec=applyH)

            if k <= 0:
                k = 10
            if self.absorber:
                if self.verbosity > 2:
                    print('H_solver: diagonalize sparse non-hermitian matrix.')

                E, W = ssla.eigs(H, k, which='SA')
            else:
                if self.verbosity > 2:
                    print('H_solver: diagonalize sparse hermitian matrix.')
                E, W = ssla.eigsh(H, k, which='SA')
        else:
            # avg factor is used to control the time average potential strength
            H = self.H_mat()
            t0 = time()
            if self.absorber:
                if self.verbosity > 2:
                    print('H_solver: diagonalize non-hermitian matrix.')
                E, W = la.eig(H)
            else:
                if self.verbosity > 2:
                    print('H_solver: diagonalize hermitian matrix.')
                E, W = la.eigh(H)
            if k > 0:
                E = E[:k]
                W = W[:, :k]

        t1 = time()

        if self.verbosity:
            print(
                f'H_solver: {self.model} Hamiltonian solved. Time spent: {t1 - t0:.2f}s.')
            print(
                f"H_solver: eigenstates memory usage: {W.nbytes/2**20: .2f} MiB.")
        # No absorber, all eigenstates are real
        return E, W
