from numbers import Number
from typing import Iterable
import numpy as np
import sys
from mimetypes import init
from statistics import mode
from tkinter import N
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
import sparse
from opt_einsum import contract
from time import time

# from einops import rearrange, reduce, repeat

# Fundamental constants
a0 = 5.29177E-11  # Bohr radius, in unit of meter
# Eha = 6579.68392E12 * 2 * np.pi  # Hartree energy, in unit of Hz
amu = 1.66053907E-27  # atomic mass, in unit of kg
h = 6.62607015E-34  # Planck constant
# l = 780E-9 / a0  # 780nm, light wavelength

dim = 3  # space dimension


class harray(np.ndarray):

    @property
    def H(self):
        return self.conj().T


# TODO: Write code for elliptic tweezer, ie. wx!=wy.
#       Effecitvely this means I add a scaling factor to y direction.
# NOTE: 1. Harmonic length propto sqrt(w)
#       2. All length units are in wx, wy are rep by a factor wy/wx.
#          z direction zRx, zRy are also in unit of wx.


class DVR:

    def update_n(self, n: np.ndarray, R0: np.ndarray):
        # Change n with fixing R0
        self.n = n.copy()
        self.R0 = R0.copy()
        self.dx = np.zeros(n.shape)
        self.nd = n != 0
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]
        self.update_ab()

    def update_R0(self, R0: np.ndarray, dx: np.ndarray):
        self.R0 = R0.copy()
        self.dx = dx.copy()
        self.nd = R0 != 0
        self.n[self.nd == 0] = 0
        self.dx[self.nd == 0] = 0
        self.n[self.nd] = (self.R0[self.nd] / self.dx[self.nd]).astype(int)
        self.update_ab()

    def update_ab(self):
        print('DVR: dx={}w is set.'.format(self.dx[self.nd]))
        print('DVR: n={} is set.'.format(self.n[self.nd]))
        print('DVR: R0={}w is set.'.format(self.R0[self.nd]))
        if self.absorber:
            print('DVR: Absorber width LI={:g}w'.format(self.LI))
            # if __debug__:
            #     print(self.R0[0])
            #     print(self.dx[0])
            #     a = self.LI / self.dx[self.nd]
            #     print(a[0])
            #     print(np.rint(self.LI / self.dx[self.nd]).astype(int))
            self.n[self.nd] += np.rint(self.LI / self.dx[self.nd]).astype(int)
            print('DVR: n is set to {} by adding absorber.'.format(
                self.n[self.nd]))
            self.R[self.nd] = self.n[self.nd] * self.dx[self.nd]
            print('DVR: R={}w is set.'.format(self.R[self.nd]))

    def update_p(self, p):
        self.p = p
        self.init = get_init(self.n, p)

    def __init__(
            self,
            n: np.ndarray,
            R0: np.ndarray,
            avg=1,
            model='Gaussian',
            trap=(104.52,
                  1000),  # 2nd entry in array is (wx, wy), in number is (w, w)
            atom=6.015122,  # Atom mass, in amu. Default Lithium-6
            laser=780,  # 780nm, laser wavelength
            zR=None,  # Rayleigh range input by hand
            symmetry: bool = False,
            absorber: bool = False,
            ab_param=(57.04, 1),
            sparse: bool = False) -> None:
        self.n = n.copy()
        self.R0 = R0.copy()  # Physical region size, In unit of waist
        self.R = R0.copy()  # Total region size, R = R0 + LI
        self.avg = avg
        self.model = model
        self.absorber = absorber
        self.symmetry = symmetry
        self.nd = n != 0  # Nonzero dimensions
        self.sparse = sparse

        self.dx = np.zeros(n.shape)
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]  # In unit of waist
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
        if symmetry:
            self.p[self.nd] = 1
            axis = np.array(['x', 'y', 'z'])
            print('{}-reflection symmetry is used.'.format(axis[self.nd]))
        self.init = get_init(self.n, self.p)

        if model == 'Gaussian':
            # Experiment parameters in atomic units
            self.hb = h / (2 * np.pi)  # Reduced Planck constant
            self.m = atom * amu  # Atom mass, in unit of electron mass
            self.l = laser * 1E-9  # Laser wavelength, in unit of Bohr radius
            self.kHz = 1E3  # Make in the frequency unit of kHz
            self.kHz_2p = 2 * np.pi * 1E3  # Make in the agnular kHz frequency
            self.V0 = trap[
                0] * self.kHz_2p  # Input V0 is frequency in unit of kHz, convert to angular frequency 2 * pi * kHz

            # Input in unit of nm, converted to m
            if isinstance(trap[1], Iterable):  # Convert to np.array
                self.w = np.array(trap[1][0]) * 1E-9
                self.wy = np.array(trap[1]) / trap[1][0]  # wi/wx
            elif isinstance(trap[1], Number):  # Number convert to np.array
                self.w = trap[1] * 1E-9
                self.wy = np.ones(2)

            # TO GET A REASONABLE ENERGY SCALE, WE SET V0=1 AS THE ENERGY UNIT HEREAFTER
            self.mtV0 = self.m * self.V0
            # Rayleigh range, a vector of (zRx, zRy), in unit of wx
            self.zR = np.pi * self.w * self.wy**2 / self.l
            # Rayleigh range input by hand, in unit of wx
            if isinstance(zR, Number):
                self.zR = zR * np.ones(2) / self.w
            elif isinstance(zR, Iterable):
                self.zR = zR / self.w
            # "Effective" Rayleigh range
            self.zR0 = np.prod(self.zR) / la.norm(self.zR)

            # Trap frequencies
            self.omega = np.array([*(2 / self.wy), 1 / self.zR0])
            self.omega *= np.sqrt(avg * self.hb * self.V0 / self.m) / self.w
            # Trap harmonic lengths
            self.hl = np.sqrt(self.hb / (self.m * self.omega))

            print("param_set: trap parameter V0={}kHz w={}nm".format(
                trap[0], trap[1]))
        elif model == 'sho':
            # Harmonic parameters
            self.hb = 1  # Reduced Planck constant
            self.omega = np.ones(dim)  # Harmonic frequencies
            self.m = 1.0
            self.w = 1.0
            self.mtV0 = self.m
            self.V0 = 1.0
            self.kHz = 1.0
            self.kHz_2p = 1.0
            self.hl = np.sqrt(self.hb /
                              (self.m * self.omega))  # Harmonic lengths

            print("param_set: trap parameter V0={} w={}".format(
                self.V0, self.w))

        self.R0 *= self.nd
        self.R *= self.nd
        self.dx *= self.nd
        # To cancel effect of any h.l. multiplication
        self.hl[np.logical_not(self.nd)] = 1

        ## Abosorbers
        if absorber:
            self.VI *= self.kHz_2p  # Absorption potential strength in unit of angular kHz frequency
            self.VIdV0 = self.VI / self.V0  # Energy in unit of V0

    def Vfun(self, x, y, z):
        # Potential function
        if self.model == 'Gaussian':
            # Tweezer potential funciton, Eq. 2 in PRA
            d0 = 1 + (z / self.zR0)**2 / 2
            dxy = (x / self.wy[0])**2 / (1 + (z / self.zR[0])**2)
            dxy += (y / self.wy[1])**2 / (1 + (z / self.zR[1])**2)
            V = -1 / d0 * np.exp(-2 * dxy)
        elif self.model == 'sho':
            # Harmonic potential function
            V = self.m / 2 * self.omega**2 * (x**2 + y**2 + z**2)
        return V

    def Vabs(self, x, y, z):
        r = np.array([x, y, z]).transpose(1, 2, 3, 0)
        np.set_printoptions(threshold=sys.maxsize)
        d = abs(r) - self.R0
        d = (d > 0) * d
        L = self.R - self.R0
        L[L == 0] = np.inf
        # if __debug__:
        #     print(self.R)
        #     print(self.R0)
        #     print(L)
        Vi = np.sum(d / L, axis=3)
        if Vi.any() != 0.0:
            V = -1j * self.VIdV0 * Vi  # Energy in unit of V0
        return V


def get_init(n, p):
    init = np.zeros(n.shape, dtype=int)
    init[p == 0] = -n[p == 0]
    init[p == 1] = 0
    init[p == -1] = 1
    return init


def Vmat(dvr: DVR):
    # Potential energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       potential(x, y, z) is a function handle to be processed as potential function for solving
    x = []
    for i in range(dim):
        x.append(np.arange(dvr.init[i], dvr.n[i] + 1) *
                 dvr.dx[i])  # In unit of w
    X = np.meshgrid(*x, indexing='ij')
    # 3 index tensor V(x, y, z)
    V = dvr.avg * dvr.Vfun(*X)
    if dvr.absorber:  # add absorber
        V = V.astype(complex) + dvr.Vabs(*X)
    no = dvr.n + 1 - dvr.init
    # V * identity rank-6 tensor
    if not dvr.sparse:
        V = np.diag(V.reshape(-1))
        V = V.reshape(*no, *no)
    return V, no


def psi(n, dx, W: np.ndarray, x, y, z, p=np.zeros(dim,
                                                  dtype=int)) -> np.ndarray:
    init = get_init(n, p)
    # V = np.sum(
    #     W.reshape(*(np.append(n + 1 - init, -1))), axis=1
    # )  # Sum over y, z index to get y=z=0 cross section of the wavefunction
    deltax = dx.copy()
    nd = deltax == 0
    deltax[nd] = 1
    xn = [np.arange(init[i], n[i] + 1) for i in range(dim)]
    V = delta(p, [x / deltax[0], y / deltax[1], z / deltax[2]], xn)
    # if W.ndim < 4:
    #     W = W[..., *[None for i in range(4-W.ndim)]]
    if W.ndim == 3:
        W = W[..., None]
    psi = 1 / np.sqrt(np.prod(deltax)) * contract('il,jm,kn,lmnp', *V, W)
    return psi


def delta(p, x, xn):
    # Symmetrized sinc DVR basis funciton, x, y, z are in unit of dx
    W = []
    for i in range(dim):
        Wx = np.sinc(x[i][:, None] - xn[i][None])
        if p[i] != 0:
            Wx += p[i] * np.sinc(x[i][:, None] + xn[i][None])
            Wx /= np.sqrt(2)
            if p[i] == 1:
                Wx[:, 0] /= np.sqrt(2)
        W.append(Wx)
    return W


def kinetic_offdiag(T: np.ndarray):
    # Kinetic energy matrix off-diagonal elements
    non0 = T != 0  # To avoid warning on 0-divide-0
    T[non0] = 2 * np.power(-1., T[non0]) / T[non0]**2
    return T


def Tmat_1d(dvr: DVR, i: int):
    # Kinetic energy matrix for 1-dim
    n = dvr.n[i]
    # dx is dimensionless by dividing w,
    # to restore unit we need to multiply w
    dx = dvr.dx[i] * dvr.w
    p = dvr.p[i]

    init = get_init(np.array([n]), p)[0]

    # Off-diagonal part
    T0 = np.arange(init, n + 1, dtype=float)[None]
    T = kinetic_offdiag(T0 - T0.T)

    # Diagonal part
    T[np.diag_indices(n + 1 - init)] = np.pi**2 / 3
    if p != 0:
        T += p * kinetic_offdiag(T0 + T0.T)
        if p == 1:
            T[:, 0] /= np.sqrt(2)
            T[0, :] /= np.sqrt(2)
            T[0, 0] = np.pi**2 / 3
    # get kinetic energy in unit of V0, hb is cancelled as V0 is in unit of angular freq, mtV0 = m * V0
    T *= dvr.hb / (2 * dx**2 * dvr.mtV0)
    return T


def Tmat(dvr: DVR):
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: 1. here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       2. p=0, d=-1 means no symmetry applied
    delta = []
    T0 = []
    for i in range(dim):
        delta.append(np.eye(dvr.n[i] + 1 - dvr.init[i]))  # eg. delta_xx'
        if dvr.n[i]:
            T0.append(Tmat_1d(dvr, i))  # append p-sector
        # If the systems is set to have only 1 grid point (N = 0) in this direction, ie. no such dimension
        else:
            T0.append(None)

    if dvr.sparse:
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


def H_op(dvr: DVR, T: list, V, no, psi0: np.ndarray):
    # Define Hamiltonian operator for sparse solver

    dvr.p *= dvr.n != 0
    dvr.dx *= dvr.n != 0

    psi0 = psi0.reshape(*no)
    psi = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
    psi += np.einsum('ij,jkl->ikl', T[0], psi0)  # T_xx' delta_yy' delta_zz'
    psi += np.einsum('jl,ilk->ijk', T[1], psi0)  # delta_xx' T_yy' delta_zz'
    psi += np.einsum('ij,klj->kli', T[2], psi0)  # delta_xx' delta_yy' T_zz'
    return psi.reshape(-1)


def H_mat(dvr: DVR):
    # Construct Hamiltonian matrix

    dvr.p *= dvr.n != 0
    dvr.dx *= dvr.n != 0

    # np.set_printoptions(precision=2, suppress=True)
    print("H_mat: n={} dx={}w p={} {} starts.".format(dvr.n[dvr.nd],
                                                      dvr.dx[dvr.nd],
                                                      dvr.p[dvr.nd],
                                                      dvr.model))
    T = Tmat(dvr)
    V, no = Vmat(dvr)
    H = T + V
    del T, V
    N = np.prod(no)
    H = H.reshape((N, N))
    if not dvr.absorber:
        H = (H + H.T.conj()) / 2
    print("H_mat: H matrix memory usage: {:.2f} MiB.".format(H.nbytes / 2**20))
    return H


def H_solver(dvr: DVR, k: int = -1) -> tuple[np.ndarray, np.ndarray]:
    # Solve Hamiltonian matrix

    if dvr.sparse:
        print(
            "H_op: n={} dx={}w p={} {} sparse diagonalization is enabled. Lowest {} states are to be calculated."
            .format(dvr.n[dvr.nd], dvr.dx[dvr.nd], dvr.p[dvr.nd], dvr.model,
                    k))

        T = Tmat(dvr)
        V, no = Vmat(dvr)
        print("H_op: n={} dx={}w p={} {} operator constructed.".format(
            dvr.n[dvr.nd], dvr.dx[dvr.nd], dvr.p[dvr.nd], dvr.model))

        def applyH(psi):
            return H_op(dvr, T, V, no, psi)

        t0 = time()
        N = np.product(no)
        H = LinearOperator((N, N), matvec=applyH)

        if k <= 0:
            k = 10
        if dvr.absorber:
            print('H_solver: diagonalize sparse non-hermitian matrix.')
            E, W = sla.eigs(H, k, which='SA')
        else:
            print('H_solver: diagonalize sparse hermitian matrix.')
            E, W = sla.eigsh(H, k, which='SA')
    else:
        # avg factor is used to control the time average potential strength
        H = H_mat(dvr)
        t0 = time()
        if dvr.absorber:
            E, W = la.eig(H)
        else:
            E, W = la.eigh(H)
        if k > 0:
            E = E[:k]
            W = W[:, :k]

    t1 = time()

    if dvr.avg > 0:
        print('H_solver: {} Hamiltonian solved. Time spent: {:.2f}s.'.format(
            dvr.model, t1 - t0))
    elif dvr.avg == 0:
        print(
            'H_solver: free particle Hamiltonian solved. Time spent: {:.2f}s.'.
            format(t1 - t0))
    print("H_solver: eigenstates memory usage: {:.2f} MiB.".format(W.nbytes /
                                                                   2**20))
    return E, W
