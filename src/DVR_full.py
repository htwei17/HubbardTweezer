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
a0 = 5.29E-11  # Bohr radius, in unit of meter
Eha = 6579.68392E12 * 2 * np.pi  # Hartree energy, in unit of Hz
amu = 1822.89  # atomic mass, in unit of electron mass
hb = 1  # Reduced Planck const
l = 780E-9 / a0  # 780nm, light wavelength

dim = 3  # space dimension


class harray(np.ndarray):

    @property
    def H(self):
        return self.conj().T


class DVR:

    def update_n(self, n: np.ndarray, R0: np.ndarray):
        # Change n with fixing R0
        self.n = n.copy()
        self.R0 = R0.copy()
        self.dx = np.zeros(n.shape)
        self.nd = n != 0
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]
        self.update_ab()

    # def update_R0(self, R0: np.ndarray, dx: np.ndarray):
    #     self.R0 = R0.copy()
    #     self.dx = dx.copy()
    #     self.nd = R0 != 0
    #     self.dx[np.invert(self.nd)] = 0
    #     self.n[self.nd] = (self.R0[self.nd] / self.dx[self.nd]).astype(int)
    #     self.update_ab()

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
            print('DVR: n is set to {} by adding absorber.'.format(self.n))
            self.R[self.nd] = self.n[self.nd] * self.dx[self.nd]
            print('DVR: R={}w is set.'.format(self.R[self.nd]))

    def __init__(self,
                 n: np.ndarray,
                 R0: np.ndarray,
                 avg=1,
                 model='Gaussian',
                 trap=(104.52, 1E-6),
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
            self.m = 6.015122 * amu  # Lithium-6 atom mass, in unit of electron mass

            self.kHz_2p = 2 * np.pi * 1E3  # Make in the frequency unit of 2 * pi * kHz
            self.V0_SI = trap[0] * self.kHz_2p * hb  # Input in unit of kHz
            self.w = trap[
                1] / a0  # Input in unit of meter, converted to Bohr radius

            self.V0 = self.V0_SI / Eha  # potential depth in unit of Hartree energy
            # TO GET A REASONABLE ENERGY SCALE, WE SET v=1 AS THE ENERGY UNIT HEREAFTER

            self.mtV0 = self.m * self.V0
            self.zR = np.pi * self.w / l  # ~4000nm, Rayleigh range, in unit of waist

            self.Nmax = np.array([20, 20, 20])  # Max number of grid points

            print("param_set: trap parameter V0={}kHz w={}m".format(
                trap[0], trap[1]))
        elif model == 'sho':
            # Harmonic parameters
            self.dx0 = 1 / 3 * np.ones(3, dtype=float)
            self.Nmax = 30
            self.R0 = self.Nmax * self.dx0
            self.omega = 1.0
            self.m = 1.0
            self.w = 1.0
            self.mtV0 = self.m
            self.V0_SI = 1.0
            self.kHz = 1.0

            print("param_set: trap parameter V0={} w={}".format(
                self.V0_SI, self.w))

        self.R0 *= self.nd
        self.R *= self.nd
        self.dx *= self.nd
        ## Abosorbers
        if absorber:
            self.VI *= self.kHz_2p / self.V0_SI  # Absorption potential strength in unit of V0

    def Vfun(self, x, y, z):
        # Potential function
        if self.model == 'Gaussian':
            # Tweezer potential funciton, Eq. 2 in PRA
            den = 1 + (z / self.zR)**2
            V = -1 / den * np.exp(-2 * (x**2 + y**2) / den)
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
            V = -1j * self.VI * Vi
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
        x.append(np.arange(dvr.init[i], dvr.n[i] + 1) * dvr.dx[i])
    X = np.meshgrid(*x, indexing='ij')
    # 3 index tensor V(x, y, z)
    V = dvr.avg * dvr.Vfun(*X)
    if dvr.absorber:  # add absorption layer
        V = V.astype(complex) + dvr.Vabs(X[0], X[1], X[2])
    no = dvr.n + 1 - dvr.init
    # V * identity rank-6 tensor
    if not dvr.sparse:
        V = np.diag(V.reshape(-1))
        V = V.reshape(np.concatenate((no, no)))
    return V, no


def psi(n, dx, W, x, p=0) -> np.ndarray:
    init = get_init(n, p)
    V = np.sum(W.reshape(*(np.append(n + 1 - init, -1))), axis=(
        1, 2
    ))  # Sum over y, z index to get y=z=0 cross section of the wavefunction
    xn = np.arange(init[0], n[0] + 1)[None] * dx
    W = np.sinc((x - xn) / dx)
    if p != 0:
        W += p * np.sinc((x + xn) / dx)
    W /= np.sqrt(2)
    if p == 1:
        W[:, 0] /= np.sqrt(2)
    psi = 1 / np.sqrt(dx) * W @ V
    return psi


def kinetic_offdiag(T):
    # Kinetic energy matrix off-diagonal elements
    non0 = T != 0  # To avoid warning on 0-divide-0
    T[non0] = 2 * np.power(-1., T[non0]) / T[non0]**2
    return T


def Tmat_1d(n, dx, mtV0, p=0):
    # Kinetic energy matrix for 1-dim

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
    # get kinetic energy in unit of V0, mtV0 = m * V0
    T *= hb**2 / (2 * dx**2 * mtV0)
    return T


def Tmat(dvr: DVR) -> np.ndarray:
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: 1. here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       2. p=0, d=-1 means no symmetry applied
    delta = []
    T0 = []
    for i in range(dim):
        delta.append(np.eye(dvr.n[i] + 1 - dvr.init[i]))  # eg. delta_xx'
        if dvr.n[i] > 0:
            T0.append(Tmat_1d(dvr.n[i], dvr.dx[i] * dvr.w, dvr.mtV0,
                              dvr.p[i]))  # append p-sector
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


def H_op(DVR: DVR, T, V, no, psi0):
    # Define Hamiltonian operator for sparse solver

    DVR.p *= DVR.n != 0
    DVR.dx *= DVR.n != 0

    psi0 = psi0.reshape(*no)
    psi = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
    psi += np.einsum('ij,jkl->ikl', T[0], psi0)  # T_xx' delta_yy' delta_zz'
    psi += np.einsum('jl,ilk->ijk', T[1], psi0)  # delta_xx' T_yy' delta_zz'
    psi += np.einsum('ij,klj->kli', T[2], psi0)  # delta_xx' delta_yy' T_zz'
    return psi.reshape(-1)


def H_mat(DVR: DVR):
    # Construct Hamiltonian matrix

    DVR.p *= DVR.n != 0
    DVR.dx *= DVR.n != 0

    # np.set_printoptions(precision=2, suppress=True)
    print("H_mat: n={} dx={}w p={} {} starts.".format(DVR.n[DVR.nd],
                                                      DVR.dx[DVR.nd],
                                                      DVR.p[DVR.nd],
                                                      DVR.model))
    T = Tmat(DVR)
    V, no = Vmat(DVR)
    H = T + V
    del T, V
    N = np.prod(no)
    H = H.reshape((N, N))
    if not DVR.absorber:
        H = (H + H.T.conj()) / 2
    print("H_mat: H matrix memory usage: {:.2f} MiB.".format(H.nbytes / 2**20))
    return H


def H_solver(dvr: DVR, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
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
        if dvr.absorber:
            E, W = sla.eigs(H, k, which='SA')
        else:
            E, W = sla.eigsh(H, k, which='SA')
    else:
        # avg factor is used to control the time average potential strength
        H = H_mat(dvr)
        # [E, W] = sla.eigsh(H, which='SA')
        t0 = time()
        if dvr.absorber:
            E, W = la.eig(H)
        else:
            E, W = la.eigh(H)

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
