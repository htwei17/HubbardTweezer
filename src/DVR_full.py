import numpy as np
import sys
from mimetypes import init
from statistics import mode
from tkinter import N
import numpy as np
import scipy.linalg as la
# import scipy.sparse.linalg as sla
# import scipy.sparse as sp
# from scipy.sparse.linalg import LinearOperator
# import sparse
from opt_einsum import contract
from time import time
# from einops import rearrange, reduce, repeat

k = 10  # Number of energy levels to track

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
    def update_n(self, n: np.ndarray):
        self.n = n
        self.dx = np.zeros(n.shape)
        if self.absorber:
            self.dx[n != 0] = (self.R[n != 0] - self.LI) / n[n != 0]
            self.n[n != 0] = int(self.R[n != 0] / self.dx[n != 0])
        else:
            self.dx[n != 0] = self.R[n != 0] / n[n != 0]

    def update_R(self, R):
        nd = self.n != 0
        self.R = R
        if self.absorber:
            self.R += self.LI
        self.n[nd] = int(self.R[nd] / self.dx[nd])

    def __init__(self,
                 n,
                 R,
                 avg=1,
                 model='Gaussian',
                 trap=(104.52, 1E-6),
                 symmetry=False,
                 absorber=False,
                 ab_param=(57.04, 1)) -> None:
        self.n = n
        self.R = R  # In unit of waist
        self.avg = avg
        self.model = model
        self.absorber = absorber

        self.dx = np.zeros(n.shape)
        self.dx[n != 0] = self.R[n != 0] / n[n != 0]  # In unit of waist
        if self.absorber:
            self.VI, self.LI = ab_param
            self.R += self.LI
            self.n[n != 0] = int(self.R[n != 0] / self.dx[n != 0])
        else:
            self.VI = 0
            self.LI = 0

        self.p = np.zeros(dim, dtype=int)
        if symmetry:
            self.p[n == 0] = 0
            axis = np.array(['x', 'y', 'z'])
            print('{}-reflection symmetry is used.'.format(axis[n != 0]))
        self.init = get_init(self.n, self.p)

        if model == 'Gaussian':
            # Experiment parameters in atomic units
            self.m = 6.015122 * amu  # Lithium-6 atom mass, in unit of electron mass

            self.kHz_2p = 2 * np.pi * 1E3  # Make in the frequency unit of 2 * pi * kHz
            self.V0_SI = trap[0] * self.kHz_2p * hb  # Input in unit of kHz
            self.w = trap[1] / a0  # Input in unit of meter
            self.R *= self.w
            self.dx *= self.w

            # NORMAL WAIST
            # V0_SI = 1.0452E5 * 2 * np.pi  # 104.52kHz * h, potential depth, in SI unit, since hbar is set to 1 this should be multiplied by 2pi
            # w = 1E-6 / Par.a0  # ~1000nm, waist length, in unit of Bohr radius
            # FATTEST WAIST
            # V0_SI = 1.56E5 * 2 * np.pi  # trap depth for fattest waist
            # w = 1.18E-6 / a0  # fattest waist length
            # TIGHTEST WAIST
            # V0_SI = 7.6E4 * 2 * np.pi  # trap depth for tightest waist
            # w = 8.61E-7 / a0  # tightest waist length

            self.V0 = self.V0_SI / Eha  # potential depth in unit of Hartree energy
            # TO GET A REASONABLE ENERGY SCALE, WE SET v=1 AS THE ENERGY UNIT HEREAFTER

            self.mtV0 = self.m * self.V0
            self.zR = np.pi * self.w**2 / l  # ~4000nm, Rayleigh range

            self.Nmax = np.array([20, 20, 20])  # Max number of grid points
            self.dx0 = self.w * np.array([.15, 1.5, 0.36
                                          ])  # Fixed delta x value
            # dx0 = np.array([.1 * w, .1 * w, 0.25 * w])  # Fixed delta x value
            # Fixed R value, should be larger than Rayleigh length scale
            self.R0 = self.w * np.array([1, 1, 2.4])

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

        ## Abosorbers
        if absorber:
            self.VI *= self.kHz_2p / self.V0_SI  # Absorption potential strength in unit of V0
            self.LI *= self.w  # Absorption region, in unit of w

        self.dx *= n != 0

    def Vfun(self, x, y, z):
        # Potential function
        if self.model == 'Gaussian':
            # Tweezer potential funciton, Eq. 2 in PRA
            den = 1 + (z / self.zR)**2
            V = -1 / den * np.exp(-2 * (x**2 + y**2) / (self.w**2 * den))
        elif self.model == 'sho':
            # Harmonic potential function
            V = self.m / 2 * self.omega**2 * (x**2 + y**2 + z**2)
        return V

    def Vabs(self, x, y, z, L=np.inf * np.ones(dim), R=np.zeros(dim)):
        r = np.array([x, y, z]).transpose(1, 2, 3, 0)
        np.set_printoptions(threshold=sys.maxsize)
        d = abs(r) - R
        d = (d > 0) * d
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


def Vmat(DVR):
    # Potential energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       potential(x, y, z) is a function handle to be processed as potential function for solving
    x = []
    for i in range(dim):
        x.append(np.arange(DVR.init[i], DVR.n[i] + 1) * DVR.dx[i])
    X = np.meshgrid(x[0], x[1], x[2], indexing='ij')
    # 3 index tensor V(x, y, z)
    V = DVR.avg * DVR.Vfun(X[0], X[1], X[2])
    if DVR.absorber:  # add absorption layer
        L = DVR.LI * (DVR.n > 0)
        Ri = DVR.n * DVR.dx - DVR.LI
        L[DVR.n == 0] = np.inf
        V = V.astype(complex) + DVR.Vabs(X[0], X[1], X[2], L, Ri)
    # V * identity rank-6 tensor, sparse
    no = DVR.n + 1 - DVR.init
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


def Tmat(DVR) -> np.ndarray:
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: 1. here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       2. p=0, d=-1 means no symmetry applied
    delta = []
    T0 = []
    for i in range(dim):
        delta.append(np.eye(DVR.n[i] + 1 - DVR.init[i]))  # eg. delta_xx'
        if DVR.n[i] > 0:
            T0.append(Tmat_1d(DVR.n[i], DVR.dx[i], DVR.mtV0,
                              DVR.p[i]))  # append p-sector
        # If the systems is set to have only 1 grid point (N = 0) in this direction, ie. no such dimension
        else:
            T0.append(None)

    # delta_xx' delta_yy' T_zz'
    # delta_xx' T_yy' delta_zz'
    # T_xx' delta_yy' delta_zz'
    T = 0
    for i in range(dim):
        if isinstance(T0[i], np.ndarray):
            T += contract('ij,kl,mn->ikmjln', *delta[:i], T0[i],
                          *delta[i + 1:])
    return T


def H_mat(DVR):
    # Construct Hamiltonian matrix

    DVR.p *= DVR.n != 0
    DVR.dx *= DVR.n != 0

    np.set_printoptions(precision=2, suppress=True)
    print("H_mat: n={} dx={}w p={} {} starts.".format(DVR.n, DVR.dx / DVR.w,
                                                      DVR.p, DVR.model))
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


def H_solver(DVR) -> tuple[np.ndarray, np.ndarray]:
    # Solve Hamiltonian matrix
    # avg factor is used to control the time average potential strength
    H = H_mat(DVR)
    # [E, W] = sla.eigsh(H, which='SA')
    t0 = time()
    if DVR.absorber:
        E, W = la.eig(H)
    else:
        E, W = la.eigh(H)
    t1 = time()
    if DVR.avg > 0:
        print('H_solver: {} Hamiltonian solved. Time spent: {:.2f}s.'.format(
            DVR.model, t1 - t0))
    elif DVR.avg == 0:
        print(
            'H_solver: free particle Hamiltonian solved. Time spent: {:.2f}s.'.
            format(t1 - t0))
    print("H_solver: eigenstates memory usage: {:.2f} MiB.".format(W.nbytes /
                                                                   2**20))
    return E, W
