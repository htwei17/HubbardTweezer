import numpy as np
import scipy.sparse as sp
import numpy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import LinearOperator
from opt_einsum import contract
from time import time
import sys

# Fundamental constants
a0 = 5.29E-11  # Bohr radius, in unit of meter
Eha = 6579.68392E12 * 2 * np.pi  # Hartree energy, in unit of Hz
amu = 1822.89  # atomic mass, in unit of electron mass
hb = 1  # Reduced Planck const
m = 6.015122 * amu  # Lithium-6 atom mass, in unit of electron mass
dim = 3  # space dimension

# Experiment parameters in atomic units
V0_SI = 1.0452E5 * 2 * np.pi  # 104.52kHz * h, potential depth, in SI unit, since hbar is set to 1 this should be multiplied by 2pi
V0 = V0_SI / Eha  # potential depth in unit of Hartree energy
# TO GET A REASONABLE ENERGY SCALE, WE SET v=1 AS THE ENERGY UNIT HEREAFTER
w = 1E-6 / a0  # ~1000nm, waist length, in unit of Bohr radius
l = 780E-9 / a0  # 780nm, light wavelength
zR = np.pi * w**2 / l  # ~4000nm, Rayleigh range

Nmax = np.array([7, 7, 15])  # Max number of grid points
dx0 = np.array([1 / 8 * w, 1 / 8 * w, 0.4738 * w])  # Fixed delta x value
# Fixed R value, should be larger than Rayleigh length scale
R0 = np.array([w, w, 2 * zR])
k = 10  # Number of energy levels to track
# TODO:
#       3. Make use of system symmetries, reflection, cylindrical, etc.

## Abosorbers
VI0 = 4E5 / V0_SI  # Absorption potential strength in unit of V0
LI = .3 * w  # Absorption region, in unit of dx

# Harmonic parameters
dx0_sho = 1 / 3 * np.ones(3, dtype=float)
Nmax_sho = 30
R0_sho = Nmax_sho * dx0_sho
omega = 1
m_sho = 1


def Vfun(x, y, z):
    # Potential function, Eq.2 in PRA
    den = 1 + (z / zR)**2
    V = -1 / den * np.exp(-2 * (x**2 + y**2) / (w**2 * den))
    return V


def Vabs(x, y, z, VI=VI0, L=np.inf * np.ones(dim), R=np.zeros(dim)):
    r = np.array([x, y, z]).transpose(1, 2, 3, 0)
    np.set_printoptions(threshold=sys.maxsize)
    d = abs(r) - R
    d = (d > 0) * d
    Vi = np.sum(d / L, axis=3)
    if Vi.any() != 0.0:
        V = -1j * VI * Vi
    return V


def Vsho(x, y, z):
    # Harmonic potential function
    V = m_sho / 2 * omega**2 * (x**2 + y**2 + z**2)
    return V


def Vmat(n, dx, potential, avg=1, absorber=False, ab_param=(LI, VI0)):
    # Potential energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       potential(x, y, z) is a function handle to be processed as potential function for solving
    x = []
    for i in range(dim):
        x.append(np.arange(-n[i], n[i] + 1) * dx[i])
    X = np.meshgrid(x[0], x[1], x[2], indexing='ij')
    # 3 index tensor V(x, y, z)
    V = avg * potential(X[0], X[1], X[2])
    if absorber:  # add absorption layer
        L = ab_param[0] * (n > 0)
        Ri = n * dx - ab_param[0]
        L[np.nonzero(n == 0)] = np.inf
        V = V.astype(complex) + Vabs(
            X[0], X[1], X[2], VI=ab_param[1], L=L, R=Ri)
    return V


def kinetic_offdiag(n):
    # Kinetic energy matrix off-diagonal elements
    T = 2 * np.power(-1., n) / n**2
    return T


def Tmat_1d(n, dx, mtV0):
    # Kinetic energy matrix for 1-dim
    # Off-diagonal part
    T = np.arange(-n, n + 1).reshape(1, -1)
    T = T - T.transpose()
    # Force diagonals to be 1, to avoid warning on 0-dived-0, but this value assignment can be ignored as the diagonal entries are to be replaced in the next step
    T[np.diag_indices(2 * n + 1)] = 1
    T = kinetic_offdiag(T)
    # Diagonal part
    T[np.diag_indices(2 * n + 1)] = np.pi**2 / 3
    # get kinetic energy in unit of V0, mtV0 = m * V0
    T *= hb**2 / (2 * dx**2 * mtV0)
    return T


def Tmat(n, dx, mtV0):
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    T = []
    for i in range(dim):
        if n[i] != 0:
            T.append(Tmat_1d(n[i], dx[i], mtV0))  # eg. T_xx'
        else:  # If the systems is set to have only 1 grid point in this direction, ie. no such dimension
            T.append(np.zeros((1, 1)))
    return T


def mem_est(n):
    n = np.array(n)
    Ndim = np.product(2 * n + 1)
    mem = 2**(2 * np.log2(Ndim) + 3 - 30)
    print("mem_set: Matrix size= {}".format(Ndim))
    print(
        "mem_set: Estimated full matrix memory usage: {:.2f} GiB".format(mem))


def H_op(T, V, n, psi0):
    N = 2 * n + 1
    psi0 = psi0.reshape(*N)
    psi = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
    psi += np.einsum('ij,jkl->ikl', T[0], psi0)  # T_xx' delta_yy' delta_zz'
    psi += np.einsum('jl,ilk->ijk', T[1], psi0)  # delta_xx' T_yy' delta_zz'
    psi += np.einsum('ij,klj->kli', T[2], psi0)  # delta_xx' delta_yy' T_zz'
    return psi.reshape(-1)


def H_mat(n,
          dx,
          avg=1,
          potential=Vfun,
          model='Gaussian',
          absorber=False,
          ab_param=(LI, VI0)):
    # Construct Hamiltonian matrix

    if model == 'Gaussian':
        mtV0 = m * V0
    elif model == 'sho':
        mtV0 = m_sho
    else:
        exit(2)

    np.set_printoptions(precision=2, suppress=True)
    print("H_mat: n={} dx={}w {} starts.".format(n, dx / w, model))
    T = Tmat(n, dx, mtV0)
    V = Vmat(n, dx, potential, float(avg), absorber, ab_param)
    return T, V


def H_solver(n,
             dx,
             k=k,
             avg=1,
             potential=Vfun,
             model='Gaussian',
             absorber=False,
             ab_param=(LI, VI0)):
    # Solve Hamiltonian matrix
    # avg factor is used to control the time average potential strength

    T, V = H_mat(n, dx, avg, potential, model, absorber, ab_param)
    mem_est(n)

    def applyH(psi):
        return H_op(T, V, n, psi)

    t0 = time()
    N = np.product(2 * n + 1)
    H = LinearOperator((N, N), matvec=applyH)
    if absorber:
        E, W = sla.eigs(H, k, which='SA')
    else:
        E, W = sla.eigsh(H, k, which='SA')
    t1 = time()
    if avg > 0:
        print('H_solver: {} Hamiltonian solved. Time spent: {:.2f}s.'.format(
            model, t1 - t0))
    elif avg == 0:
        print(
            'H_solver: free particle Hamiltonian solved. Time spent: {:.2f}s.'.
            format(t1 - t0))
    return E, W


def N_convergence(N, R, k=k, avg=1, dim=3, wavefunc=False, level=1):
    # Convergence of energy vs N, to reproduce Fig. 5a in PRA paper
    E = np.array([]).reshape(0, k)
    dim_factor = np.zeros(3, dtype=int)
    dim_factor[:dim] = 1
    R = R * dim_factor
    if wavefunc:
        x = np.linspace(-1.1 * R[0], 1.1 * R[0], int(1000))[:, None]
        p = []

    def psi(n, dx, W, x):
        V = W.reshape(
            *(np.append(2 * n + 1, -1)
              ))[:, n[1],
                 n[2], :]  # Get y=z=0 cross section of the wavefunction
        xn = np.arange(-n[0], n[0] + 1)[None] * dx[0]
        psi = 1 / np.sqrt(np.product(dx[:dim])) * np.sinc((x - xn) / dx[0]) @ V
        return psi

    for i in N:
        n = i * np.array([1, 1, 1])  # + np.array([0, 0, 3])
        dx = R / n
        n = n * dim_factor
        V, W = H_solver(n, dx, k, avg)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
        if wavefunc:
            p.append(psi(n, dx, W, x)[:, :level])
    dE = np.diff(E, axis=0)

    if wavefunc:
        return np.array(N), dE, E, x / R[0], p
    else:
        return np.array(N), dE, E


def R_convergence(N, dx):
    # Convergence of energy vs R, to reproduce Fig. 5b in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        Nlist = i * np.array([1, 1, 2]) + np.array([0, 0, 1])
        V, W = H_solver(Nlist, dx)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    R = np.array(N) * dx[0] / w
    return R, dE, E
