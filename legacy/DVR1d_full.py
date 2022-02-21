import numpy as np
import numpy.linalg as la
# import scipy.sparse as sp
# from scipy.sparse.linalg import LinearOperator
# import sparse
from opt_einsum import contract
# from einops import rearrange, reduce, repeat

# Harmonic parameters
dx0 = 1/3
Nmax = 100
R0 = Nmax * dx0
dim = 3
k = 10
w = 1
hb = 1  # Reduced Planck const
m = 1

# Gaussian parameters
# Fundamental constants
a0 = 5.29E-11  # Bohr radius, in unit of meter
Eha = 6579.68392E12  # Hartree energy, in unit of Hz
amu = 1822.89  # atomic mass, in unit of electron mass
m = 6.015122 * amu  # Lithium-6 atom mass, in unit of electron mass
dim = 3  # space dimension
# Atomic units, most parameter scales are estimated from the parameters in the PRA paper
V0 = 1.0452E5/Eha  # 104.52kHz, potential depth, in unit of Hartree energy
# TO GET A REASONABLE ENERGY SCALE, WE SET v=1 AS THE ENERGY UNIT HEREAFTER
w = 1E-6/a0  # ~1000nm, waist length, in unit of Bohr radius
# Lithium-6 atom mass, in unit of electron mass, then V0 is absorbed into it as shown on th denominator of kinetic energy
m = 6.015122 * amu * V0
Nmax = 60  # Max number of grid points
dx0 = 0.12 * w  # fixed delta x in PRA paper convergence benchmark
R0 = 6 * w  # fixed R in PRA paper convergence benchmark
k = 5  # Number of energy levels to track


def Vfun(x, y, z):
    # Potential function
    V = -np.exp(-2*(x**2+y**2+z**2)/w**2)
    return V


def Vmat(n, dx, potential):
    # Potential energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    #       potential(x, y, z) is a function handle to be processed as potential function for solving
    x = []
    for i in range(dim):
        x.append(np.arange(-n[i], n[i]+1) * dx[i])
    X = np.meshgrid(x[0], x[1], x[2], indexing='ij')
    V = potential(X[0], X[1], X[2])  # 3 index tensor V(x, y, z)
    # V * identity rank-6 tensor, sparse
    N = np.product(2*n+1)
    # V = sp.spdiags(V.reshape(-1), 0, N, N)
    # V = sparse.COO(V)
    V = np.diag(V.reshape(-1))
    V = V.reshape(np.concatenate((2*n+1, 2*n+1)))
    return V


def kinetic_offdiag(n):
    # Kinetic energy matrix off-diagonal elements
    T = 2 * np.power(-1., n) / n**2
    return T


def Tmat_1d(n, dx):
    # Kinetic energy matrix for 1-dim
    # Off-diagonal part
    T = np.arange(-n, n+1).reshape(1, -1)
    T = T - T.transpose()
    # Force diagonals to be 1, to avoid warning on 0-dived-0, but this value assignment can be ignored as the diagonal entries are to be replaced in the next step
    T[np.diag_indices(2*n+1)] = 1
    T = kinetic_offdiag(T)
    # Diagonal part
    T[np.diag_indices(2*n+1)] = np.pi**2/3
    # get kinetic energy in unit of V0, the V0 is absorbed in m. NOTE: DO GET THE ORDER MATCHES CORRECTLY
    T *= hb**2 / (2 * m * dx**2)
    return T


def Tmat(n, dx):
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
    delta = []
    T0 = []
    for i in range(dim):
        # delta.append(sparse.COO(sp.eye(2*n[i]+1)))  # eg. delta_xx'
        delta.append(np.eye(2*n[i]+1))  # eg. delta_xx'
        if n[i] > 0:
            # T0.append(sparse.COO(Tmat_1d(n[i], dx[i])))  # eg. T_xx'
            T0.append(Tmat_1d(n[i], dx[i]))  # eg. T_xx'
        # If the systems is set to have only 1 grid point (N = 0) in this direction, ie. no such dimension
        else:
            # T0.append(sparse.COO(np.zeros((1, 1))))
            T0.append(np.zeros((1, 1)))

    # # delta_xx' delta_yy' T_zz'
    # T1 = sparse.tensordot(delta[0], delta[1], axes=0)
    # T = sparse.tensordot(T1, T0[2], axes=0).transpose((0, 2, 4, 1, 3, 5))
    # # delta_xx' T_yy' delta_zz'
    # T2 = sparse.tensordot(delta[0], T0[1], axes=0)
    # T += sparse.tensordot(T2, delta[2], axes=0).transpose((0, 2, 4, 1, 3, 5))
    # # T_xx' delta_yy' delta_zz'
    # T3 = sparse.tensordot(T0[0], delta[1], axes=0)
    # T += sparse.tensordot(T3, delta[2], axes=0).transpose((0, 2, 4, 1, 3, 5))
    
    # delta_xx' delta_yy' T_zz'
    T = contract('ij,kl,mn->ikmjln', delta[0], delta[1], T0[2])
    # delta_xx' T_yy' delta_zz'
    T += contract('ij,kl,mn->ikmjln', delta[0], T0[1], delta[2])
    # T_xx' delta_yy' delta_zz'
    T += contract('ij,kl,mn->ikmjln', T0[0], delta[1], delta[2])
    return T


def H_mat(n, dx, avg=1, potential=Vfun):
    # Construct Hamiltonian matrix

    T = Tmat(n, dx)
    V = Vmat(n, dx, potential)
    H = T + avg * V
    N = np.product(2*n+1)
    H = H.reshape((N, N))
    H = (H + H.T.conj())/2
    # H = H.todense()
    return H


def H_solver(n, dx, avg=1, potential=Vfun):
    # Solve Hamiltonian matrix
    # avg factor is used to control the time average potential strength
    H = H_mat(n, dx, avg, potential)
    [E, W] = la.eigh(H)
    return E, W


def N_convergence(N, R):
    # Convergence of energy vs N, to reproduce Fig. 5a in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        Nlist = i * np.array([1, 1, 1])
        dx = R/i
        V, W = H_solver(Nlist, dx)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    return np.array(N), dE, E


def R_convergence(N, dx):
    # Convergence of energy vs R, to reproduce Fig. 5b in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        Nlist = i * np.array([1, 1, 1])
        V, W = H_solver(Nlist, dx)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    R = np.array(N) * dx0 / w
    return R, dE, E
