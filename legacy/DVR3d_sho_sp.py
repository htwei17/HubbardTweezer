import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.sparse.linalg import LinearOperator

dx0 = 1/3
Nmax = 100
R0 = Nmax * dx0
k = 10
w = 1
# hb = 1, m = 1


def potential(x, y, z):
    # Potential function
    V = 1/2 * (x**2+y**2+z**2)
    return V


def Vmat(n, dx):
    # Potential energy tensor, index order [x y z x' y' z']
    # NOTE: here n is a 3-element np.array s.t. n = [nx, ny, nz]
    x = []
    for i in range(3):
        x.append(np.arange(-n[i], n[i]+1) * dx)
    X = np.meshgrid(x[0], x[1], x[2])
    V = potential(X[0], X[1], X[2]).transpose(
        1, 0, 2)  # 3 index tensor V(x, y, z)
    # NOTE: here x,y,z are in their physical direction, which means x varies horizontally and y varies vertically
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
    # get kinetic energy in unit of v, NOTE: DO GET THE ORDER MATCHES CORRECTLY
    T /= 2 * dx**2
    return T


def Tmat(n, dx):
    # Kinetic energy tensor, index order [x y z x' y' z']
    # NOTE: here n is a 3-element np.array s.t. n = [nx, ny, nz]
    T = []
    for i in range(3):
        if n[i] != 0:
            T.append(Tmat_1d(n[i], dx))  # eg. T_xx'
        else: # If the systems is set to have only 1 grid point in this direction, ie. no such dimension
            T.append(np.zeros((1, 1)))
    return T


def Hop(T, V, n, psi0):
    N = 2*n+1
    psi0 = psi0.reshape(*N)
    psi = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
    psi += np.einsum('ij,jkl->ikl', T[0], psi0)  # T_xx' delta_yy' delta_zz'
    psi += np.einsum('jl,ilk->ijk', T[1], psi0)  # delta_xx' T_yy' delta_zz'
    psi += np.einsum('ij,klj->kli', T[2], psi0)  # delta_xx' delta_yy' T_zz'
    return psi.reshape(-1)


def H_solver(n, dx):
    # Construct and solve Hamiltonian operator
    T = Tmat(n, dx)
    V = Vmat(n, dx)

    def applyH(psi):
        return Hop(T, V, n, psi)

    N = np.product(2*n+1)
    H = LinearOperator((N, N), matvec=applyH)
    [E, W] = la.eigsh(H, k=k, which='SA')
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