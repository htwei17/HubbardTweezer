import numpy as np
import numpy.linalg as la

w = 1
dx0 = 0.4
Nmax = 25
R0 = 8
k = 5
# hb = 1, m = 1
# Harmonic length sqrt(hb / m w) = 1


def potential(x):
    # Potential function
    V = 1 / 2 * w**2 * x**2
    return V


def Vmat(n, dx):
    # Potential energy matrix
    X = np.arange(-n, n + 1) * dx
    V = np.diag(potential(X))
    return V


def kinetic_offdiag(n):
    # Kinetic energy matrix off-diagonal elements
    T = 2 * np.power(-1., n) / n**2
    return T


def Tmat(n, dx):
    # Kinetic energy matrix
    # Off-diagonal part
    T = np.arange(-n, n + 1).reshape(1, -1)
    T = T - T.transpose()
    # Force diagonals to be 1, to avoid warning on 0-dived-0, but this value assignment can be ignored as the diagonal entries are to be replaced in the next step
    T[np.diag_indices(2 * n + 1)] = 1
    T = kinetic_offdiag(T)
    # Diagonal part
    T[np.diag_indices(2 * n + 1)] = np.pi**2 / 3
    T /= 2 * dx**2
    return T


def H_solver(n, dx):
    # Construct and solve Hamiltonian matrix
    T = Tmat(n, dx)
    V = Vmat(n, dx)
    H = T + V
    H = (H + H.transpose()) / 2
    [E, W] = la.eigh(H)
    return E, W


def N_convergence(N, R):
    # Convergence of energy vs N, with R fixed, to reproduce Fig. 5a in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        dx = R / i
        V, W = H_solver(i, dx)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    return np.array(N), dE, E


def R_convergence(N, dx):
    # Convergence of energy vs R, with dx fixed, to reproduce Fig. 5b in PRA paper
    E = np.array([]).reshape(0, k)

    for i in N:
        V, W = H_solver(i, dx)
        E = np.append(E, V[:k].reshape(1, -1), axis=0)
    dE = np.diff(E, axis=0)
    R = np.array(N) * dx0  # in unit of w
    return R, dE, E
