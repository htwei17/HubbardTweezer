import numpy as np
import numpy.linalg as la

# Atomic units, most parameter scales are estimated from the parameters in the PRA paper
v = 1E-10  # in unit of Hartree energy
# TO GET A REASONABLE ENERGY SCALE, WE SET v=1 AS THE ENERGY UNIT HEREAFTER
w = 1E4  # in unit of Bohr radius
hb = 1
m = 87 * 1823  # in unit of electron mass
Nmax = 60
dx0 = 0.05 * w  # fixed delta x in PRA paper convergence benchmark
R0 = 3 * w  # fixed R in PRA paper convergence benchmark
k = 5  # number of energy levels to track


def potential(x):
    # Potential function
    V = -np.exp(-2*x**2/w**2)
    return V


def Vmat(n, dx):
    # Potential energy matrix
    X = np.arange(-n, n+1) * dx
    V = np.diag(potential(X))
    return V


def kinetic_offdiag(n):
    # Kinetic energy matrix off-diagonal elements
    T = 2 * np.power(-1., n) / n**2
    return T


def Tmat(n, dx):
    # Kinetic energy matrix
    # Off-diagonal part
    T = np.arange(-n, n+1).reshape(1, -1)
    T = T - T.transpose()
    # Force diagonals to be 1, to avoid warning on 0-dived-0, but this value assignment can be ignored as the diagonal entries are to be replaced in the next step
    T[np.diag_indices(2*n+1)] = 1
    T = kinetic_offdiag(T)
    # Diagonal part
    T[np.diag_indices(2*n+1)] = np.pi**2/3
    T *= hb**2 / (2 * m * dx**2 * v)  # get kinetic energy in unit of v
    return T


def H_solver(n, dx):
    # Construct and solve Hamiltonian matrix
    T = Tmat(n, dx)
    V = Vmat(n, dx)
    H = T + V
    H = (H + H.transpose())/2
    [E, W] = la.eigh(H)
    return E, W


def N_convergence(N, R):
    # Convergence of energy vs N, to reproduce Fig. 5a in PRA paper
    dE = np.array([]).reshape(0, k)

    for i in N:
        dx = R/i
        E, W = H_solver(i, dx)
        dE = np.append(dE, E[:k].reshape(1, -1), axis=0)
    dE = np.diff(dE, axis=0)
    return N, dE


def R_convergence(N, dx):
    # Convergence of energy vs R, to reproduce Fig. 5b in PRA paper
    dE = np.array([]).reshape(0, k)

    for i in N:
        E, W = H_solver(i, dx)
        dE = np.append(dE, E[:k].reshape(1, -1), axis=0)
    dE = np.diff(dE, axis=0)
    R = np.array(N) * 0.05  # in unit of w
    return R, dE
