import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.linalg.basic import matmul_toeplitz
from DVR1d_full import *

# TODO: set the potential as a tunable function parameter


class herm_array(np.ndarray):
    @property
    def H(self):
        return self.conj().T


def initial_state(n, dx, avg=1/2):
    # Calculate GS of time-averaged potential
    E, W = H_solver(n, dx, avg)
    psi = W[:, 0]
    return psi, E, W


def int_evo_ops(n, dx, t2, avg=1, potential=Vfun):
    # interacting dynamics
    # NOTE: here time scale is in unit of 1/V0
    if t2 > 0:
        E, W = H_solver(n, dx, avg, potential)
        U1 = W @ (np.exp(-1j * E * t2 / hb)[:, None] * W.conj().T)
        # Pade approx., might be faster than diagonalization?
        # U1 = sla.expm(-1j * H_mat(n, dx, avg, potential) * t2 / hb)
    else:
        N = np.product(2*n+1)
        U1 = np.eye(N)
    return U1


def exp_tensor(l, a, n):
    real_space_f = np.exp(a * l**2/(2*n+1)**2)
    return real_space_f


def free_evo_ops(n, dx, t1):
    # U0_{mn} = FFT^dagger * exp(-i k^2 t / 2m) * FFT, DEPRECATED
    fft_tensor = []
    for i in range(dim):
        if n[i] != 0:
            a = -4j * hb * np.pi**2 * t1/(2*m*dx[i]**2)
            l = np.arange(-n[i], n[i]+1)
            real_space_f = exp_tensor(l, a, n[i])
        else:
            real_space_f = np.array([1])
        # Set the ordering of space from [0, 2N] back to [-N, N]
        basis = np.roll(np.eye(2 * n[i] + 1), n[i], axis=1)
        V = np.fft.fft(basis, axis=0, norm="ortho")
        # For fft function. Set the same ordering for momentum space
        V = np.roll(V, n[i], axis=0)
        # V = np.append(V, np.flip(V[1:, :], axis=0).conj(), axis=0) # For rfft function. Accuracy slightly worse
        fft_tensor.append(V @ (real_space_f[:, None] * V.conj().T))
    U0 = np.einsum('ij,kl,mn->ikmjln',
                   fft_tensor[0], fft_tensor[1], fft_tensor[2])
    N = np.product(2*n+1)
    U0 = U0.reshape(N, N)
    return U0


def one_period_evo(n, dx, t1, t2):
    # NOTE: In interacting part we scale the energy by V0, which means time is also rescaled to 1/V0
    # interacting part
    U1 = int_evo_ops(n, dx, t2)

    # free part
    # U0 = free_evo_ops(n, dx, t1)
    U0 = int_evo_ops(n, dx, t1, 0)

    U = U0 @ U1
    return U


def dynamics_by_long_time(t, T, E, W, psi0):
    # Solve matrix power by given eigenenergies E and eigenvectors
    n_period = int(t/T)
    if n_period > 0:
        # U = W @ ((E**n_period)[:, None] * W.conj().T)
        U = W @ E**n_period @ W.conj().T
        psi0 = U @ psi0
    return psi0


def dynamics_by_short_time(t, T, U, psi0):
    # Solve matrix power by fast power method, useful when n_period is small (<100)
    n_period = int(t/T)
    if n_period > 0:
        W = la.matrix_power(U, n_period)
        psi0 = W @ psi0
    return psi0


def measure_lifetime(freq, psi0, n, dx, t_step):
    t1 = 1/(2*freq)  # time period of free system
    t2 = 1/(2*freq)  # time period of stroboscopic potential system
    if t_step < (t1+t2):
        print("Dynamic time step is too small.")
        return None, None
    U = one_period_evo(n, dx, t1, t2)
    E, W = sla.schur(U)
    rho_rt = np.array([1])
    psi = psi0
    t = np.array([0])
    t_count = 0
    while np.abs(rho_rt[-1])**2 > np.exp(-1, dtype=float) and t_count < 1E7:
        psi = dynamics_by_long_time(t_step, t1+t2, E, W, psi)
        rho_rt = np.append(rho_rt, psi0.conj().T @ psi)
        t_count += t_step
        t = np.append(t, t_count)
    return t[-1], rho_rt[-1]


def lifetime_vs_frequency(freq_list, t_step, n, dx):
    psi0, E0, W0 = initial_state(n, dx)
    lifetime = np.array([]).reshape(0, 2)
    for freq in freq_list:
        tau, rho_rt = measure_lifetime(freq, psi0, n, dx, t_step)
        lifetime = np.append(lifetime, np.array([[freq, tau]]), axis=0)
    return lifetime


def strob_vs_const(freq, n, dx, t):
    # Compute the final state of stroboscopic time evolution, comparing with a time-avged potential

    # Stroboscopic
    psi0, E0, W0 = initial_state(n, dx)
    t1 = 1/(2*freq)  # time period of free system
    t2 = 1/(2*freq)  # time period of stroboscopic potential system
    U1 = one_period_evo(n, dx, t1, t2)
    E1, W1 = sla.schur(U1)
    psi1 = dynamics_by_long_time(t, t1+t2, E1, W1, psi0)
    n_period = int(t/(t1+t2))
    U1 = W1 @ ((E1**n_period)[:, None] * W1.conj().T)

    # Constant
    U2 = int_evo_ops(n, dx, t, 1/2)
    E2, W2 = sla.schur(U2)
    # print("Norm difference b/t two time evolution ops:",
    #       np.linalg.norm(U1-U2, ord='fro'))
    psi2 = dynamics_by_long_time(t, t, E2, W2, psi0)
    # print("Final state overlap by eigenstate evolution:",
    #       np.abs(psi2.conj().T @ psi0))
    return np.abs(psi1.conj().T @ psi2)


def shftdVfun(x, y, z, x0):
    return Vfun(x-x0, y, z)


def coherent_state_dynamics(n, dx, t1, t2, x0):
    def firstfun(x, y, z):
        return shftdVfun(x, y, z, x0)

    def lastfun(x, y, z):
        return shftdVfun(x, y, z, -x0)

    U0 = int_evo_ops(n, dx, t1, potential=firstfun)
    U1 = int_evo_ops(n, dx, t2, potential=lastfun)

    U = U1 @ U0
    return U


def coherent_state(n, dx, x0):
    def firstfun(x, y, z):
        return shftdVfun(x, y, z, x0)

    E, W = H_solver(n, dx, potential=firstfun)
    psi = W[:, 0]
    return psi, E, W


def wavepocket_dynamics(n, dx, psi0, t, T, x0):
    t1 = T/2
    t2 = t % T
    # print('t={:g}'.format(t))
    U = coherent_state_dynamics(n, dx, t1, t1, x0)
    n_period = int(t/T)
    if n_period > 100:
        # Diagonalizing unitary matrix using Schur decomposition
        E, W = sla.schur(U)
        psi1 = dynamics_by_long_time(n_period * T, T, E, W, psi0)
    else:
        psi1 = dynamics_by_short_time(n_period * T, T, U, psi0)
    if t2 > t1:
        U1 = coherent_state_dynamics(n, dx, t1, t2 - t1, x0)
    else:
        U1 = coherent_state_dynamics(n, dx, t2, 0, x0)
    psi1 = U1 @ psi1
    return psi1


def free_vs_0V(n, dx, t):
    # Compare the final state and evolution operator of zero potential time evolution vs FFT time evolution operator, DEPRECATED

    psi0, E0, W0 = initial_state(n, dx, 0)

    # H = T + 0 * V
    U1 = int_evo_ops(n, dx, t, 0)
    E1, W1 = sla.schur(U1)
    psi1 = dynamics_by_long_time(t, t, E1, W1, psi0)

    # Free FFT
    U2 = free_evo_ops(n, dx, t)
    E2, W2 = sla.schur(U2)
    # print("Norm difference b/t two time evolution ops:",
    #       np.linalg.norm(U1-U2, ord='fro'))
    psi2 = dynamics_by_long_time(t, t, E2, W2, psi0)
    # print("Final state overlap by eigenstate evolution:",
    #       np.abs(psi2.conj().T @ psi0))
    return np.abs(psi1.conj().T @ psi2)
