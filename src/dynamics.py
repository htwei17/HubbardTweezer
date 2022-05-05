from typing import Iterable
import numpy as np
import scipy.linalg as la
from DVR_full import *
import sys
import copy


class dynamics(DVR):

    def update_N(self, N, R0: np.ndarray):
        self.N = N
        n = np.zeros(3, dtype=int)
        n[:self.dim] = N
        super().update_n(n, R0)

    def __init__(self,
                 N: int = 10,
                 R0: np.ndarray = 3 * np.array([1, 1, 2.4]),
                 freq_list: np.ndarray = np.arange(20, 200, 20),
                 time=(1000.0, 0),
                 avg=1,
                 dim=3,
                 model='Gaussian',
                 trap=(104.52, 1000),
                 mem_eff=False,
                 wavefunc=False,
                 realtime=False,
                 smooth=(-1, 10),
                 symmetry: bool = False,
                 absorber: bool = False,
                 ab_param=(57.04, 1)) -> None:
        # self.R0 = R0.copy()
        # if __debug__:
        #     print(R0)
        print("param_set: model is {} potential.".format(model))

        if model == 'Gaussian' and N == 0:
            N = 10
        elif model == 'sho' and N == 0:
            N = 15

        self.N = N
        n = np.zeros(3, dtype=int)
        n[:dim] = N

        super().__init__(n,
                         R0,
                         avg,
                         model,
                         trap,
                         symmetry=symmetry,
                         absorber=absorber,
                         ab_param=ab_param)
        self.freq_list_len = len(freq_list)
        self.step_no = time[0]
        self.stop_time_list = get_stop_time(freq_list, time[1],
                                            self.V0)  # Time in SI unit
        self.freq_list = np.array(freq_list)
        self.dim = dim
        self.mem_eff = mem_eff
        self.realtime = realtime
        self.wavefunc = wavefunc

        self.t_step = 0
        self.is_period = False

        self.t_step_list = self.stop_time_list / self.step_no

        self.smooth = False
        self.T0, self.Nslice = smooth
        if smooth[0] >= 0:
            self.smooth = True
            self.smooth_mode = smooth[0]

    def init_state(self) -> np.ndarray:
        # Calculate GS of time-averaged potentiala
        print('init_state: initial state of T+%.1fV is calculated.' % self.avg)
        ab = copy.copy(self.absorber)
        self.absorber = False
        __, W = H_solver(self)
        psi = W[:, 0]
        del W
        self.absorber = ab
        return psi

    def set_each_freq(self, fi) -> float:
        self.freq = self.freq_list[fi]
        self.stop_time = self.stop_time_list[fi]
        self.t_step = self.t_step_list[fi]
        # for ax in axs:
        #     ax.label_outer()
        # ax = fig.add_subplot(1, len_freq_list, fi + 1)

        # time period of free system, in unit of s
        self.t1 = 1 / (2 * self.kHz * self.freq)
        # time period of stroboscopic potential system, in unit of s
        self.t2 = 1 / (2 * self.kHz * self.freq)
        self.T = self.t1 + self.t2

        # axs[1].plot(N_list, final_val)
        # axs[1].set_xlabel('N')
        # # ax0.set_ylabel('Final value of averaged $\\rho$')
        # # plt.setp(ax0.get_yticklabels(), visible=False)
        # axs[1].set_title('Saturation of {}D {:g}s-avged {} GS population \n\
        #         @ stop time={:g}s '.format(dim, 16 * avg_no / step_no *
        #                                    stop_time, model, stop_time) +
        #                  final_str)
        # plt.savefig('3d_cvg.jpg')

        return self.set_t_step(self.t_step, self.T)

    def set_t_step(self, t_step: float, T: float) -> float:
        if self.is_period:
            t_step = T
        if self.realtime:
            print("Use detailed dynamics.")
            t_step = T / (2 * self.step_no)
        elif t_step < T:
            str = "Dynamic time step={:g} is too small compared to driving period {:g}. ".format(
                t_step, T)
            print(str + "Set time step to driving period.")
            t_step = T
        else:
            t_step = int(t_step / T) * T  # round t_step to integer times of T
        return t_step

    def filename_gen(self, t_step):
        rt_str = add_str(self.realtime, 'rt')
        sym_str = add_str(self.symmetry, 'sym')
        ab_str = add_str(self.absorber, 'ab', (self.LI, self.VI))
        sm_str = add_str(self.smooth, 'sm', (self.T0, self.Nslice))
        np.set_printoptions(precision=2, suppress=True)
        filename = '{} {} {:g} {:g} {:g} {:.2g} {:.2g} '.format(
            self.n[self.nd], self.dx[self.nd], self.V0 / self.kHz_2p,
            self.w * self.wy, self.freq, self.stop_time, t_step)
        for str in (self.model, rt_str, sym_str, ab_str, sm_str):
            filename += str
        filename += '.h5'
        return filename


def add_str(flag, label, param=None):
    if flag:
        str = ' ' + label
        if isinstance(param, Iterable):
            for item in param:
                str += ' {:.2g}'.format(item)
    else:
        str = ''
    return str


def get_stop_time(freq_list: np.ndarray, t=0, V0=0) -> np.ndarray:
    # NOTE: input freq_list must be in unit of kHz
    if t is 0:
        st = 4E-5 * np.exp(freq_list * 0.085)  # More accurate scaling
        # st = 2.5E-5 * np.exp(
        #     freq_list * 0.0954747)  # Legacy scaling to access 3D data
        # st[np.nonzero(freq_list < 39.4)] = 1E-3
        # if V0 > 1.5E5 * 2 * np.pi:
        #     st *= 2
    else:
        if not isinstance(t, Iterable):
            st = copy_to_list(t, len(freq_list))
        else:
            st = t
    if isinstance(st, Iterable):
        st = np.array(st)
    return st


def copy_to_list(n, copy_time):
    if isinstance(n, Iterable):
        n_list = n * np.ones(copy_time)
        return n_list
    else:
        return n


def int_evo_ops(dvr: dynamics, E, W, t2, Winv=None):
    # interacting dynamics, calculate U by given eigensolution E, W of Hamiltonian
    # NOTE: here energy is in unit of angular frequency, we need to multiply V0
    if t2 > 0:
        if dvr.absorber:
            U1 = W @ (np.exp(-1j * E * dvr.V0 * t2)[:, None] * Winv)
        else:
            U1 = W @ (np.exp(-1j * E * dvr.V0 * t2)[:, None] * W.conj().T)
    else:
        N = W.shape[0]
        U1 = np.eye(N)
    return U1


def exp_tensor(l, a, n):
    real_space_f = np.exp(a * l**2 / (2 * n + 1)**2)
    return real_space_f


# def free_evo_ops(n, dx, t1):
#     # U0_{mn} = FFT^dagger * exp(-i k^2 t / 2m) * FFT, DEPRECATED
#     fft_tensor = []
#     for i in range(dim):
#         if n[i] != 0:
#             a = -4j * hb * np.pi**2 * t1 / (2 * m * dx[i]**2)
#             l = np.arange(-n[i], n[i] + 1)
#             real_space_f = exp_tensor(l, a, n[i])
#         else:
#             real_space_f = np.array([1])
#         # Set the ordering of space from [0, 2N] back to [-N, N]
#         basis = np.roll(np.eye(2 * n[i] + 1), n[i], axis=1)
#         V = np.fft.fft(basis, axis=0, norm="ortho")
#         # For fft function. Set the same ordering for momentum space
#         V = np.roll(V, n[i], axis=0)
#         # V = np.append(V, np.flip(V[1:, :], axis=0).conj(), axis=0) # For rfft function. Accuracy slightly worse
#         fft_tensor.append(V @ (real_space_f[:, None] * V.conj().T))
#     U0 = np.einsum('ij,kl,mn->ikmjln', fft_tensor[0], fft_tensor[1],
#                    fft_tensor[2])
#     # # U0 = \int k |k> e^{-i E t1 / \hbar} <k|
#     # #    = 1/4 (i-1) (m/\pi \hbar t)^{3/2} dx dy dz /4
#     # #      * exp{-i m (x_l - x_m)^2 / 2 \hbar t} * exp{...y} * exp{...z}
#     # dim = np.sum(np.sign(n))
#     # U0 = np.product(dx**np.sign(n)) / 2**dim * (1-1j)**dim * (m/(np.pi * hb * t1))**(dim/2)
#     # x = []
#     # for i in range(3):
#     #     x0 = np.arange(-n[i], n[i]+1) * dx[i]
#     #     xmx = x0[None] - x0[:, None]
#     #     x.append(xmx)
#     # U0 = U0 * np.einsum('ij,kl,mn->ikmjln',
#     #                     exp_tensor(x[0], t1), exp_tensor(x[1], t1), exp_tensor(x[2], t1))
#     N = np.product(2 * n + 1)
#     U0 = U0.reshape(N, N)
#     return U0


def one_period_evo(E_list,
                   W_list,
                   t1,
                   t2,
                   dvr: dynamics,
                   Winv_list=[None, None]):
    # interacting part
    U0 = int_evo_ops(dvr, E_list[0], W_list[0], t1, Winv_list[0])
    # free part
    # U1 = free_evo_ops(n, dx, t1)
    U1 = int_evo_ops(dvr, E_list[1], W_list[1], t2, Winv_list[1])
    if dvr.realtime:
        return U0, U1
    else:
        U = U1 @ U0
        return U


def cos_func(t, T0=0.01):
    # function of t as factor in H = T + f*V
    # t is actually reduced time, t / T
    # f = 1 / (np.exp(-t / T0) + 1) * 1 / (np.exp((t - 0.5) / T0) + 1)
    # f += 1 / (np.exp(-(t - 1) / T0) + 1) * 1 / (np.exp((t - 1.5) / T0) + 1)
    return (1 + np.cos(2 * np.pi * t)) / 2


def sqr_func(t: np.ndarray) -> float:
    if isinstance(t, Iterable):
        t = np.array(t)
    if isinstance(t, np.ndarray):
        ans = np.ones(t.shape)
        ans[t >= 0.5] = 0
    elif isinstance(t, (float, int)):
        if t >= 0.5:
            ans = 0
        else:
            ans = 1
    return ans


def one_period_evo_smooth(dvr: dynamics):
    # f: function of t as factor in H = T + f*V

    def H_mat_w_f(dvr: dynamics, f):
        dvr.avg = f
        return H_mat(dvr)

    dt = dvr.T / dvr.Nslice
    n = np.arange(0, dvr.T, dt) / dvr.T
    if dvr.smooth_mode == 1:
        ft = cos_func(n, dvr.T0)
    elif dvr.smooth_mode == 0:
        ft = sqr_func(n)

    no = dvr.n + 1 - dvr.init
    N = np.prod(no)
    U = np.eye(N)
    for i in range(dvr.Nslice):
        H = H_mat_w_f(dvr, ft[i])

        # H in unit of V0, ie. angular frequency
        U = la.expm(-1j * dt * H * dvr.V0) @ U
    return U


def mem_eff_int_ops(t1, DVR):
    E1, W1 = H_solver(DVR)
    if DVR.absorber:
        Winv = la.inv(W1)
    else:
        Winv = None
    U0 = int_evo_ops(t1, E1, W1, DVR.absorber, Winv)
    del E1, W1, Winv
    return U0


def dynamics_by_period(n_period,
                       E,
                       W,
                       psi0,
                       dense_output=False,
                       fixed_period=False,
                       fast_power=False):
    # Solve matrix power by given eigenenergies E and eigenvectors
    # n_period = int(t / T)
    if n_period > 0:
        if fixed_period:
            # If every time we only need to evolve the state by a fixed amount of time, we can directly save the matrix E^n_period, rather than calculate it every time on-the-fly
            # NOTE: here E is the matrix E^n_period, the powered diagonal eigenvalue matrix of U
            if len(E.shape) == 2:
                psi0 = E @ psi0
            elif len(E.shape) == 1:
                psi0 = E * psi0
        elif dense_output:
            # In the dense output mode, input psi0 is actually W^\dagger |psi>, and the output psi0 needs to be acted by W to get real final |psi>
            # NOTE: here psi0 is in the eigenbasis of U
            if len(E.shape) == 2:
                psi0 = E**n_period @ psi0
            elif len(E.shape) == 1:
                psi0 = E**n_period * psi0
        else:
            if fast_power:
                # Solve matrix power by fast power method, useful when n_period is small (<100)
                # NOTE: here E is the original U, the time evolution operator
                U = la.matrix_power(E, n_period)  # W is left unused
            else:
                # U = W @ ((E**n_period)[:, None] * W.conj().T)
                U = W @ E**n_period @ W.conj().T
            psi0 = U @ psi0
    return psi0


# def dynamics_by_short_time(n_period, U, psi0):
#     # Solve matrix power by fast power method, useful when n_period is small (<100)
#     # n_period = int(t / T)
#     if n_period > 0:
#         W = la.matrix_power(U, n_period)
#         psi0 = W @ psi0
#     return psi0


def measure_lifetime(freq, psi0, n, dx, t_step):
    t1 = 1 / (2 * freq)  # time period of free system
    t2 = 1 / (2 * freq)  # time period of stroboscopic potential system
    if t_step < (t1 + t2):
        print("Dynamic time step is too small. Time step is increased.")
        t_step = t1 + t2
    U = one_period_evo(n, dx, t1, t2)
    E, W = la.schur(U)
    rho_rt = np.array([1])
    psi = psi0
    t = np.array([0])
    t_count = 0
    n_period = int(t_step / (t1 + t2))
    print('matrix power p in each time step: {:g}.'.format(n_period))
    while np.abs(rho_rt[-1])**2 > np.exp(-1, dtype=float) and t_count < 1E7:
        psi = dynamics_by_period(n_period, E, W, psi)
        rho_rt = np.append(rho_rt, psi0.conj().T @ psi)
        t_count += t_step
        t = np.append(t, t_count)
    return t[-1], rho_rt[-1]


def lifetime_vs_frequency(freq_list: np.ndarray, t_step, dvr: dynamics):
    psi0 = dvr.init_state()
    lifetime = np.array([]).reshape(0, 2)
    for freq in freq_list:
        tau, rho_rt = measure_lifetime(freq, psi0, dvr.n, dvr.dx, t_step)
        lifetime = np.append(lifetime, np.array([[freq, tau]]), axis=0)
    return lifetime


# def strob_vs_const(freq, dvr: dynamics, t):
#     # Compute the final state of stroboscopic time evolution, comparing with a time-avged potential

#     # Stroboscopic
#     psi0 = dvr.initial_state()
#     t1 = 1 / (2 * freq)  # time period of free system
#     t2 = 1 / (2 * freq)  # time period of stroboscopic potential system
#     U1 = one_period_evo(n, dx, t1, t2)
#     E1, W1 = la.schur(U1)
#     psi1 = dynamics_by_period(int(t / (t1 + t2)), E1, W1, psi0)
#     n_period = int(t / (t1 + t2))
#     U1 = W1 @ ((E1**n_period)[:, None] * W1.conj().T)

#     # Constant
#     U2 = int_evo_ops(n, dx, t, 1 / 2)
#     E2, W2 = la.schur(U2)
#     # print("Norm difference b/t two time evolution ops:",
#     #       np.linalg.norm(U1-U2, ord='fro'))
#     psi2 = dynamics_by_period(1, E2, W2, psi0)
#     # print("Final state overlap by eigenstate evolution:",
#     #       np.abs(psi2.conj().T @ psi0))
#     return np.abs(psi1.conj().T @ psi2)


def shftdVfun(x, y, z, x0):
    return DVR.Vfun(x - x0[0], y - x0[1], z - x0[2])


def coherent_state(n, dx, x0):

    def firstfun(x, y, z):
        return shftdVfun(x, y, z, x0)

    E, W = H_solver(n, dx, potential=firstfun, model='sho')
    psi = W[:, 0]
    return psi


def wavepocket_dynamics(psi0, U0, U1, step_count, step_no):
    r = int(step_count // step_no)
    if r % 2 == 0:  # quotient is even, means time falls in the 1st half of period
        psi0 = U0 @ psi0
    else:  # quotient is odd, means time falls in the 2nd half of period
        psi0 = U1 @ psi0
    return psi0


# def free_vs_0V(n, dx, t):
#     # Compare the final state and evolution operator of zero potential time evolution vs FFT time evolution operator, DEPRECATED

#     psi0, E0, W0 = initial_state(n, dx, 0)

#     # H = T + 0 * V
#     U1 = int_evo_ops(n, dx, t, 0)
#     E1, W1 = la.schur(U1)
#     psi1 = dynamics_by_long_time(1, E1, W1, psi0)

#     # Free FFT
#     U2 = free_evo_ops(n, dx, t)
#     E2, W2 = la.schur(U2)
#     # print("Norm difference b/t two time evolution ops:",
#     #       np.linalg.norm(U1-U2, ord='fro'))
#     psi2 = dynamics_by_long_time(1, E2, W2, psi0)
#     # print("Final state overlap by eigenstate evolution:",
#     #       np.abs(psi2.conj().T @ psi0))
#     return np.abs(psi1.conj().T @ psi2)
