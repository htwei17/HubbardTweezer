from dynamics import *
from DVR_core import *
import h5py
from time import time
from pympler.asizeof import asizeof
import tracemalloc
from display_top import display_top


def mem_est(n, p):
    n = np.array(n)
    init = get_init(n, p)
    Ndim = np.prod(n + 1 - init)
    mem = 2**(2 * np.log2(Ndim) + 3 - 20)
    print("Matrix size= {}".format(Ndim))
    print(
        "Estimated full matrix memory usage, float: {:.2f} MiB, complex: {:.2f} MiB"
        .format(mem, 2 * mem))


def eigen_list(dvr: dynamics):
    dvr.avg = 1
    E1, W1 = H_solver(dvr)
    # NOTE: this potential indication is needed to determine what parameters are used in kinetic energy
    dvr.avg = 0
    E2, W2 = H_solver(dvr)
    E_list = [E1, E2]
    W_list = [W1, W2]
    print(
        'n={}, dx={}, p={}, model={}, t={} stroboscopic states preparation finished.'
        .format(dvr.n[dvr.nd], dvr.dx[dvr.nd], dvr.p[dvr.nd], dvr.model,
                dvr.stop_time_list))
    print("eigen_list: eigenstates memory usage: {:.2f} MiB.".format(
        asizeof(W_list) / 2**20))
    return E_list, W_list


class Output:

    def __init__(self,
                 t=None,
                 gs=None,
                 wavefunc=False,
                 trap=(None, None)) -> None:
        self.t = t
        self.rho_gs = gs
        self.wavefunc = wavefunc
        if wavefunc:
            self.rho_trap = trap[0]
            self.psi = trap[1]

    def write_to_file(self, fn: str):
        with h5py.File(fn, "a") as f:
            append_to_table(f, 't', self.t)
            # if __debug__:
            #     print('t OK')
            append_to_table(f, 'rho_gs', self.rho_gs)
            # if __debug__:
            #     print('gs OK')
            if self.wavefunc:
                append_to_table(f, 'rho_trap', self.rho_trap)
                # if __debug__:
                #     print('trap OK')
                append_to_table(f,
                                'psi',
                                self.psi.astype(np.complex),
                                dtype=np.complex)
                # if __debug__:
                #     print('wavefunc OK')

    def read_file(self, fn: str, path: str = '../output/'):
        with h5py.File(path + fn, 'r') as f:
            self.t = np.array(f['t'])
            self.rho_gs = np.array(f['rho_gs'])
            if self.wavefunc:
                self.rho_trap = np.array(f['rho_trap'])
                self.psi = np.array(f['psi'])
            else:
                self.rho_trap = None
                self.psi = None


def append_to_table(f, dset, t, dtype=np.float):
    t = np.array([t]).reshape(1, -1)
    if dset in f.keys():
        t_table = f[dset]
        t_table.resize(t_table.shape[0] + 1, axis=0)
        t_table[-1, :] = t
    else:
        f.create_dataset(dset,
                         data=t,
                         dtype=dtype,
                         chunks=True,
                         maxshape=(None, t.shape[1]))


def DVR_exe(dvr: dynamics) -> None:

    tracemalloc.start()

    np.set_printoptions(precision=2, suppress=True)
    print("{}D N={} R0={}w\nfreq={}kHz\n{} potential starts.".format(
        dvr.dim, dvr.N, dvr.R0[dvr.nd], dvr.freq_list, dvr.model))

    time0 = time()

    print("n={}, dx={}w, p={}, model={},\nt={},\nt_step={}\nstarts.".format(
        dvr.n[dvr.nd], dvr.dx[dvr.nd], dvr.p[dvr.nd], dvr.model,
        dvr.stop_time_list, dvr.t_step_list))
    mem_est(dvr.n, dvr.p)

    time1 = time()
    print('Parameter setting time: {:.2f}s.\n'.format(time1 - time0))

    dvr.avg = 1 / 2
    psi0 = dvr.init_state()
    dvr.avg = 1

    time2 = time()
    print('Initial state preparation finished. Time spent: {:.2f}s.\n'.format(
        time2 - time1))

    if dvr.mem_eff:
        print('Memory efficient features are enabled.')
    else:
        E_list, W_list = eigen_list(dvr)
        if dvr.absorber:
            Winv_list = [la.inv(W_list[i]) for i in range(2)]
        else:
            Winv_list = [None, None]

    time3 = time()
    print('Stroboscopic eigensolver time spent: {:.2f}s.\n'.format(time3 -
                                                                   time2))
    mem_check(limit=4)

    if dvr.wavefunc:
        print('Wavefunction output is enabled.')
    if dvr.absorber:
        print(
            'Absorption potential is enabled. Paramter: L={:.2f}w V_OI={:.2f}kHz\n'
            .format(dvr.LI, dvr.VI / dvr.kHz_2p))

    for fi in range(dvr.freq_list_len):  # frequency unit, V0 ~ 104.52kHz
        time4 = time()

        t_step = dvr.set_each_freq(fi)

        cond_str = "freq={:g}kHz model={} ".format(dvr.freq, dvr.model)

        print('\n' + cond_str +
              "starts. Time step={:g}, driving period={:g}.\n".format(
                  t_step, dvr.T))

        dvr.step_count = 0

        if dvr.realtime:  # Calculate real time dynamics, every time evolution operator is calculated individually
            print(cond_str + "detailed dynamics is being calculated.")

            psi = psi0
            psi1 = psi.conj().T

            time5 = time()
            time6 = [time()]

            fn = init_save(dvr, t_step, psi0)

            if dvr.mem_eff:
                U0, U1 = one_period_mem_eff(t_step, t_step, dvr)
            else:
                U0, U1 = one_period_evo(E_list, W_list, t_step, t_step, dvr,
                                        Winv_list)
                if fi == dvr.freq_list_len - 1:
                    del E_list, W_list, Winv_list  # Only enabled when one frequency is calculated

            print(cond_str + "time evolution operator prepared.")
            mem_check(limit=6)

            dynamics_realtime(dvr, t_step, cond_str, psi, psi1, time6, fn, U0,
                              U1)

        else:  # Calculate dynamics at integer times of driving period
            if dvr.mem_eff:
                U = one_period_mem_eff(dvr.t1, dvr.t2, dvr)
            elif dvr.smooth:
                U = one_period_evo_smooth(dvr)
            else:
                U = one_period_evo(E_list, W_list, dvr.t1, dvr.t2, dvr,
                                   Winv_list)
                # Delete when last frequency is calculated
                if fi == dvr.freq_list_len - 1:
                    del E_list, W_list, Winv_list

            print(cond_str + "time evolution operator prepared.")
            mem_check(limit=6)

            if dvr.absorber:
                E, W = la.eig(U)
                del U
                psi = la.inv(W) @ psi0
                psi1 = psi0.conj().T @ W
            else:
                E, W = la.schur(U)
                del U
                E = np.diag(E)
                psi = W.conj().T @ psi0  # transform to eigen basis
                psi1 = psi.conj().T  # get bra of initial state psi0
            if not dvr.wavefunc:
                W = None  # Clear memory-demanding variable W

            time5 = time()
            n_period = int(t_step / dvr.T)
            print(cond_str +
                  'time evolution operator diagonalized. Time spent: {:.2f}s.'.
                  format(time5 - time4))
            print(
                'Time step is set to: {:g}s. Matrix power p in each time step: {:g}.'
                .format(t_step, n_period))

            E_power_n = E**n_period
            del E  # Clear variable E

            time6 = [time()]

            fn = init_save(dvr, t_step, psi0)

            dynamics_period(dvr, t_step, cond_str, psi1, psi, time6, fn, W,
                            n_period, E_power_n)

        time7 = time()
        print(cond_str +
              "finished. Time spent on this freq: {:.2f}s.".format(time7 -
                                                                   time6[0]))
    timef = time()
    print('All done. Total time spent: {:.2f}s.\n'.format(timef - time0))


def init_save(dvr: dynamics, t_step, psi0):
    fn = dvr.filename_gen(t_step)
    io = Output(0, 1, dvr.wavefunc, (1, psi0))
    io.write_to_file(fn)
    return fn


def mem_check(limit=6):
    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot, limit=limit)


def dynamics_period(dvr: dynamics, t_step, cond_str, psi1, psi, time6, fn, W,
                    n_period, E_power_n):
    for t in np.arange(t_step, dvr.stop_time + t_step, t_step):
        psi = dynamics_by_period(n_period,
                                 E_power_n,
                                 W,
                                 psi0=psi,
                                 fixed_period=True)

        rho_gs = abs(psi1 @ psi)**2
        if dvr.wavefunc:
            psi_w = W @ psi
            rho_trap = la.norm(
                psi_w
            )**2  # Roughly estimate state proportion remaining within DVR space
        else:
            rho_trap, psi_w = None, None

        io = Output(t, rho_gs, dvr.wavefunc, (rho_trap, psi_w))
        io.write_to_file(fn)

        dvr.step_count += 1

        print_progress(cond_str, dvr.step_count, time6)


def dynamics_realtime(dvr: dynamics, t_step, cond_str, psi, psi1, time6, fn,
                      U0, U1):
    for t_count in np.arange(t_step, dvr.stop_time + t_step, t_step):
        dvr.step_count += 1
        psi = wavepocket_dynamics(psi, U0, U1, dvr.step_count, dvr.step_no)

        t = t_count * dvr.t_unit
        rho_gs = abs(psi1 @ psi)**2
        rho_trap = la.norm(
            psi
        )**2  # Roughly estimate state proportion remaining within DVR space
        io = Output(t, rho_gs, dvr.wavefunc, (rho_trap, psi))
        io.write_to_file(fn)

        print_progress(cond_str, dvr.step_count, time6)


def print_progress(cond_str, step_count, time6):
    if step_count % 50 == 0:
        time6.append(time())
        print(cond_str + "step={:g} finished. Time spent: {:.2f}s.".format(
            step_count, time6[-1] - time6[-2]))


def one_period_mem_eff(t1, t2, dynamics):
    dynamics.avg = 1
    U0 = mem_eff_int_ops(t1, dynamics)
    dynamics.avg = 0
    if dynamics.realtime:
        U1 = mem_eff_int_ops(t2, dynamics)
        return U0, U1
    else:
        U0 = mem_eff_int_ops(t2, dynamics) @ U0
        return U0
