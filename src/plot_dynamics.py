from turtle import color
from attr import NOTHING
import numpy as np
import matplotlib.pyplot as plt
from DVR_exe import *
from matplotlib import gridspec
import h5py
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.colors as colors

mpl.rcParams['figure.dpi'] = 300


class plot(dynamics):

    def __init__(self,
                 cvg='N',
                 quantity='gs',
                 N=10,
                 R: np.ndarray = ...,
                 freq_list: np.ndarray = ...,
                 time=...,
                 avg=1,
                 dim=3,
                 model='Gaussian',
                 trap=...,
                 mem_eff=False,
                 wavefunc=False,
                 realtime=False,
                 symmetry=False,
                 absorber=False,
                 ab_param=...) -> None:
        super().__init__(N, R, freq_list, time, avg, dim, model, trap, mem_eff,
                         wavefunc, realtime, symmetry, absorber, ab_param)
        self.cvg = cvg
        self.quantity = quantity

    def set_each_n(self, N_list, R0_list, i):
        self.update_N(N_list[i])
        self.update_R(R0_list[i])
        if self.cvg is 'N':
            self.cvg_str = 'R={}w'.format(self.R[:self.dim] / self.w)
        elif self.cvg is 'R':
            self.cvg_str = 'dx={}w'.format(self.dx[:self.dim] / self.w)
        self.n_list.append(self.n)
        self.dx_list.append(self.dx)
        # n_list, dx_list are mutable, no need to output
        np.set_printoptions(precision=2, suppress=True)
        if self.model is 'Gaussian':
            self.freq_unit_str = 'kHz'
            self.freq_unit = 1
            self.t_unit = 's'
            self.xlabel = 't/s'
        elif self.model is 'sho':
            self.freq_unit_str = '$\omega$'
            self.freq_unit = 1
            self.t_unit = ''
            self.xlabel = 't/$\omega^{-1}$'
        # return n_list, dx_list

    def set_all_n(self, N_list, R0_list, avg_no, avg):
        self.n_list = []
        self.dx_list = []
        for i in range(len(N_list)):
            self.set_each_n(N_list, R0_list, i)
            self.title1 = '{}D {:g}'.format(
                self.dim, avg_no / self.step_no * self.stop_time)
            self.title1 = self.title1 + self.t_unit
            self.title1 = self.title1 + ' moving-avged {} {} population \n'.format(
                self.model, self.quantity)
            final_str = 'w/ freq={:.3f}'.format(self.freq * self.freq_unit) + self.freq_unit_str \
                            + ' ' + self.cvg_str
            self.title1 = self.title1 + final_str
            self.title2 = '{}D {} {} population \n'.format(
                self.dim, self.model, self.quantity)
            self.title2 = self.title2 + final_str
            if not avg:
                self.title1 = self.title2
                self.title2 = None

    def filename_gen(self, t_step, i):
        self.update_n(self.n_list[i])
        self.update_R(self.R_list[i])
        return super().filename_gen(t_step)


def fit_fun(x, b):
    return np.exp(-x / b)


def avg_data(data, avg_no):
    rho_avg = np.array([])
    for m in range(data.shape[0]):
        rho_avg = moving_avg(data[:m + 1], rho_avg, avg_no)
    return rho_avg


# TODO: use dynamics as object passed to all the plot functions
def plot_dynamics(N_list,
                  R0_list,
                  dvr: plot,
                  length=1,
                  ax=None,
                  fit=True,
                  avg_no=100):

    if avg_no == 0:
        avg_no = 10
        figno = 1
        avg = False
    else:
        figno = 2
        avg = True
    if ax == None:
        fig = plt.figure(figsize=[6 * figno, 5 * dvr.freq_list_len])
        if dvr.freq_list_len == 1:
            subfigs = [fig.subfigures(dvr.freq_list_len, 1)]
        else:
            subfigs = fig.subfigures(dvr.freq_list_len, 1)
        ax_list = []
    else:
        subfigs, ax_list = ax
    if dvr.quantity == 'gs':
        dataidx = 1
    elif dvr.quantity == 'trap':
        dataidx = 2

    N_list = list(N_list)

    for fi in range(dvr.freq_list_len):
        t_step = dvr.set_each_freq(fi)
        # set height ratios for subplots
        if ax == None:
            gs = subfigs[fi].add_gridspec(1, figno, wspace=0)
            axs = gs.subplots(sharey=True)
            axs = [axs]
        else:
            axs = ax_list[fi]

        dvr.set_all_n(N_list, R0_list, avg_no, avg)

        io = Output(wavefunc=dvr.wavefunc)
        fn = lambda i: dvr.filename_gen(i, t_step)
        data = [io.read_file(fn(i), dvr.wavefunc) for i in range(len(N_list))]

        plot_length = int(data[i][0].shape[0] / length)
        # final_val = np.array([])
        lifetime = np.array([])

        for i in range(len(N_list)):
            # final_val = moving_avg(data[i][1], final_val, 16 * avg_no)
            # if i == len(N_list)-1:
            #     np.set_printoptions(precision=6, suppress=False)
            #     print(data[i][1][:plot_length])
            if fit:
                fit_x = data[i][0].reshape(-1)
                fit_y = data[i][dataidx].reshape(-1)
                popt, pcov = curve_fit(fit_fun, fit_x, fit_y)
                lifetime = np.append(lifetime, 1 / popt[-1])
            if avg:
                rho_avg = avg_data(data[i][dataidx], avg_no)
                axs[0].plot(data[i][0][:plot_length],
                            rho_avg[:plot_length][:, None],
                            label='N={}'.format(N_list[i]))
                axs[2].plot(data[i][0][:plot_length],
                            data[i][dataidx][:plot_length],
                            label='N={}'.format(N_list[i]))
            else:
                axs[0].semilogy(data[i][0][:plot_length],
                                data[i][dataidx][:plot_length],
                                label='N={} $V_I$={:.2f}kHz'.format(
                                    N_list[i],
                                    dvr.VI * dvr.V0_SI / dvr.kHz_2p))
                if fit:
                    axs[0].semilogy(
                        data[i][0][:plot_length],
                        fit_fun(data[i][0][:plot_length], *popt),
                        '--',
                        label='fitting N={} $V_I$={:.2f}kHz'.format(
                            N_list[i], dvr.VI * dvr.V0_SI / dvr.kHz_2p),
                        lw=3)
        if fit:
            left, bottom, width, height = [0.3, 0.6, 0.2, 0.2]
            # ax2 = subfigs[fi].add_axes([left, bottom, width, height])
            # ax2.plot(N_list, lifetime)
            # ax2.set_xlabel('N')
            # ax2.set_ylabel('$\\tau/s$')
            # ax2.set_title('Lifetime')
        axs[0].set_title(dvr.title1)
        if ax == None:
            axs[0].axhline(y=1 / np.e, color='gray', label='$\\rho=1/e$')
            axs[0].grid()
        axs[0].legend()
        # plt.savefig('3D_{}.png'.format(model))
        axs[0].set_xlabel(dvr.xlabel)
        axs[0].set_ylabel('$\\rho$')
        if avg:
            axs[2].legend()
            axs[2].set_xlabel(dvr.xlabel)
            axs[2].set_title(dvr.title2)
        ax_list.append(axs)
    plt.savefig('{}d_{}.jpg'.format(dvr.dim, dvr.quantity))
    return subfigs, ax_list


def moving_avg(rho_gs, rho_avg, avg_no):
    # Calculate the moving time-averaged quantities as an array form
    rho_avg = np.append(rho_avg, np.mean(rho_gs[-avg_no:]))
    return rho_avg


def plot_lifetime(N_list,
                  R_list,
                  dvr: plot,
                  ext_ax=None,
                  file=False,
                  err=False,
                  avg_no=10,
                  tau=np.inf):

    N_list = list(N_list)
    # sat_freq = np.array([]).reshape(0, len(N_list))
    lt_vs_freq = np.array([]).reshape(0, len(N_list))

    if err:
        lt_err = [lt_vs_freq, lt_vs_freq]

    fn = 'tau %gd %.1f.csv' % (dim, tau)
    if file:
        no_file = False
        try:
            sav = np.loadtxt(fn, delimiter=',')
        except:
            no_file = True
    else:
        no_file = True

    if dvr.quantity == 'gs':
        dataidx = 1
        wavefunc = False
    elif dvr.quantity == 'trap':
        dataidx = 2
        wavefunc = True

    for fi in range(dvr.freq_list_len):
        t_step = dvr.set_each_freq(fi)

        # NORMAL WAIST
        dvr.V0_SI = 104.52 * dvr.kHz_2p * hb  # 104.52kHz * h, potential depth, in SI unit, since hbar is set to 1 this should be multiplied by 2pi
        dvr.w = 1E-6 / a0  # ~1000nm, waist length, in unit of Bohr radius

        # TODO: CHECK IF R AND L VARYING WITH W CAUSES THE DIFFERENCE ON THE LIFETIME
        lt_vs_freq, cvg_str, freq_SIunit_str, freq_SIunit = tau_from_waist(
            N_list, R_list, dvr, t_step, avg_no, tau, no_file, dataidx,
            lt_vs_freq)
        freq_with_unit = [freq_list[:, None] * freq_SIunit]

        if err:
            # TIGHTEST WAIST
            VSI = 7.6E4 * 2 * np.pi  # trap depth for tightest waist
            w0 = 8.61E-7 / a0  # tightest waist length

            __, lt_err[0], __, __, __ = tau_from_waist(
                N_list, w0, VSI, step_no, dim, model, cvg, symmetry,
                auto_t_step, realtime, avg_no, tau, wavefunc, absorber, ap,
                sat_freq, no_file, dataidx, freq, stop_time, T, R_list,
                lt_err[0])
            freq_with_unit.append(freq_list[:, None] * VSI / 1000)

            # FATTEST WAIST
            VSI = 1.56E5 * 2 * np.pi  # trap depth for fattest waist
            w0 = 1.18E-6 / a0  # fattest waist length

            __, lt_err[1], __, __, __ = tau_from_waist(
                N_list, w0, VSI, step_no, dim, model, cvg, symmetry,
                auto_t_step, realtime, avg_no, tau, wavefunc, absorber, ap,
                sat_freq, no_file, dataidx, freq, stop_time, T, R_list,
                lt_err[1])
            freq_with_unit.append(freq_list[:, None] * VSI / 1000)

    if no_file:
        sav = np.concatenate((freq_list[:, None] * freq_SIunit, lt_vs_freq),
                             axis=1)
        np.savetxt(fn, sav, delimiter=',')

    if ext_ax == None:
        fig = plt.figure(figsize=[8, 6])
        ax = fig.add_subplot()
        fmt = 'o-'
    else:
        ax = ext_ax
        fmt = 's-.'
    for ni in range(len(N_list)):
        if err:
            ax.fill(
                np.append(freq_with_unit[1], freq_with_unit[2][::-1]),
                np.append(lt_err[0][:, ni], lt_err[1][::-1, ni]),
                # interpolate=True,
                alpha=0.3)
        ax.semilogy(sav[:, 0],
                    sav[:, ni + 1],
                    fmt,
                    label='{}D {} N={} $V_I$={:.2f}kHz'.format(
                        dim, quantity, N_list[ni],
                        ab_param[1] * V0_SI / (2 * np.pi * 1E3)))
    # ax.set_ylim([0, 30])
    if ext_ax == None:
        if tau < np.inf:
            ax.axhline(y=tau, color='gray', label='$\\tau=%.2fs$' % tau)
        ax.grid()
        ax.set_xlabel('freq/' + freq_SIunit_str)
        # ax.set_ylabel('$\\rho$')
        ax.set_ylabel('$\\tau/s$')
        # ax.set_ylim([.3, 20])
        # ax.set_xlim([0, 1000])
        ax.set_xlim([70, 240])
        ax = expt_data(ax)

    ax.legend()
    # ax.set_title('Saturation value of {}D {:g}s-averaged {} GS population @ \n\
    # stop time {:g}s '.format(dim, 16 * avg_no / step_no *
    #                  stop_time, model, stop_time) + final_str)
    if ext_ax == None:
        ax.set_title('Lifetime of {}D {} population @'.format(dim, model) +
                     ' ' + cvg_str)
    else:
        ax.set_title(
            'Lifetime of {}D {}\ncompared w/ exp\'t @'.format(dim, model) +
            ' $R_0$={}w'.format((R0_list[0] - ab_param[0])[:dim] / w0))
        # ax.set_title(
        #     'Lifetime of 3D {} GS\n with $\\tau_{{eff}}$ vs w/o $\\tau_{{eff}}$ @'.
        #     format(model) + ' $R_0$={}w'.format(R0 / w))
    # print(freq_list[:, None] * freq_SIunit)
    # print(lt_vs_freq)
    # return freq_list[:, None] * freq_SIunit, lt_vs_freq
    plt.savefig('3d{}d_{}_lt.jpg'.format(dim, quantity))
    return ax


def tau_from_waist(N_list, R_list, dvr: plot, t_step, avg_no, tau, no_file,
                   dataidx, lt_vs_freq):
    if avg_no == 0:
        avg_no = 10
        avg = False
    else:
        avg = True

    dvr.set_all_n(N_list, R_list, avg_no, avg)

    if no_file:
        lt_vs_freq = get_tau(N_list, dvr, avg_no, tau, dataidx, lt_vs_freq,
                             t_step)
    return lt_vs_freq


def get_tau(N_list, dvr: plot, avg_no, tau, dataidx, lt_vs_freq, t_step):
    io = Output(wavefunc=dvr.wavefunc)
    fn = lambda i: dvr.filename_gen(i, t_step)
    data = [io.read_file(fn(i), dvr.wavefunc) for i in range(len(N_list))]

    final_val = np.array([])
    lifetime = np.array([])
    for i in range(len(N_list)):
        final_val = moving_avg(data[i][1], final_val, 16 * avg_no)
        fit_x = data[i][0].reshape(-1)
        fit_y = data[i][dataidx].reshape(-1)
        popt, pcov = curve_fit(fit_fun, fit_x, fit_y, bounds=(1E-5, 1E8))
        lifetime = np.append(lifetime, 1 / (1 / tau + 1 / popt[-1]))
    sat_freq = np.append(sat_freq, final_val[None], axis=0)
    lt_vs_freq = np.append(lt_vs_freq, lifetime[None], axis=0)
    return lt_vs_freq


def expt_data(ax):
    strobe = np.array([300, 200, 400, 150, 175])
    lifetime = np.array([9.89, 7.01, 9.4, .363, 2.93])
    ub = np.array([12.6, 7.54, 11, .409, 3.81])
    err = ub - lifetime

    strobe2 = np.array([175, 160, 300, 500, 1000, 750, 250, 0])
    lifetime2 = np.array([2.66, .844, 11.2, 9.07, 6.44, 15.1, 14.8, 17.9])
    ub2 = np.array([3.02, .866, 12.1, 9.81, 6.56, 15.6, 18.2, 19])
    err2 = ub2 - lifetime2

    strobe3 = np.array([750])
    lifetime3 = np.array([14.6])
    ub3 = np.array([17])
    err3 = ub3 - lifetime3

    ax.errorbar(strobe, lifetime, yerr=err, fmt='v', label='exp\'t 12/12')
    ax.errorbar(strobe2, lifetime2, yerr=err2, fmt='v', label='exp\'t 12/13')
    ax.errorbar(strobe3, lifetime3, yerr=err3, fmt='v', label='exp\'t 12/14')
    # ax.set_yscale('log', nonposy='clip')
    return ax


def plot_wavefunction(N_list,
                      R0_list,
                      freq_list,
                      stop_time_list=0,
                      step_no=1000.0,
                      length=1,
                      dim=1,
                      model='Gaussian',
                      cvg='N',
                      auto_t_step=False,
                      symmetry=False,
                      realtime=False,
                      absorber=False,
                      ab_param=(LI, VI0)):

    freq_list = np.array(freq_list)
    len_freq_list = len(freq_list)
    N_list = list(N_list)
    p = 0
    stop_time_list = copy_to_list(stop_time_list, len_freq_list)
    if symmetry:
        p = 1

    for fi in range(len_freq_list):
        freq, stop_time, period = preset_each_freq(freq_list, stop_time_list,
                                                   fi)
        n_list = []
        dx_list = []
        for i in range(len(N_list)):
            t_step, time_unit, t_step_is_period, cvg_str, freq_unit_str, freq_unit, freq_SIunit_str, freq_SIunit, t_unit, xlabel = preset_each_n(
                N_list, R0_list, step_no, dim, model, cvg, symmetry,
                auto_t_step, stop_time, n_list, dx_list, i)

        fn = find_fn(step_no, model, realtime, True, absorber, ab_param, freq,
                     stop_time, period, n_list, dx_list, t_step, time_unit,
                     t_step_is_period)

        for i in range(len(N_list)):
            t, rho, __, W = read_file(fn(i), wavefunc=True)
            dx = dx_list[i][0]
            R0 = R0_list[i][0] - ab_param[0]
            t_len = int(len(t) / length)
            n_period = int(t[t_len - 1] / (period * time_unit))

            x = np.linspace(-R0_list[i][0], R0_list[i][0], int(1000))[:, None]
            psi_xt = psi(n_list[i], dx, W[:t_len, :].T, x, p)
            psi_xt = abs(psi_xt)**2
            X, T = np.meshgrid(x.reshape(-1) / R0,
                               t.reshape(-1)[:t_len],
                               indexing='ij')
            fig = plt.figure(figsize=[6 * 2, 5])
            sf = fig.subfigures(1, 2)
            ax = sf[0].subplots()
            pcm = ax.pcolormesh(X,
                                T,
                                psi_xt,
                                label='N={}'.format(N_list[i]),
                                norm=colors.LogNorm(vmin=1E-8,
                                                    vmax=psi_xt.max()))
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel('x/R')
            ax.set_ylabel('t/s')
            for i in range(1, n_period + 1, 1):
                ax.axhline(y=i * period * time_unit, color='gray')
            ax.axvline(x=-1, color='w')
            ax.axvline(x=1, color='w')
            gs = sf[1].subplots()
            ab_str = '$\Gamma$={:.2f}kHz'.format(ab_param[1] /
                                                 (1E3 * 2 * np.pi / V0_SI))
            final_str = 'freq={:.3f}'.format(freq * freq_unit) +\
                        freq_unit_str + ' {:.3f}'.format(freq * freq_SIunit) +\
                        freq_SIunit_str + ' ' + cvg_str +\
                         ' ' + ab_str
            ax.set_title('{}D {} GS probability @ \n\
                    stop time {:.2g}s '.format(dim, model, stop_time) +
                         final_str)

            for i in range(5):
                slc = int(i / 5 * X.shape[1])
                gs.plot(X[:, slc],
                        psi_xt[:, slc],
                        label='t={:.2g}s'.format(T[0, slc]))
            gs.legend()
            gs.set_xlabel('x/R')
            gs.set_ylabel('$\\rho$')
            gs.set_title('{}D {} probabilities @ '.format(dim, model) +
                         cvg_str + ' ' + ab_str)
        plt.savefig('{}d_wavefunc_{:.2f}_{:.2f}.jpg'.format(
            dim, freq, ab_param[1] / (2 * np.pi / V0_SI)))
