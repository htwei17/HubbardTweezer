from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
import numpy as np
from scipy.optimize import curve_fit

from .io import DVRDynamicsIO
from .dynamics import *
from .wavefunc import psi

params = {
    'figure.dpi': 300,
    # 'figure.figsize': (15, 5),
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large'
}
mpl.rcParams.update(params)


class DVRplot(DVRdynamics):

    def __init__(self, cvg='N', quantity='gs', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cvg = cvg
        self.set_quantity(quantity)

    def set_quantity(self, quantity):
        self.quantity = quantity
        if self.quantity == 'gs':
            self.wavefunc = False
        elif self.quantity == 'trap':
            self.wavefunc = True

    def set_each_n(self, N_list, R0_list, i):
        self.update_N(N_list[i], R0_list[i])
        if self.cvg == 'N':
            self.cvg_str = '$R_0$={}w'.format(self.R0[:self.dim])
        elif self.cvg == 'R':
            self.cvg_str = 'dx={}w'.format(self.dx[:self.dim])
        self.n_list.append(self.n)
        self.dx_list.append(self.dx)
        # n_list, dx_list are mutable, no need to output
        np.set_printoptions(precision=2, suppress=True)
        if self.model == 'Gaussian':
            self.freq_unit_str = '(kHz)'
            self.freq_unit = 1
            self.t_unit = '(s)'
            self.xlabel = '$t$ (s)'
        elif self.model == 'sho':
            self.freq_unit_str = '$\omega$'
            self.freq_unit = 1
            self.t_unit = ''
            self.xlabel = '$t$ ($\omega^{-1}$)'
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
            final_str = 'w/ f={:g}'.format(self.freq * self.freq_unit) + self.freq_unit_str \
                + ' ' + self.cvg_str
            self.title1 = self.title1 + final_str
            self.title2 = '{}D {} {} population \n'.format(
                self.dim, self.model, self.quantity)
            self.title2 = self.title2 + final_str
            if not avg:
                self.title1 = self.title2
                self.title2 = None

    def filename_gen(self, N_list: list, R0_list: list, t_step, i: int):
        self.update_N(N_list[i], R0_list[i])
        return super().filename_gen(t_step)


def fit_fun(x, b, rvs_flg=False):
    if rvs_flg:
        return np.exp(-x * b)
    else:
        return np.exp(-x / b)


def avg_data(data, avg_no):
    rho_avg = np.array([])
    for m in range(data.shape[0]):
        rho_avg = moving_avg(data[:m + 1], rho_avg, avg_no)
    return rho_avg


def plot_dynamics(N_list,
                  R0_list,
                  dvr: DVRplot,
                  length=1,
                  fig=None,
                  fit=True,
                  avg_no=100):

    if avg_no == 0:
        avg_no = 10
        figno = 1
        avg = False
    else:
        figno = 2
        avg = True
    first_fig = False
    if fig == None:
        fig = plt.figure(figsize=[6 * figno, 5 * dvr.freq_list_len])
        first_fig = True
        if dvr.freq_list_len == 1:
            ax_list = np.array(
                [fig.subplots(dvr.freq_list_len, figno, sharey=True)])
        else:
            ax_list = fig.subplots(dvr.freq_list_len, figno, sharey=True)
    else:
        ax_list = fig.axes

    N_list = list(N_list)

    for fi in range(dvr.freq_list_len):
        t_step = dvr.set_each_freq(fi)
        # set height ratios for subplots
        if figno == 1:
            axs = [ax_list[fi]]
        else:
            axs = ax_list[fi, :]

        dvr.set_all_n(N_list, R0_list, avg_no, avg)

        data = get_data(N_list, R0_list, dvr, t_step)
        plot_length = int(data[0].t.shape[0] / length)
        lifetime = np.array([])

        for i in range(len(N_list)):
            if fit:
                lifetime, popt, rvs_flag = fit_tau(lifetime, data[i])
            if avg:
                rho_avg = avg_data(data[i].rho_gs, avg_no)
                axs[0].plot(data[i].t[:plot_length],
                            rho_avg[:plot_length][:, None],
                            label='$N_0$={}'.format(N_list[i]))
                axs[2].plot(data[i].t[:plot_length],
                            data[i].rho_gs[:plot_length],
                            label='$N_0$={}'.format(N_list[i]))
            else:
                if dvr.cvg == 'N':
                    cvg_str = ' $N_0$={} '.format(N_list[i])
                elif dvr.cvg == 'R':
                    cvg_str = ' $R_0$={}w '.format(N_list[i] *
                                                   dvr.dx[:dvr.dim])

                ax_label = ''
                # ax_label += '{}D {}'.format(dvr.dim, dvr.quantity)
                ax_label += cvg_str
                if dvr.smooth:
                    if dvr.smooth_mode == 0:
                        ax_label += 'smooth sqre check'
                    else:
                        ax_label += 'smooth'
                # ax_label += '$\Gamma$={:.2g}kHz'.format(dvr.VI * dvr.V0_SI /
                #                                         dvr.kHz_2p)
                # ax_label += ' '
                # ax_label += 'L={:.2g}w'.format(dvr.L)
                # ax_label += ' '
                # + '$t_{{stop}}$={:.2g}$t_0$'.format(
                #     float(dvr.stop_time /
                #           get_stop_time(np.array([dvr.freq_list[-1]]))))

                axs[0].semilogy(data[i].t[:plot_length],
                                data[i].rho_gs[:plot_length],
                                label=ax_label)

                savdata = np.concatenate(
                    (data[i].t[:plot_length], data[i].rho_gs[:plot_length]),
                    axis=1)
                np.savetxt('{}d_{}.csv'.format(dvr.dim, dvr.freq),
                           savdata,
                           delimiter=',')
                if fit:
                    axs[0].semilogy(data[i].t[:plot_length],
                                    fit_fun(data[i].t[:plot_length],
                                            *popt,
                                            rvs_flg=rvs_flag),
                                    '--',
                                    label='fitting ' + ax_label,
                                    lw=2)
                # axs[0].set_ylim([0.9, 1.1])
        # if fit:
        #     left, bottom, width, height = [0.3, 0.6, 0.2, 0.2]
        #     ax2 = subfigs[fi].add_axes([left, bottom, width, height])
        #     ax2.plot(N_list, lifetime)
        #     ax2.set_xlabel('N')
        #     ax2.set_ylabel('$\\tau/s$')
        #     ax2.set_title('Lifetime')
        axs[0].set_title(dvr.title1)
        if first_fig:
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
        ax_list = np.append(ax_list, axs)
    plt.savefig('{}d_{}.pdf'.format(dvr.dim, dvr.quantity))
    return fig


def get_data(N_list, R0_list, dvr: DVRplot, t_step):
    def fn(i): return dvr.filename_gen(N_list, R0_list, t_step, i)
    data = []
    for i in range(len(N_list)):
        io = DVRDynamicsIO(wavefunc=dvr.wavefunc)
        io.read_file(fn(i))
        data.append(io)
        if dvr.quantity == 'trap':
            data[i].rho_gs = data[i].rho_trap
            data[i].rho_trap = None
    return data


def moving_avg(rho_gs, rho_avg, avg_no):
    # Calculate the moving time-averaged quantities as an array form
    rho_avg = np.append(rho_avg, np.mean(rho_gs[-avg_no:]))
    return rho_avg


def plot_lifetime(N0_list,
                  R0_list,
                  dvr: DVRplot,
                  fig=None,
                  file=False,
                  length=1,
                  err=False,
                  avg_no=10,
                  show_fgr=False,
                  tau=np.inf,
                  extrapolte=None):

    N0_list = list(N0_list)
    lt_vs_freq = np.array([]).reshape(0, len(N0_list))

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

    for fi in range(dvr.freq_list_len):
        t_step = dvr.set_each_freq(fi)

        # NORMAL WAIST
        # 104.52kHz * 2 * pi, potential depth, in unit of angular freq
        dvr.V0 = 104.52 * dvr.kHz_2p
        dvr.w = 1E-6  # ~1000nm, waist length, in unit of Bohr radius
        dvr.wxy = np.ones(2)
        dvr.VIdV0 = dvr.VI / dvr.V0  # Update VI in unit of V0

        # TODO: CHECK IF R AND L VARYING WITH W CAUSES THE DIFFERENCE\ ON THE LIFETIME
        lt_vs_freq = tau_from_waist(N0_list, R0_list, dvr, t_step, avg_no, tau,
                                    length, no_file, lt_vs_freq)

        if err:
            # TIGHTEST WAIST
            dvr.V0 = 76 * dvr.kHz_2p  # trap depth for tightest waist
            dvr.w = 8.61E-7  # tightest waist length
            dvr.wxy = np.ones(2)
            dvr.VIdV0 = dvr.VI / dvr.V0  # Update VI in unit of V0

            lt_err[0] = tau_from_waist(N0_list, R0_list, dvr, t_step, avg_no,
                                       tau, length, no_file, lt_err[0])

            # FATTEST WAIST
            dvr.V0 = 156 * dvr.kHz_2p  # trap depth for fattest waist
            dvr.w = 1.18E-6  # fattest waist length
            dvr.wxy = np.ones(2)
            dvr.VIdV0 = dvr.VI / dvr.V0  # Update VI in unit of V0

            lt_err[1] = tau_from_waist(N0_list, R0_list, dvr, t_step, avg_no,
                                       tau, length, no_file, lt_err[1])

    if no_file:
        if err:
            sav = np.concatenate(
                (dvr.freq_list[:, None], lt_vs_freq, lt_err[0], lt_err[1]),
                axis=1)
        else:
            sav = np.concatenate((dvr.freq_list[:, None], lt_vs_freq), axis=1)
        np.savetxt(fn, sav, delimiter=',')

    first_fig = False
    if fig == None:
        fig = plt.figure(figsize=[8, 6])
        ax = set_axes(fig)
        first_fig = True
    else:
        ax = fig.axes[0]

    if extrapolte != None and isinstance(extrapolte, int):
        Nmin = extrapolte
        ext_lt = np.array([]).reshape(0, 2)
        inset = True
        for i in range(sav.shape[0]):
            fit_x = 1. / np.array(N0_list[Nmin:])
            fit_y = sav[i, 1:][Nmin:]
            # POLY FIT
            # fit = np.polyfit(np.log(fit_x), np.log(fit_y), 1)
            # p = np.poly1d(fit)
            # EXP FIT
            # fit_func = lambda x, a, b: a * np.exp(x * b)
            # popt, pcov = curve_fit(fit_func, fit_x, fit_y)
            # p = lambda x: fit_func(x, *popt)
            # ext = np.array([p(0), abs(p(0) - fit_y[-1])])[None]

            # USE LAST DATAPOINT
            ext = np.array([fit_y[-1], abs(fit_y[-1] - fit_y[-2])])[None]
            ext_lt = np.append(ext_lt, ext, axis=0)
            # if sav[i, 0] >= 220 and inset:
            #     f2 = plt.figure()
            #     ax2 = f2.add_subplot()
            #     # inset = False
            #     # ax2 = inset_axes(ax, width=1.3, height=0.9, loc=4)
            #     ax2.semilogy(1 / np.array(N_list), sav[i, 1:], '.')
            #     x = np.linspace(0, 1 / 15.)
            #     # ax2.semilogy(x, np.exp(p(np.log(x))), '-')
            #     ax2.semilogy(x, p(x), '-')
            #     ax2.grid()
            #     ax2.set_xlabel('1/N')
            #     ax2.set_ylabel('$\\tau/s$')
            #     ax2.set_title('FSS f=%dkHz w/ ' % sav[i, 0] + dvr.cvg_str)
            #     f2.savefig('{}d_{}_{}_fss.jpg'.format(dvr.dim, dvr.cvg,
            #                                            sav[i, 0]))
        ax.errorbar(sav[:, 0],
                    ext_lt[:, 0],
                    yerr=ext_lt[:, 1],
                    label='{}D {} with err'.format(dvr.dim, dvr.quantity)
                    # + ' $\Gamma$={:.2g}kHz'.format(dvr.VI / dvr.kHz_2p)
                    )
    for ni in range(len(N0_list)):
        dvr.update_N(N0_list[ni], N0_list[ni] * dvr.dx)
        if dvr.cvg == 'N':
            arr_str = ndarray_printoout(dvr.n[dvr.nd])
            cvg_str = ' $N$=' + arr_str + ' '
        elif dvr.cvg == 'R':
            arr_str = ndarray_printoout(dvr.R[dvr.nd])
            cvg_str = ' $L$=' + arr_str + '$w_0$ '
        if err:
            ax.fill_between(
                dvr.freq_list.reshape(-1),
                lt_err[0][:, ni],
                lt_err[1][:, ni],
                # interpolate=True,
                alpha=0.3)
        ax_label = ''
        # ax_label += '{}D {}'.format(dvr.dim, dvr.quantity)
        ax_label += cvg_str
        if dvr.smooth:
            if dvr.smooth_mode == 0:
                ax_label += 'sqr smooth'
            elif dvr.smooth_mode == 1:
                ax_label += 'cos smooth'
        # ax_label += '$\Gamma$={:.2g}kHz'.format(dvr.VI /
        #                                         dvr.kHz_2p)
        # ax_label += ' '
        # ax_label += 'L={:.2g}w'.format(dvr.L)
        # ax_label += ' '
        # + '$t_{{stop}}$={:.2g}$t_0$'.format(
        #     float(dvr.stop_time /
        #           get_stop_time(np.array([dvr.freq_list[-1]]))))
        ax.semilogy(sav[:, 0], sav[:, ni + 1], label=ax_label)
    # ax.set_ylim([0, 30])
    if first_fig:
        if tau < np.inf:
            ax.axhline(y=tau, color='gray', label='$\\tau=%.2fs$' % tau)
        if show_fgr:
            ax = expt_data(ax)
            ax = fgr(ax, 1, factor=1E3)  # dvr.dim)
            ax = fgr(ax, 3, factor=1E3)  # dvr.dim)
        ax.grid(visible=True)
        ax.set_xlabel('$f_s$ (kHz)')
        ax.set_ylabel('$\\tau$ (s)')
        # ax.set_ylim([.3, 20])
        # ax.set_xlim([0, 1000])
        ax.set_xlim([80, 250])

    ax.legend(loc='upper left')
    # ax.set_title('Saturation value of {}D {:g}s-averaged {} GS population @ \n\
    # stop time {:g}s '.format(dim, 16 * avg_no / step_no *
    #                  stop_time, model, stop_time) + final_str)
    # ax.set_title(
    #     'Lifetime of {}D {} {} @'.format(dvr.dim, dvr.model, dvr.quantity) +
    #     ' ' + dvr.cvg_str)
    # else:
    #     ax.set_title('Lifetime of {}D {}\ncompared w/ exp\'t @'.format(
    #         dvr.dim, dvr.model) + ' $R_0$={}w'.format(R0_list[0][:dim]))
    # ax.set_title(
    #     'Lifetime of 3D {} GS\n with $\\tau_{{eff}}$ vs w/o $\\tau_{{eff}}$ @'.
    #     format(model) + ' $R_0$={}w'.format(R0 / w))
    # print(freq_list[:, None] * freq_SIunit)
    # print(lt_vs_freq)
    # return freq_list[:, None] * freq_SIunit, lt_vs_freq
    fig.savefig('{}d_{}_{}.pdf'.format(dvr.dim, dvr.cvg, dvr.quantity))
    return fig


def ndarray_printoout(arr: np.ndarray):
    arr_str = np.array2string(arr,
                              separator=',',
                              formatter={'float_kind': lambda x: "%.3g" % x})
    arr_str = arr_str.replace('[', '(')
    arr_str = arr_str.replace(']', ')')
    return arr_str


def set_axes(fig):
    markercycle = cycler(marker=[
        'o', "s", "v", "^", 'D', "p", '+', '*', '.', "<", "d", ">", "8", 'x',
        'l'
    ])
    linecycle = cycler(ls=[
        '-', '--', '-.', ':', (0, (1,
                                   10)), (0,
                                          (5,
                                           10)), (0,
                                                  (5,
                                                   1)), (0, (3, 5, 1, 5, 1, 5))
    ])
    colorcycle = cycler(
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'])

    ax = fig.add_subplot()
    ax.set_prop_cycle(markercycle[:len(linecycle)] +
                      colorcycle[:len(linecycle)] + linecycle)
    # gca()=current axis
    return ax


def tau_from_waist(N_list, R0_list, dvr: DVRplot, t_step, avg_no, tau, length,
                   no_file, lt_vs_freq) -> np.ndarray:
    if avg_no == 0:
        avg_no = 10
        avg = False
    else:
        avg = True

    dvr.set_all_n(N_list, R0_list, avg_no, avg)

    if no_file:
        lt_vs_freq = get_tau(N_list, R0_list, dvr, avg_no, tau, lt_vs_freq,
                             t_step, length)
    return lt_vs_freq


def get_tau(N_list,
            R0_list,
            dvr: DVRplot,
            avg_no,
            tau,
            lt_vs_freq,
            t_step,
            length=1):
    data = get_data(N_list, R0_list, dvr, t_step)

    final_val = np.array([])
    lifetime = np.array([])
    for i in range(len(N_list)):
        final_val = moving_avg(data[i].rho_gs, final_val, 16 * avg_no)
        lifetime, __, __ = fit_tau(lifetime, data[i], tau, length)

    # sat_freq = np.append(sat_freq, final_val[None], axis=0)
    lt_vs_freq = np.append(lt_vs_freq, lifetime[None], axis=0)
    return lt_vs_freq


def fit_tau(lifetime, data, tau=np.inf, length=1):

    def ognl_fit_fun(x, b):
        return fit_fun(x, b, rvs_flg=False)

    def rvs_fit_fun(x, b):
        return fit_fun(x, b, rvs_flg=True)

    fit_x = data.t.reshape(-1)
    fit_length = int(fit_x.shape[0] / length)
    fit_x = fit_x[:fit_length]
    fit_y = data.rho_gs.reshape(-1)[:fit_length]
    rvs_flag = False

    popt, pcov = curve_fit(ognl_fit_fun, fit_x, fit_y, bounds=(1E-5, 1E6))
    # print('popt =', popt[0])
    # print('pcov =', pcov[0][0])
    if pcov[0][-1] < np.inf:
        lifetime = np.append(lifetime, 1 / (1 / tau + 1 / popt[-1]))
    else:
        rvs_flag = True
        popt, pcov = curve_fit(rvs_fit_fun, fit_x, fit_y, bounds=(1E-10, 1))
        lifetime = np.append(lifetime, 1 / (1 / tau + popt[-1]))
    return lifetime, popt, rvs_flag


def expt_data(ax: plt.Axes):
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

    strobe = np.concatenate((strobe, strobe2, strobe3))
    lifetime = np.concatenate((lifetime, lifetime2, lifetime3))
    err = np.concatenate((err, err2, err3))
    ax.errorbar(strobe, lifetime, yerr=err, fmt='D', label='experiment')
    # ax.errorbar(strobe, lifetime, yerr=err, fmt='D', label='exp\'t 12/12')
    # ax.errorbar(strobe2, lifetime2, yerr=err2, fmt='D', label='exp\'t 12/13')
    # ax.errorbar(strobe3, lifetime3, yerr=err3, fmt='D', label='exp\'t 12/14')
    # ax.set_yscale('log', nonposy='clip')
    return ax


def fgr(ax: plt.Axes, dim: int = 3, factor=1) -> plt.Axes:
    w = 1E-6
    zR = 4 * w
    m = 6.015122 * 1.66E-27
    # h = 6.626E-34
    hb = h / (2 * np.pi)
    V = 104.52E3 * 2 * np.pi  # The perturbed V is V
    avg = 1 / 2
    # The time-avged potential is V/2
    f = np.sqrt(avg * hb * V / m) * np.array([2 / w, 2 / w, 1 / zR])[:dim]
    l = np.sqrt(hb / (m * f))
    Eg = np.sum(f) / 2 - avg * V
    # print('V=', V * 1E-3)
    # print('f=', *(f * 1E-3))
    print('Eg=', Eg * 1E-3 / (2 * np.pi))

    if dim == 3:
        # Recover effective Gaussian waist from harmonic trap approx.
        w0 = np.array([w, w, np.sqrt(2) * zR])
        print('w0=', *w0)
        leff2 = 1 / (4 / w0**2 + 1 / l**2)
        print('leff=', *np.sqrt(leff2))

        def fgr_func(freq):
            if not isinstance(freq, Iterable):
                freq = np.array([freq])
            omega = 2 * np.pi * freq * 1E3
            kw = omega + Eg
            kf = np.sqrt(2 * m * kw / hb)

            tau = hb * np.sqrt(leff2[-1] - leff2[0]) / (V**2 * np.pi *
                                                        m) * np.prod(l / leff2)
            tau *= np.exp(2 * m * kw * leff2[0] / hb)
            return tau

    elif dim == 1:
        print('w=', w)
        leff2 = 1 / (4 / w**2 + 1 / l**2)
        print('leff=', *np.sqrt(leff2))

        def fgr_func(freq):
            if not isinstance(freq, Iterable):
                freq = np.array([freq])
            omega = 2 * np.pi * freq * 1E3
            kw = omega + Eg
            tau = 4 * l / (V**2 * leff2) * np.sqrt(hb * kw / (2 * np.pi * m))
            tau *= np.exp(2 * m * kw * leff2 / hb)
            return tau

    x = np.linspace(100, 240)
    ax.plot(x,
            factor * fgr_func(x),
            marker='',
            ls='--',
            label='{:g}x {}D FGR'.format(factor, dim))
    return ax


def plot_wavefunction(N_list, R0_list, dvr: DVRplot, length=1):

    N_list = list(N_list)
    dvr.wavefunc = True
    p = 0
    if dvr.dvr_symm:
        p = 1

    for fi in range(dvr.freq_list_len):
        t_step = dvr.set_each_freq(fi)
        dvr.set_all_n(N_list, R0_list, 0, False)
        def fn(i): return dvr.filename_gen(N_list, R0_list, i, t_step)

        for i in range(len(N_list)):
            io = DVRDynamicsIO(wavefunc=dvr.wavefunc)
            io.read_file(fn(i))

            dx = dvr.dx_list[i][0]
            dvr.update_R0(R0_list[i])
            R = dvr.R[0]
            R0 = dvr.R0[0]
            t_len = int(len(io.t) / length)
            n_period = int(io.t[t_len - 1] / dvr.T)

            x = np.linspace(-R, R, int(1000))[:, None]
            psi_xt = psi(x, dvr.n_list[i], dx, io.psi[:t_len, :].T, p)
            psi_xt = abs(psi_xt)**2
            X, T = np.meshgrid(x.reshape(-1) / R0,
                               io.t.reshape(-1)[:t_len],
                               indexing='ij')
            fig = plt.figure(figsize=[6 * 2, 5])
            sf = fig.subfigures(1, 2)
            ax = sf[0].subplots()
            pcm = ax.pcolormesh(X,
                                T,
                                psi_xt,
                                label='$N_0$={}'.format(N_list[i]),
                                norm=colors.LogNorm(vmin=1E-8,
                                                    vmax=psi_xt.max()))
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel('$x$ ($L$)')
            ax.set_ylabel('$t$ (s)')
            for i in range(1, n_period + 1, 1):
                ax.axhline(y=i * dvr.T, color='gray')
            ax.axvline(x=-1, color='w')
            ax.axvline(x=1, color='w')
            gs = sf[1].subplots()
            ab_str = '$\Gamma$={:.2f}kHz'.format(dvr.VI / dvr.kHz_2p)
            final_str = 'f={:.3f}kHz '.format(
                dvr.freq) + dvr.cvg_str + ' ' + ab_str
            ax.set_title('{}D {} GS probability @ \n\
                    stop time {:.2g}s '.format(dvr.dim, dvr.model,
                                               dvr.stop_time) + final_str)

            for i in range(5):
                slc = int(i / 5 * X.shape[1])
                gs.plot(X[:, slc],
                        psi_xt[:, slc],
                        label='t={:.2g}s'.format(T[0, slc]))
            gs.legend()
            gs.set_xlabel('x/R')
            gs.set_ylabel('$\\rho$')
            gs.set_title('{}D {} probabilities @ '.format(dvr.dim, dvr.model) +
                         dvr.cvg_str + ' ' + ab_str)
        plt.savefig('{}d_wavefunc_{:.2f}_{:.2f}.jpg'.format(
            dvr.dim, dvr.freq, dvr.VI / dvr.kHz_2p))
