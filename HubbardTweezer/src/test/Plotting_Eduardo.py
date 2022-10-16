"""
Created on Wed Mar 16 09:39:39 2022
@author: eibarragp

Plotting modules for Hao-Tian Wei
Using Mac OS or Windows will requiere getting different packages. 
1) The os package is not necessary but is how I access my data
2) First example has for example usage of insets and defining axes
3) Second example is slightly simpler but illustrates simple figures.
4) With colormaps and logarithmic scales + fancy legends.
5) Probably you'll need a combo of 1 and 4.
"""

import numpy as np
import os
import pandas as pd

# For plotting in general
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                   mark_inset,
                                                   zoomed_inset_axes)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FixedLocator, FuncFormatter,
                               LogLocator, NullFormatter)
from matplotlib.legend_handler import HandlerTuple

# For making inset plots
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                   mark_inset)
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

#import scipy.ndimage as ndimage
from scipy.optimize import root
from configobj import ConfigObj
#from scipy.signal import savgol_filter
from scipy.integrate import quad
from scipy.special import binom

plt.close('all')
## Needed for using latex and python in MAC OS
os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'

# For formating the text and having latex capabilities
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

source_path = os.getcwd()
folders_list = os.listdir(source_path)

########### FINITE SIZE EFFECTS ################
fs_eff_folder = 'DOS_ED_Finite_size_effect_analysis'  # Using DOS and Nptcl


def data_dict(source_path, folder, Nsp):
    """
    Given a source path, folde and Nspecies returns a
    dictionary with the arrays for different system sizes.
    Results are presented after adiabatic loading calculation.
    """
    current_path = os.path.join(source_path, folder)
    files = os.listdir(current_path)
    data = {}
    for file in files:
        if '.csv' in file and 'SU' + str(Nsp) in file:
            key = int(file.split("_")[0][1])
            try:
                data[key] = csv_reader(os.path.join(current_path, file))
            except:
                print(str(key) + 'is corrupted')
    return data


def finite_size_subplot(fss_SU2, fss_SU6):
    # Fontsize
    fontlegend = 13
    fontlabel = 14
    # Import data
    fs_data_SU2 = data_dict(source_path, fs_eff_folder, 2)
    fs_data_SU6 = data_dict(source_path, fs_eff_folder, 6)
    # Finite size scaled data
    fss_SU2_S = fss_SU2[0]
    fss_SU2_A = fss_SU2[1]
    fss_SU2_I = fss_SU2[3]
    fss_SU6_S = fss_SU6[0]
    fss_SU6_A = fss_SU6[1]
    fss_SU6_I = fss_SU6[3]

    # Finite size scaling at S = 1.427
    single_fss = fss_one_S(fs_data_SU6, 1, 6, 5)
    # Keys dictionary
    val_keys = {'T': 0, 'col1': 1, 'Ntot': 2, 'S': 3, 'A': 4, 'I': 5}
    x_key = 'S'
    # Make figures
    w, h = 7.2, 5.4
    #w,h=8,6
    fig, axs = plt.subplots(2,
                            2,
                            figsize=(w, h),
                            gridspec_kw={
                                'hspace': 0.0,
                                'wspace': 0.0
                            },
                            sharex='col',
                            sharey='row')

    # Plot fss
    axs[0, 0].errorbar(fss_SU2_S,
                       fss_SU2_A,
                       color='green',
                       linestyle='solid',
                       linewidth=2,
                       fmt='o',
                       ms=3,
                       label='fss')
    axs[1, 0].errorbar(fss_SU2_S,
                       fss_SU2_I,
                       color='green',
                       linestyle='solid',
                       linewidth=2,
                       fmt='o',
                       ms=3)
    axs[0, 1].errorbar(fss_SU6_S,
                       fss_SU6_A,
                       color='green',
                       linestyle='solid',
                       linewidth=2,
                       fmt='o',
                       ms=3)
    axs[1, 1].errorbar(fss_SU6_S,
                       fss_SU6_I,
                       color='green',
                       linestyle='solid',
                       linewidth=2,
                       fmt='o',
                       ms=3)

    # Make limits
    axs[0, 0].set_xlim(0.05, 2.0)
    axs[0, 1].set_xlim(0.5, 3.5)
    axs[0, 0].set_ylim(0, 0.8)
    axs[1, 0].set_ylim(0, 1)

    # Put ticks on x
    axs[0, 0].set_xticks(np.arange(0.0, 2.0, 0.5))
    axs[1, 0].set_xticks(np.arange(0.0, 2.0, 0.5))
    axs[0, 1].set_xticks(np.arange(0.5, 4.0, 0.5))
    axs[1, 1].set_xticks(np.arange(0.5, 4.0, 0.5))

    # Put ticks on y
    axs[0, 0].set_yticks(np.arange(0.2, 0.8, 0.2))
    axs[1, 0].set_yticks(np.arange(0.2, 1.2, 0.2))
    axs[0, 1].set_yticks(np.arange(0.2, 0.8, 0.2))
    axs[1, 1].set_yticks(np.arange(0.2, 1.2, 0.2))

    # Make ticks look the same
    axs[0, 0].tick_params(axis='both',
                          which='major',
                          direction='in',
                          top=True,
                          right=True,
                          length=4,
                          width=1.1,
                          labelsize=fontlegend)
    axs[1, 0].tick_params(axis='both',
                          which='major',
                          direction='in',
                          top=True,
                          right=True,
                          length=4,
                          width=1.1,
                          labelsize=fontlegend)
    axs[0, 1].tick_params(axis='both',
                          which='major',
                          direction='in',
                          top=True,
                          right=True,
                          length=4,
                          width=1.1,
                          labelsize=fontlegend)
    axs[1, 1].tick_params(axis='both',
                          which='major',
                          direction='in',
                          top=True,
                          right=True,
                          length=4,
                          width=1.1,
                          labelsize=fontlegend)

    for row in range(0, 2):
        for col in range(0, 2):
            if col == 0:
                keys = fs_data_SU2.keys()
                fs_data = fs_data_SU2
                #flag = True # For data using harmonic trap (Haotian data)
                flag = False  # My data using the DOS
            else:
                keys = fs_data_SU6.keys()
                fs_data = fs_data_SU6
                flag = False
            #lines = ["-","--","-.",":"]
            lines = ["-"]
            linecycler = cycle(lines)
            if fs_eff_folder == 'ED_Finite_size_effect_analysis_old':
                color = iter(plt.cm.coolwarm(np.linspace(0, 1, len(keys) - 3)))
                key_array = sorted(keys)[3:]
            else:
                color = iter(plt.cm.coolwarm(np.linspace(0, 1, len(keys))))
                key_array = sorted(keys)
            for key in key_array:
                #print(key_array)
                c_data = fs_data[key]
                if row == 0:
                    y_key = 'A'
                else:
                    y_key = 'I'
                x = c_data[:, val_keys[x_key]]
                y = 2 * c_data[:, val_keys[y_key]]
                if (y_key == 'I' and fs_eff_folder
                        == 'ED_Finite_size_effect_analysis_old') or (
                            y_key == 'I' and flag):
                    y = c_data[:, val_keys[y_key] + int(np.floor(key / 2)) - 1]
                elif y_key == 'I':
                    y = 2 * (2 * c_data[:, val_keys['A']]) / (
                        y + 2 * c_data[:, val_keys['A']])
                if col == 0 and row == 0:
                    axs[row,
                        col].plot(x,
                                  y,
                                  linestyle=next(linecycler),
                                  linewidth=2,
                                  color=next(color),
                                  label=r"$\displaystyle L = $ " + str(key))
                    axs[row, col].legend(edgecolor='black',
                                         loc='best',
                                         fontsize=fontlegend - 1)
                    #axs[row,col].legend(edgecolor='black',loc=9,bbox_to_anchor=(1,1.25),ncol=5,fontsize=10)
                    axs[row, col].set_title("SU(2)", fontsize=fontlabel)
                else:
                    axs[row, col].plot(x,
                                       y,
                                       linestyle=next(linecycler),
                                       linewidth=2,
                                       color=next(color))
                if col == 1 and row == 0:
                    axs[row, col].set_title("SU(6)", fontsize=fontlabel)
                if y_key == 'A' and col == 0:
                    axs[row, col].set_ylabel('STO amplitude $A$',
                                             fontsize=fontlabel)
                if y_key == 'I' and col == 0:
                    axs[row, col].set_ylabel('STO imbalance $I$',
                                             fontsize=fontlabel)
                if row == 1:
                    axs[row, col].set_xlabel(
                        r"$\displaystyle S/N_{\mathrm{ptcl}}k_B$",
                        fontsize=fontlabel)

    L_inv = single_fss['L_inv']
    L_array = np.linspace(0, 0.2, 10)
    Obs_array = 2 * single_fss['A']['data']
    x0, m = 2 * single_fss['A']['fit'][0][0], 2 * single_fss['A']['fit'][0][1]

    ax1 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(axs[0, 1], [0.49, 0.59, 0.48, 0.36])
    ax1.set_axes_locator(ip)
    ax1.errorbar(L_inv, Obs_array, color='black', fmt='o')
    ax1.errorbar(L_array, x0 + m * L_array, color='black', linestyle='solid')
    ax1.set_xlim(0, 0.21)
    ax1.set_xticks(np.arange(0, 0.3, 0.1))
    ax1.set_yticks(np.arange(0.58, 0.72, 0.04))
    #ax1.set_yticks(np.arange(0.16,0.17,0.005))
    ax1.tick_params(axis='both', which='major', labelsize=fontlegend - 3)
    ax1.set_xlabel(r"$\displaystyle 1/L$", fontsize=fontlegend - 3)
    ax1.set_ylabel(r"$\displaystyle A$", fontsize=fontlegend - 3)
    #ax1.text(0.076,0.307, r"$\displaystyle S/N_{\mathrm{ptcl}}k_B = 1$",fontsize=8)
    ax1.text(0.02,
             0.667,
             r"$\displaystyle S/N_{\mathrm{ptcl}}k_B = 1$",
             fontsize=fontlegend - 3)
    ax1.tick_params(axis='both',
                    which='major',
                    direction='in',
                    top=True,
                    right=True)
    #axs[0,1].axhline(0.171)
    #axs[0,1].axvline(1.8)

    axs[0, 0].text(0.03,
                   0.08,
                   "(a)",
                   fontsize=fontlegend,
                   transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.03,
                   0.08,
                   "(b)",
                   fontsize=fontlegend,
                   transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.03,
                   0.08,
                   "(c)",
                   fontsize=fontlegend,
                   transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.03,
                   0.08,
                   "(d)",
                   fontsize=fontlegend,
                   transform=axs[1, 1].transAxes)

    #plt.tight_layout()
    #fig.suptitle("Finite size effects 1D",fontsize=16,y=0.97)
    #fig.suptitle("Finite size effects 1D, " + r"$\displaystyle U/t=15.3$",fontsize=16,y=0.97)
    #plt.savefig('SU2a6_U15p3_1D_finite_size_final.pdf',dpi=300,bbox_inches="tight")
    return None


def AvsS_talk():
    # Make subplots with inset
    w, h = 5, 4.2
    fig, ax = plt.subplots(1, 1, figsize=((w, h)))
    ax.errorbar(data_1D_I_S,
                data_1D_I,
                xerr=data_1D_I_S_err,
                yerr=data_1D_I_err,
                color='green',
                mfc='white',
                mec='green',
                fmt='s',
                mew=1.5,
                ms=7,
                label='SU(6) 1D')
    ax.errorbar(data_2_I_S,
                data_2_I,
                xerr=data_2_I_S_err,
                yerr=data_2_I_err,
                color='blue',
                mfc='white',
                mec='blue',
                fmt='^',
                mew=1.5,
                ms=7,
                label='SU(2) 1D')
    ax.errorbar(data_ED_6_s,
                data_ED_6_I,
                color='green',
                linestyle='solid',
                linewidth=2)
    ax.errorbar(data_ED_2_s,
                data_ED_2_I,
                color='blue',
                linestyle='solid',
                linewidth=2)
    ax.fill_between(data_ED_2_s,
                    data_ED_2_I - data_ED_2_Ierr[0],
                    data_ED_2_I,
                    color='blue',
                    alpha='0.3')
    ax.fill_between(data_ED_2_s,
                    data_ED_2_I,
                    data_ED_2_I + data_ED_2_Ierr[1],
                    color='blue',
                    alpha='0.3')
    ax.fill_between(data_ED_6_s,
                    data_ED_6_I - data_ED_6_Ierr[0],
                    data_ED_6_I,
                    color='green',
                    alpha='0.3')
    ax.fill_between(data_ED_6_s,
                    data_ED_6_I,
                    data_ED_6_I + data_ED_6_Ierr[1],
                    color='green',
                    alpha='0.3')

    ax.errorbar(data_3D_A_S,
                data_3D_I,
                xerr=data_3D_I_S_err,
                yerr=data_3D_I_err,
                color='red',
                fmt='o',
                ms=7,
                label='SU(6) 3D')
    ax.errorbar(S_f,
                I_f,
                linewidth=2,
                linestyle='dashed',
                color='red',
                alpha=1)
    #ax.fill_between(S_f,I_f-I_f_err,I_f+I_f_err,color='red', alpha='0.3')
    ax.fill_between(S_f, I_f - I_err_linear[0], I_f, color='red', alpha='0.3')
    ax.fill_between(S_f, I_f, I_f + I_err_linear[1], color='red', alpha='0.3')

    plt.xlim(1, 3.2)
    plt.ylim(0, 1)
    ax.legend(edgecolor='black',
              loc=9,
              bbox_to_anchor=(0.5, 1.16),
              ncol=3,
              fontsize=12)
    ax.tick_params(axis='both',
                   which='both',
                   direction='in',
                   bottom=True,
                   left=True,
                   top=True,
                   right=True,
                   labelsize=14)
    ax.set_yticks(np.arange(0.2, 1.2, 0.2))
    ax.set_xticks(np.arange(1, 3.2, 0.5))
    #ax.set_ylabel(r"$\vert$ Nearest-neighbor spin-correlation $\vert$",fontsize=16)
    #ax.set_ylabel("Normalized STO amplitude $A$",fontsize=16)
    ax.set_ylabel("Singlet-triplet imbalance $I \propto C_{\mathrm{NN}}$",
                  fontsize=16)
    ax.set_xlabel(r"$\displaystyle S/N_{\mathrm{ptcl}}k_B$", fontsize=16)
    #ax.set_title(r"$\displaystyle U/t=15.3$",fontsize=16)

    ax1 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.56, 0.65, 0.4, 0.3])
    ax1.set_axes_locator(ip)
    ax1.errorbar(data_ED_6_s,
                 data_ED_6_A,
                 color='green',
                 linestyle='dashed',
                 linewidth=2,
                 label='ED')
    ax1.errorbar(S_f,
                 A_f,
                 linewidth=2,
                 linestyle='solid',
                 color='red',
                 alpha=1,
                 label='DQMC')
    ax1.fill_between(S_f,
                     A_f - A_f_err,
                     A_f + A_f_err,
                     color='red',
                     alpha='0.3')
    ax1.set_ylim(0.0, 0.05)
    ax1.set_xlim(2.2, 3.2)
    ax1.set_xticks(np.arange(2.4, 3.4, 0.2))
    ax1.set_yticks(np.arange(0.0, 0.06, 0.01))
    ax1.tick_params(axis='both',
                    which='both',
                    direction='in',
                    bottom=True,
                    left=True,
                    top=True,
                    right=True,
                    labelsize=10)

    plt.tight_layout()
    #plt.savefig('IvsS_werrorbars_seq_4.png',dpi=500)
    return None


def Nscurve_fixedrho_vsT(Us, Nspecies, filling, obs, save=False):
    # Make canvas
    #w,h = 6,8.1
    w, h = 6, 100
    m_s_DQMC = 3
    m_s_NLCE = 3
    fig, axs = plt.subplots(len(Us),
                            1,
                            figsize=(w, h),
                            sharex='col',
                            sharey='row',
                            gridspec_kw={
                                'hspace': 0,
                                'wspace': 0
                            })

    # Define colors and markers and location
    locator = {4: 0, 8: 1, 12: 2, 40: 3}
    #locator={4:0,8:1,12:2,40:3}
    #locator={15.3:0,20:1,40:2,100:3}
    SU2_dict = {'doublons': 3, 'ken': 5, 'energy': 1}
    colors = cm.coolwarm(np.linspace(0, 1, 4))
    ldic = {
        2: ['o', colors[0], 2],
        3: ['s', colors[1], 3],
        4: ['v', colors[2], 4],
        6: ['<', colors[3], 6]
    }
    colors2 = cm.viridis_r(np.linspace(0, 1, 4))
    #colors2 = cm.coolwarm(np.linspace(0,1,4))
    ldic2 = {
        2: ['o', colors2[0], 2],
        3: ['s', colors2[1], 3],
        4: ['v', colors2[2], 4],
        6: ['<', colors2[3], 6]
    }

    #panel_dict = {"energy":[0.5,0.8],"doublons":[0.8,0.12],"ken":[0.8,0.12]}
    #panel_dict = {"energy":[0.8,0.12],"doublons":[0.8,0.12],"ken":[0.8,0.12],'entropy':[0.12,0.8]}
    #panel_dict = {"energy":[0.03,0.2],"doublons":[0.8,0.12],"ken":[0.8,0.12],'entropy':[0.12,0.8]}
    panel_dict = {
        "energy": [0.8, 0.12],
        "doublons": [0.8, 0.12],
        "ken": [0.8, 0.12],
        'entropy': [0.12, 0.8]
    }

    ## 2-site calculation (2-particle sector)
    def E2_TSTP(N, U, T):
        x = np.exp(np.sqrt(16 + U**2) / (2 * T))
        y = np.exp(U / (2 * T))
        e0 = np.sqrt(16 + U**2)
        numerator = (1 - N) * (-2 * U * x + (x**2) * y * (-U + e0) - y *
                               (U + e0))
        denominator = 4 * (N - 1) * (y + x +
                                     (x**2) * y) + 4 * (N + 1) * x * (y**2)
        return numerator / denominator

    def E1_AL(N, U, T):
        return U / (2 + np.sqrt(2 * N / (N - 1)) * np.exp(U / (2 * T)))

    def E2_2nd_TSTP(N, U, T):
        return 4 * E2_TSTP(N, U, T) - 3 * E1_AL(N, U, T)

    def E2_tilde_TSTP(N, U, T):

        def f(U, T):
            return (E2_2nd_TSTP(2, U, T) - E2_2nd_TSTP(3, U, T)) / (1 / 2 -
                                                                    1 / 3)

        return E2_2nd(N, U, T) - f(U, T) / N

    ## 2-site calculation (0-4 particle sectors)
    def E2(N, U, T):
        # Definitions
        J = 4 / U
        y2 = np.exp(-U / T)
        y1 = np.exp(-U / (2 * T))
        # Partition function
        Z1 = (N + 1) / (N - 1) + np.exp(J / T)
        Z2 = np.sqrt(2 / (N * (N - 1))) * (2 * np.cosh(1 / T) *
                                           (5 * N + 2) / 3 + 2 *
                                           (N - 2) * np.cosh(2 / T) / 3)
        Z3 = 3 + np.exp(-J / T)
        Z = Z1 + y1 * Z2 + y2 * Z3
        # Energy numerator
        e1 = -J * np.exp(J / T)
        e2 = np.sqrt(2 /
                     (N *
                      (N - 1))) * (-2 * np.sinh(1 / T) * (5 * N + 2) / 3 +
                                   (2 / 3) * (N - 2) *
                                   (U * np.cosh(2 / T) - 2 * np.sinh(2 / T)) +
                                   (4 / 3) * (N + 1) * U * np.cosh(1 / T))
        e3 = 3 * U + (U + J) * np.exp(-J / T)
        return (e1 + y1 * e2 + y2 * e3) / (2 * Z)

    def E2_2nd(N, U, T):
        return 4 * E2(N, U, T) - 3 * E1_AL(N, U, T)

    def E2_tilde(N, U, T):

        def f(U, T):
            return (E2_2nd(2, U, T) - E2_2nd(3, U, T)) / (1 / 2 - 1 / 3)

        return E2_2nd(N, U, T) - f(U, T) / N

    # Get U
    for U in Us:
        dataloc = locator[U]
        ax = axs[dataloc]
        # Put text
        textx, texty = panel_dict[obs]
        ax.text(textx,
                texty,
                r"$\displaystyle U/t=%s$" % U,
                fontsize=14,
                transform=ax.transAxes)

        # Format the x-axis
        if obs == 'entropy':
            ax.set_xlim(6e-2, 10)
        else:
            #ax.set_xlim(4e-2,10)
            ax.set_xlim(4e-2, 100)
        ax.set_xscale('log')
        #ax.set_yscale('log')

        # Make y axis label
        ylabel_dict = {"doublons":r"$\displaystyle \mathcal{D}$","ken":r"$\displaystyle K/t$",\
                       "energy":r"$\displaystyle E/t$","entropy":r"$\displaystyle S/N_s$"}
        #ylabel_dict = {"doublons":r"$\displaystyle \tilde{\mathcal{D}}$","ken":r"$\displaystyle \tilde{K}/t$",\
        #               "energy":r"$\displaystyle \tilde{E}/t$","entropy":r"$\displaystyle S/N_s$"}
        ax.set_ylabel(ylabel_dict[obs], fontsize=16)

        # Define ticks
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
        #ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        # Draw all the ticks
        ax.tick_params(direction='in',
                       which='major',
                       left=True,
                       length=6,
                       width=1,
                       labelsize=14)
        ax.tick_params(direction='in',
                       which='minor',
                       left=True,
                       length=3,
                       width=1,
                       labelsize=14)

        interp_dict = {}
        interp_err = {}
        HTSE_interp_dict = {}
        interp_dict_2order = {}
        for Ns in Nspecies:
            # Plot the analytic expression
            TTSTP = np.logspace(-1, 2.7, 100)
            #ax.plot(TTSTP,4*E2_TSTP(Ns,U,TTSTP),color=ldic[Ns][1],ls='dashed')
            #ax.plot(TTSTP,E2_2nd_TSTP(Ns,U,TTSTP),color=ldic[Ns][1],ls='dotted')
            #ax.plot(TTSTP,E2_2nd(Ns,U,TTSTP),color=ldic[Ns][1],ls='dashed')
            #ax.plot(TTSTP,E2_tilde(Ns,U,TTSTP),color=ldic[Ns][1],ls='dashed')
            #ax.plot(TTSTP,E2_tilde_TSTP(Ns,U,TTSTP),color=ldic[Ns][1],ls='dotted')
            # Get the HTSE dataset
            ts = [0, 1]
            HTSE_keys = {
                'doublons': "D",
                'energy': "E",
                'entropy': "S",
                'ken': "K"
            }
            if U not in [100]:
                HTSE = {}
                for t in ts:
                    HTSE[t] = {}
                    if t in [0, 0.875, 1]:
                        HTSE_path = '/Users/eibarragp/Documents/Rice University/Hazzard Group/SU(N) Quantum Magnetism/SUN_EoS/'
                    else:
                        HTSE_path = '/Users/eibarragp/Documents/Rice University/Hazzard Group/SU(N) Quantum Magnetism/SUN_EoS/HTSE_data/'
                    raw_data = csv_reader(HTSE_path +
                                          'HTSE_U%s_N%s_t%s_lowT.csv' %
                                          (U, Ns, t))
                    #raw_data = csv_reader(HTSE_path + 'HTSE_U%s_N%s_t%s.csv' % (U,Ns,t))
                    HTSE[t]['T'] = raw_data[:, 0]
                    HTSE_idx = idx = np.searchsorted(HTSE[t]['T'], 4 / U)
                    HTSE[t]['T'] = HTSE[t]['T'][HTSE_idx:]
                    HTSE[t]['mu'] = raw_data[HTSE_idx:, 1]
                    HTSE[t]['E'] = raw_data[HTSE_idx:, 2]
                    HTSE[t]['D'] = raw_data[HTSE_idx:, 3]
                    HTSE[t]['S'] = raw_data[HTSE_idx:, 4]
                    HTSE[t]['K'] = np.array(
                        HTSE[t]['E']) - U * np.array(HTSE[t]['D'])
            # Get the DQMC dataset
            try:
                # Get the E,C,S fits
                data_fit = csv_reader('U%s' % U + '_N%s' % Ns +
                                      '_rho%s' % filling + '_fit_ECS.csv')
                data_fit2 = csv_reader('U%s' % U + '_N%s' % Ns +
                                       '_rho%s' % filling + '_fit_DKS.csv')
                fits_dict = {"T":data_fit[:,0],"energy":data_fit[:,1],"C":data_fit[:,2],"entropy":data_fit[:,3],\
                             "doublons":data_fit2[:,1],"ken":data_fit2[:,2],"dDdT": data_fit2[:,3],\
                             "dKdT":data_fit2[:,4],"S2":data_fit2[:,5]}
                if obs == 'entropy':
                    entrop_dat = csv_reader(
                        'U%s' % U + '_N%s' % Ns + '_entropy%s' % filling +
                        '_newest.csv')  #Includes some extra points at high-T
                    datT = entrop_dat[:, 0]
                    datNs = entrop_dat[:, 1]
                    errNs = entrop_dat[:, 2]
                    #idx = np.searchsorted(datT,cutoff[Ns])
                    idx = np.searchsorted(datT, cutoff[Ns][U])
                    datT, datNs, errNs = datT[idx:], datNs[idx:], errNs[idx:]
                    ax.errorbar(datT,
                                datNs,
                                yerr=errNs,
                                capsize=3,
                                color=ldic[Ns][1],
                                fmt=ldic[Ns][0],
                                ms=m_s_DQMC,
                                linestyle='none',
                                label=ldic[Ns][2])
                    ax.axhline(Sinf_filling(Ns, filling),
                               color=ldic[Ns][1],
                               linestyle='dotted')
                    #ax.arrow(9,Sinf_filling(Ns,filling),-0.5,0)
                    #ax.annotate('', xy=(11,Sinf_filling(Ns,filling)), xycoords='data', xytext=(9,Sinf_filling(Ns,filling)), arrowprops=dict(arrowstyle="->", color=ldic[Ns][1]))
                    # Plot HTSE
                    #ax.errorbar(HTSE[ts[0]]['T'],HTSE[ts[0]]['S'],color=ldic[Ns][1],linestyle='dashed')
                    #ax.errorbar(HTSE[ts[1]]['T'],HTSE[ts[1]]['S'],color=ldic[Ns][1],linestyle='dashdot')

                    #Plot fit
                    fit_idx = np.searchsorted(fits_dict["T"], datT[0])
                    #ax.errorbar(fits_dict["T"],fits_dict["entropy"],color=ldic[Ns][1],ls='solid')
                    ax.errorbar(fits_dict["T"][fit_idx:],
                                fits_dict[obs][fit_idx:],
                                color=ldic[Ns][1],
                                ls='solid')
                    ax.errorbar(fits_dict["T"][:fit_idx + 1],
                                fits_dict[obs][:fit_idx + 1],
                                color=ldic[Ns][1],
                                ls='dashdot',
                                alpha=1,
                                lw=1)

                    #ax.errorbar(fits_dict["T"],fits_dict["S2"],color=ldic[Ns][1],ls='dashed')
                    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
                    ax.yaxis.set_major_locator(MultipleLocator(1))
                else:
                    if Ns == 2:
                        if U == 8:
                            N2_data = U8_decoder(obs)
                            datT = N2_data[0]
                            datNs = N2_data[1]
                            errNs = N2_data[2]
                        else:
                            N2_data = csv_reader("E_D_KE_SU2_U%s" % U +
                                                 "_2D.csv")
                            SU2_idx = SU2_dict[obs]
                            datT = N2_data[:, 0]
                            datNs = N2_data[:, SU2_idx]
                            errNs = N2_data[:, SU2_idx + 1]
                        ##datT2,datNs2,errNs2 = datT,datNs,errNs
                    else:
                        # Sem weighted
                        raw_data = csv_reader('U%s' % U + '_N%s' % Ns +
                                              '_rho%s' % filling +
                                              '_newest.csv')
                        data_dict = {
                            'T': 0,
                            'mu': 1,
                            'rho': 2,
                            'doublons': 3,
                            'energy': 5,
                            'ken': 7
                        }
                        datT, datNs, errNs = raw_data[:, data_dict[
                            'T']], raw_data[:, data_dict[
                                obs]], raw_data[:, data_dict[obs] + 1]
                        ## Sem regular
                        ##raw_data2 = csv_reader('U%s'%U + '_N%s'%Ns + '_rho%s'%filling +'_reg.csv')
                        ##datT2,datNs2,errNs2 = raw_data2[:,data_dict['T']],raw_data2[:,data_dict[obs]],raw_data2[:,data_dict[obs]+1]
                        ## With new data points
                        #raw_data2 = csv_reader('U%s'%U + '_N%s'%Ns + '_rho%s'%filling +'_newest.csv')
                        #datT2,datNs2,errNs2 = raw_data2[:,data_dict['T']],raw_data2[:,data_dict[obs]],raw_data2[:,data_dict[obs]+1]
                    # Plot data
                    ax.errorbar(datT,
                                datNs,
                                yerr=errNs,
                                color=ldic[Ns][1],
                                fmt=ldic[Ns][0],
                                ms=m_s_DQMC,
                                linestyle='none',
                                label=ldic[Ns][2])
                    ##ax.errorbar(datT2,datNs2,yerr=errNs2,color=ldic2[Ns][1],fmt=ldic[Ns][0],ms=m_s_DQMC,linestyle='none',label=ldic[Ns][2])

                    ##print(Ns,U)
                    ##print(errNs2)
                    ##print(errNs)
                    #ax.errorbar(HTSE[ts[0]]['T'],HTSE[ts[0]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashed')
                    #ax.errorbar(HTSE[ts[1]]['T'],HTSE[ts[1]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashdot')

                    HTSE_interp_dict[Ns] = interp1d(
                        HTSE[ts[1]]['T'],
                        HTSE[ts[1]][HTSE_keys[obs]],
                        kind='linear',
                        bounds_error=False)

                    #Plot fit
                    fit_idx = np.searchsorted(fits_dict["T"], datT[0])
                    #ax.errorbar(fits_dict["T"],fits_dict[obs],color=ldic[Ns][1],ls='solid')
                    ax.errorbar(fits_dict["T"][fit_idx:],
                                fits_dict[obs][fit_idx:],
                                color=ldic[Ns][1],
                                ls='solid')
                    ax.errorbar(fits_dict["T"][:fit_idx + 1],
                                fits_dict[obs][:fit_idx + 1],
                                color=ldic[Ns][1],
                                ls='dashdot',
                                alpha=1,
                                lw=1)

                    #if obs == 'doublons' and U==4:
                    #ax.axhline(0.5*(1-1/Ns),color=ldic[Ns][1],ls='dotted')
                    #ax.arrow(40,0.5*(1-1/Ns),0.5,0)
                    #    ax.annotate('', xy=(47,0.5*(1-1/Ns)), xycoords='data', xytext=(70,0.5*(1-1/Ns)), arrowprops=dict(arrowstyle="->", color=ldic[Ns][1]))
                    if obs == 'energy' or obs == 'ken' or obs == "doublons":
                        interp_dict[Ns] = interp1d(datT,
                                                   datNs,
                                                   kind='linear',
                                                   bounds_error=False)
                        interp_err[Ns] = interp1d(datT,
                                                  errNs,
                                                  kind='linear',
                                                  bounds_error=False)
                        #vline = {4:4.448,8:3.341,12:2.794}
                        #ax.axvline(vline[U],color='black')
            except:
                print("There's no DQMC data for:U=%s" % U + ",Ns=%s" % Ns)
                pass
            # Get the NLCE dataset (cutoff data)
            NLCE_folder = os.path.join(os.path.join(source_path, "NLCE"),
                                       "new_cutoff_2")
            NLCE_folder = os.path.join(NLCE_folder, obs)
            files = os.listdir(NLCE_folder)
            if filling == 1:
                for file in files:
                    if "U%s" % U in file and "N%s" % Ns in file and obs in file:
                        if U == 4 and "40" in file:
                            pass
                        else:
                            dat_NLCE = csv_reader(
                                os.path.join(NLCE_folder, file))
                            datT, datNs = dat_NLCE[:, 0], dat_NLCE[:, 1]
                            if Ns == 6:
                                datT, datNs = datT[::3], datNs[::3]
                            ax.errorbar(datT,
                                        datNs,
                                        color=ldic[Ns][1],
                                        fmt=ldic[Ns][0],
                                        ms=m_s_NLCE,
                                        mfc='white',
                                        linestyle='none'
                                        )  #,label=ldic[Ns][2]) ,zorder=6
                            if obs == 'entropy':
                                ax.axhline(Sinf_filling(Ns, filling),
                                           color=ldic[Ns][1],
                                           linestyle='dotted')
                                ax.yaxis.set_minor_locator(
                                    MultipleLocator(0.25))
                                ax.yaxis.set_major_locator(MultipleLocator(1))
                            if obs == 'energy' or obs == 'ken' or obs == "doublons":
                                #ax.errorbar(HTSE[ts[0]]['T'],HTSE[ts[0]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashed')
                                #ax.errorbar(HTSE[ts[1]]['T'],HTSE[ts[1]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashdot')
                                HTSE_interp_dict[Ns] = interp1d(
                                    HTSE[ts[1]]['T'],
                                    HTSE[ts[1]][HTSE_keys[obs]],
                                    kind='linear',
                                    bounds_error=False)
                                if U in [
                                        4, 8, 12
                                ] or (Ns == 6 and U in [15.3, 20, 40, 100]):
                                    pass
                                else:
                                    interp_dict[Ns] = interp1d(
                                        datT,
                                        datNs,
                                        kind='linear',
                                        bounds_error=False)
                                    print(
                                        U, ts, Ns,
                                        'HTSE_U%s_N%s_t%s_lowT.csv' %
                                        (U, Ns, t))

            if obs != 'ken' or obs == 'ken':
                # Get the NLCE dataset (2nd order)
                NLCE_folder = os.path.join(os.path.join(source_path, "NLCE"),
                                           "NLCE_order2")
                NLCE_folder = os.path.join(NLCE_folder, obs)
                files = os.listdir(NLCE_folder)
                if filling == 1:
                    for file in files:
                        if "U%s" % U in file and "N%s" % Ns in file and obs in file:
                            if U == 4 and "40" in file:
                                pass
                            else:
                                dat_NLCE = csv_reader(
                                    os.path.join(NLCE_folder, file))
                                datNs = dat_NLCE[0]
                                datT = np.logspace(-1, 2, 100)
                                if Ns == 6:
                                    datT = np.logspace(-1, 0, 100)
                                    #datT,datNs = datT[::3],datNs[::3]
                                #ax.errorbar(datT,datNs,color=ldic[Ns][1],fmt=ldic[Ns][0],ms=m_s_NLCE*0,mfc='white',linestyle='dashdot',label=ldic[Ns][2])
                                if obs == 'entropy':
                                    ax.axhline(Sinf_filling(Ns, filling),
                                               color=ldic[Ns][1],
                                               linestyle='dotted')
                                if obs == 'energy' or obs == 'ken' or obs == "doublons":
                                    #ax.errorbar(HTSE[ts[0]]['T'],HTSE[ts[0]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashed')
                                    #ax.errorbar(HTSE[ts[1]]['T'],HTSE[ts[1]][HTSE_keys[obs]],color=ldic[Ns][1],linestyle='dashdot')
                                    HTSE_interp_dict[Ns] = interp1d(
                                        HTSE[ts[1]]['T'],
                                        HTSE[ts[1]][HTSE_keys[obs]],
                                        kind='linear',
                                        bounds_error=False)
                                    if U == 30 or (Ns == 6 and U
                                                   in [15.3, 20, 40, 100]):
                                        pass
                                    else:
                                        interp_dict_2order[Ns] = interp1d(
                                            datT,
                                            datNs,
                                            kind='linear',
                                            bounds_error=False)
                                        #print(U,ts,Ns,'HTSE_U%s_N%s_t%s_lowT.csv' % (U,Ns,t))
                else:
                    pass
            """
            # Used to generate the ken data only
            if obs == "ken":
                # Get the NLCE dataset (2nd order)
                NLCE_folder = os.path.join(os.path.join(source_path,"NLCE"),"NLCE_order2")
                NLCE_folder_D = os.path.join(NLCE_folder,"doublons")
                NLCE_folder_E = os.path.join(NLCE_folder,"energy")
                files_D = os.listdir(NLCE_folder_D)
                if filling ==1:
                    for file in files_D:
                        file_D = file
                        file_E = file.replace("doublons", "energy")
                        file_K = "my_data_Lalo_"+file_E.replace("energy","ken") 
                        if "U%s"%U in file and "N%s"%Ns in file and "doublons" in file:
                            if U==4 and "40" in file:
                                pass
                            else:
                                dat_NLCE_D = csv_reader(os.path.join(NLCE_folder_D,file_D))
                                dat_NLCE_E = csv_reader(os.path.join(NLCE_folder_E,file_E))
                                datNs_D = dat_NLCE_D[0]
                                datNs_E = dat_NLCE_E[0]
                                datT = np.logspace(-1,2,100)
                                if Ns==6:
                                    datT = np.logspace(-1,0,100)
                                datNs_K = np.array(datNs_E) - U*np.array(datNs_D)
                                #print(datNs_K.reshape(1,len(datNs_K)).shape)
                                #pd.DataFrame(datNs_K.reshape(1,len(datNs_K))).to_csv(file_K,header = None,index=None)
            """
            # for entropy
            if obs == 'entropy':
                ax.axvline(4 / U, color='black', linestyle='dotted')

            # Plot AL for doublons
            if obs == 'doublons':
                ax.set_yscale("log")
                #ax.axvline(4/U,color='black',linestyle='dotted')
                #ax.axhline(0.5*(1-1/Ns),color=ldic[Ns][1],ls='dotted')
                if U == 4:
                    ax.set_ylim(1e-1, 0.5)
                    ax.yaxis.set_major_locator(
                        FixedLocator([0.2, 0.3, 0.4, 0.5]))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
                elif U == 8:
                    ax.set_ylim(3e-2, 0.5)
                    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1]))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
                elif U == 12:
                    ax.set_ylim(1e-2, 0.5)
                    ax.yaxis.set_major_locator(FixedLocator([0.1, 1]))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
                elif U == 40:
                    ax.set_ylim(6e-4, 0.5)
                    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.01,
                                                             0.1]))
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
                    y_minor = LogLocator(base=10.0,
                                         subs=np.arange(1.0, 10.0) * 0.1,
                                         numticks=5)
                    ax.yaxis.set_minor_locator(y_minor)
                    ax.yaxis.set_minor_formatter(NullFormatter())

                try:
                    data_AL = csv_reader("ALdoubSUN%s" % Ns + "_U%s" % U +
                                         ".csv")
                    if U == 40:
                        #pass
                        ax.plot(data_AL[60:, 0],
                                data_AL[60:, 1],
                                color=ldic[Ns][1],
                                ls='dashed')
                    else:
                        #pass
                        ax.plot(data_AL[34:, 0],
                                data_AL[34:, 1],
                                color=ldic[Ns][1],
                                ls='dashed')
                except:
                    print("No AL data for such cases")
                    pass

            # Plot NIN for KE and format axes
            if obs == 'ken':
                if U == 4:
                    ax.set_ylim(-4, 0)
                    ax.yaxis.set_major_locator(
                        FixedLocator(np.arange(-4, 1, step=1)))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
                elif U == 8:
                    ax.set_ylim(-3.5, 0)
                    ax.yaxis.set_major_locator(
                        FixedLocator(np.arange(-3, 0, step=1)))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
                elif U == 12:
                    ax.set_ylim(-2, 0)
                    ax.yaxis.set_major_locator(
                        FixedLocator(np.arange(-2, 0, step=0.5)))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
                elif U == 40:
                    ax.set_ylim(-0.3, 0)
                    ax.yaxis.set_major_locator(
                        FixedLocator(np.arange(-0.3, 0, step=0.1)))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                try:
                    data_NIN = csv_reader("NIN_SU%s" % Ns + ".csv")
                    ax.plot(data_NIN[:, 0],
                            data_NIN[:, 4],
                            color=ldic[Ns][1],
                            ls='dotted')
                    #interp_dict[Ns] = interp1d(data_NIN[:,0],data_NIN[:,4],kind='linear',bounds_error=False) # Just to test the scaling
                except:
                    print("No NIN data for such cases")
                    pass
                #ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                #ax.yaxis.set_major_locator(MultipleLocator(0.1))
                #ax.yaxis.set_minor_locator(MultipleLocator(0.05))
                #ax.set_ylim(0,0.45)
                #ax.xaxis.set_major_locator(FixedLocator([0.1,1,10,100]))
                #ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.16g}'.format(y)))
                #ax.xaxis.set_minor_locator(MultipleLocator(0.5))

        if obs == 'ken':
            print(U)
            ken_pow = 1
            # Isosbetic point analysis (PRB 87 195140, 2013)
            #Ts = np.logspace(-1,2.9,100)
            Ts = np.logspace(-2, 2.9, 100)
            keys = sorted(interp_dict.keys())
            for i in range(0, len(keys)):
                Ns = keys[i]
                try:
                    #print(filling)
                    y1 = np.array(wiggly_function(obs, ken_pow, interp_dict,
                                                  Ts, Ns, 2, 3),
                                  dtype=float)
                    y2 = wiggly_function(obs, ken_pow, HTSE_interp_dict, Ts,
                                         Ns, 2, 3)
                    #y2 = wiggly_function(obs,ken_pow-0.5,interp_dict,Ts,Ns,2,3)
                except:
                    #print(filling)
                    y1 = np.array(wiggly_function(obs, ken_pow, interp_dict,
                                                  Ts, Ns, 3, 4),
                                  dtype=float)
                    y2 = wiggly_function(obs, ken_pow, HTSE_interp_dict, Ts,
                                         Ns, 3, 4)
                    #y2 = wiggly_function(obs,ken_pow-0.5,interp_dict,Ts,Ns,3,4)
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid')
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid',label=ldic[Ns][2])
                #ax.axvline(4/U,color='black',linestyle='dotted')
                if U not in [40]:
                    y1_err = np.array(quad_wiggly(interp_err[Ns](Ts),
                                                  interp_err[2](Ts),
                                                  interp_err[3](Ts), Ns, 2, 3),
                                      dtype=float)
                    #ax.fill_between(Ts,y1-y1_err,y1+y1_err,color=ldic2[Ns][1],alpha=0.5)
                if Ns == 6 and U in [4, 8, 40]:
                    pass
                else:
                    y3 = wiggly_function(obs, ken_pow, interp_dict_2order, Ts,
                                         Ns, 2, 3)
                #y2 = wiggly_function(obs,1,interp_dict,Ts,Ns,4,6)
                #y3 = wiggly_function(obs,1,interp_dict,Ts,Ns,3,6)
                #y3 = y3_E - U*y3_D
                #ax.errorbar(Ts,y2,color=ldic2[Ns][1],ls='dashed')
                #ax.errorbar(Ts,y3,color=ldic2[Ns][1],ls='dotted')

        if obs == 'doublons':
            print(U)
            doub_pow = 1
            # Isosbetic point analysis (PRB 87 195140, 2013)
            Ts = np.logspace(-1, 2.9, 100)
            keys = sorted(interp_dict.keys())
            for i in range(0, len(keys)):
                Ns = keys[i]
                try:
                    #print(filling)
                    y1 = np.array(wiggly_function(obs, doub_pow, interp_dict,
                                                  Ts, Ns, 2, 3),
                                  dtype=float)
                    y2 = wiggly_function(obs, doub_pow, HTSE_interp_dict, Ts,
                                         Ns, 2, 3)
                    #y2 = wiggly_function(obs,doub_pow+1,interp_dict,Ts,Ns,2,3)
                except:
                    #print(filling)
                    y1 = np.array(wiggly_function(obs, doub_pow, interp_dict,
                                                  Ts, Ns, 3, 4),
                                  dtype=float)
                    y2 = wiggly_function(obs, doub_pow, HTSE_interp_dict, Ts,
                                         Ns, 3, 4)
                    #y2 = wiggly_function(obs,doub_pow+1,interp_dict,Ts,Ns,2,3)
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid')
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid',label=ldic[Ns][2])
                #ax.axvline(4/U,color='black',linestyle='dotted')
                if U not in [40]:
                    y1_err = np.array(quad_wiggly(interp_err[Ns](Ts),
                                                  interp_err[2](Ts),
                                                  interp_err[3](Ts), Ns, 2, 3),
                                      dtype=float)
                    #ax.fill_between(Ts,y1-y1_err,y1+y1_err,color=ldic2[Ns][1],alpha=0.5)
                if Ns == 6 and U in [4, 8, 40]:
                    pass
                else:
                    y3 = wiggly_function(obs, doub_pow, interp_dict_2order, Ts,
                                         Ns, 2, 3)
                ##y2 = wiggly_function(obs,1,interp_dict,Ts,Ns,4,6)
                ##y3 = wiggly_function(obs,1,interp_dict,Ts,Ns,3,6)
                #ax.errorbar(Ts,y2,color=ldic2[Ns][1],ls='dashed')
                #ax.errorbar(Ts,y3,color=ldic2[Ns][1],ls='dotted')

        # Find the intercept for the E vs T (Ns curves) and format axes
        if obs == 'energy':
            """
            # Formatting (extended zoom version) # for the Etilde
            ax.set_xlim(4e-2,10)
            if  U==4:
                ax.set_ylim(-2.5,0.75)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-2,0.9,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==8:
                ax.set_ylim(-1.5,1)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-1.5,1,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==12:
                ax.set_ylim(-0.85,0.75)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-0.75,0.75,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==40 :
                ax.set_ylim(-0.15,0.15)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-0.25,0.15,step=0.1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            elif  U==20 :
                ax.set_ylim(-0.35,0.45)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-0.35,0.45,step=0.25)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            
            # Formatting (extended version)
            if  U==4:
                ax.set_ylim(-2.5,2.5)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-2,3,step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif  U==8:
                ax.set_ylim(-1.5,4.5)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-1.5,4.5,step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif  U==12:
                ax.set_ylim(-1,6)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-1,6,step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif  U==40 :
                ax.set_ylim(-1,17)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(0,16,step=3)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(1.5))
            """
            # Formatting (regular version) # for the energy
            if U == 4:
                ax.set_ylim(-2, 1.75)
                ax.yaxis.set_major_locator(
                    FixedLocator(np.arange(-2, 2, step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif U == 8:
                ax.set_ylim(-1.5, 3.5)
                ax.yaxis.set_major_locator(
                    FixedLocator(np.arange(-1.5, 3.5, step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif U == 12:
                ax.set_ylim(-1, 5)
                ax.yaxis.set_major_locator(
                    FixedLocator(np.arange(-1, 5, step=1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.5))
            elif U == 40:
                ax.set_ylim(-1, 14)
                ax.yaxis.set_major_locator(
                    FixedLocator(np.arange(0, 13, step=3)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(1.5))
            """ 
            # Formatting (zoom version)
            if  U==4:
                ax.set_ylim(-2,0.75)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-2,0.9,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==8:
                ax.set_ylim(-1.5,1)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-1.5,1,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==12:
                ax.set_ylim(-0.75,0.75)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-0.75,0.75,step=0.5)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.25))
            elif  U==40 :
                ax.set_ylim(-0.15,0.15)
                ax.yaxis.set_major_locator(FixedLocator(np.arange(-0.25,0.15,step=0.1)))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
                ax.yaxis.set_minor_locator(MultipleLocator(0.05))
            """
            # Plot the intercept
            data_intercept = cross_finder(interp_dict, 4)
            print(U)
            print(data_intercept)
            print(" ")
            ax.axvline(data_intercept[0], color='black')
            ax.axvspan(data_intercept[0] - data_intercept[1],
                       data_intercept[0] + data_intercept[1],
                       alpha=0.3,
                       color='black')
            #ax.axhline(data_intercept[2],color='black')
            #ax.axhspan(data_intercept[2]- data_intercept[3],data_intercept[2]+ data_intercept[3], alpha=0.3, color='black')

            # Isosbetic point analysis (PRB 87 195140, 2013)
            E_pow = 1
            Ts = np.logspace(-1, 2.9, 100)
            keys = sorted(interp_dict.keys())
            print(sorted(interp_dict_2order.keys()))
            for i in range(0, len(keys)):
                Ns = keys[i]
                try:
                    #print(filling)
                    y1 = np.array(wiggly_function(obs, E_pow, interp_dict, Ts,
                                                  Ns, 2, 3),
                                  dtype=float)
                    y2 = wiggly_function(obs, E_pow, HTSE_interp_dict, Ts, Ns,
                                         2, 3)
                    if Ns == 6 and U in [4, 8, 40]:
                        pass
                    else:
                        y3 = wiggly_function(obs, E_pow, interp_dict_2order,
                                             Ts, Ns, 2, 3)
                except:
                    #print(filling)
                    y1 = wiggly_function(obs, E_pow, interp_dict, Ts, Ns, 3, 4)
                    y2 = wiggly_function(obs, E_pow, HTSE_interp_dict, Ts, Ns,
                                         3, 4)
                    #if Ns== 6 and U in [4,8]:
                    #    pass
                    #else:
                    #    y3 = wiggly_function(obs,E_pow,interp_dict_2order,Ts,Ns,3,4)
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid',label=ldic[Ns][2])

                if U not in [40]:
                    y1_err = np.array(quad_wiggly(interp_err[Ns](Ts),
                                                  interp_err[2](Ts),
                                                  interp_err[3](Ts), Ns, 2, 3),
                                      dtype=float)
                    #ax.fill_between(Ts,y1-y1_err,y1+y1_err,color=ldic2[Ns][1],alpha=0.5)
                #ax.errorbar(Ts,y1,color=ldic2[Ns][1],ls='solid',label=ldic[Ns][2])
                #ax.axvline(4/U,color='black',linestyle='dotted')
                ##y2 = wiggly_function(obs,1,HTSE_interp_dict,Ts,Ns,4,6)
                ##y3 = wiggly_function(obs,1,interp_dict,Ts,Ns,3,6)
                #ax.errorbar(Ts,y2,color=ldic2[Ns][1],ls='dashed')
                #ax.errorbar(Ts,y3,color=ldic2[Ns][1],ls='dotted')

        #if obs =='entropy':
        #    ax.axvline(4/U,color='black',linestyle='dotted')

    # Put legend on the last panel
    #legend_elements = [Line2D([0],[0],color=lab_dict[key][1], marker=lab_dict[key][0],label=lab_dict[key][2]) for key in sorted(lab_dict.keys())]
    #axs[2].legend(handles=legend_elements,loc='best',ncol=2,edgecolor='black',fontsize=12,handlelength=0)
    handles, labels = axs[0].get_legend_handles_labels()
    handles = [h[0] for h in handles]
    #axs[2].legend(handles,labels,loc='best',ncol=2,edgecolor='black',fontsize=12,handlelength=1.5)

    custom_lines = [(Line2D([],[], color=ldic[2][1],ls='None',marker=ldic[2][0]),Line2D([],[], color=ldic[2][1],ls='None',marker=ldic[2][0],mfc='white')),\
                    (Line2D([],[], color=ldic[3][1],ls='None',marker=ldic[3][0]),Line2D([],[], color=ldic[3][1],ls='None',marker=ldic[3][0],mfc='white')),\
                    (Line2D([],[], color=ldic[4][1],ls='None',marker=ldic[4][0]),Line2D([],[], color=ldic[4][1],ls='None',marker=ldic[4][0],mfc='white')),\
                    (Line2D([],[], color=ldic[6][1],ls='None',marker=ldic[6][0]),Line2D([],[], color=ldic[6][1],ls='None',marker=ldic[6][0],mfc='white'))]
    custom_labels = [2, 3, 4, 6]

    if obs == 'ken':
        lgd = axs[0].legend(handles=custom_lines,
                            labels=custom_labels,
                            loc='center right',
                            bbox_to_anchor=(1, 0.58),
                            ncol=2,
                            edgecolor='black',
                            fontsize=13,
                            handlelength=1,
                            markerscale=0.8,
                            title=r"$\displaystyle N$",
                            handler_map={tuple: HandlerTuple(ndivide=None)})
        #lgd = axs[0].legend(handles,labels,loc=1,ncol=2,edgecolor='black',fontsize=11,handlelength=1.5,title=r"$\displaystyle N$",bbox_to_anchor=(0.99,0.9))
        #lgd = axs[0].legend(handles,labels,loc="center right",bbox_to_anchor=[1,0.6],ncol=2,edgecolor='black',fontsize=13,handlelength=0.5,title=r"$\displaystyle N$",markerscale=1.5)
        #lgd = axs[3].legend(handles,labels,loc='upper left',ncol=2,edgecolor='black',fontsize=13,handlelength=1.5,title=r"$\displaystyle N$",markerscale=1.5,labelspacing=0.3)
    elif obs == 'entropy':
        #pass
        lgd = axs[3].legend(handles=custom_lines,
                            labels=custom_labels,
                            loc='upper right',
                            ncol=2,
                            edgecolor='black',
                            fontsize=13,
                            handlelength=1,
                            markerscale=0.8,
                            title=r"$\displaystyle N$",
                            handler_map={tuple: HandlerTuple(ndivide=None)
                                         })  #,framealpha=1
    else:
        lgd = axs[3].legend(handles=custom_lines,
                            labels=custom_labels,
                            loc='upper left',
                            ncol=2,
                            edgecolor='black',
                            fontsize=13,
                            handlelength=1,
                            markerscale=0.8,
                            title=r"$\displaystyle N$",
                            handler_map={tuple: HandlerTuple(ndivide=None)})
        #lgd = axs[(len(Us)-1)].legend(handles,labels,loc='upper left',bbox_to_anchor=[0.15, 1],ncol=2,edgecolor='black',fontsize=13,handlelength=1.5,title=r"$\displaystyle N$",markerscale=1.5) # For D  tilde
        #lgd = axs[(len(Us)-1)].legend(handles,labels,loc='upper right',ncol=2,edgecolor='black',fontsize=13,handlelength=0.5,title=r"$\displaystyle N$",markerscale=1.5)
        #lgd = axs[(len(Us)-1)].legend(handles,labels,loc=9,ncol=2,edgecolor='black',fontsize=11,handlelength=1.5,title=r"$\displaystyle N$",bbox_to_anchor=(0.3,0.9))
        #lgd = axs[1].legend(handles,labels,loc='upper left',ncol=2,edgecolor='black',fontsize=13,handlelength=1.5,title=r"$\displaystyle N$",markerscale=1.5) # For E tilde
    #lgd = axs[0].legend(handles,labels,loc=2,ncol=2,edgecolor='black',fontsize=11,handlelength=1.5,title=r"$\displaystyle N$")
    #lgd = axs[0].legend(handles,labels,loc='best',ncol=2,edgecolor='black',fontsize=11,handlelength=1.5,title=r"$\displaystyle N$")
    lgd.get_title().set_fontsize(13)
    axs[len(Us) - 1].set_xlabel(r"$\displaystyle T/t$", fontsize=16)
    #fig.suptitle(r"$\displaystyle \langle n \rangle = %s$"%filling,fontsize =16,y=0.93)

    # For the tilde data
    """
    custom_lines = [Line2D([0],[0],color='black',lw=1.5,ls='dashed'),\
                    Line2D([0], [0], color='black',lw=1.5,ls='dotted'),\
                    Line2D([0], [0], color='black', lw=1.5,ls='solid')]
    if obs =='energy':
        axs[0].legend(custom_lines, ['2nd order HTSE', '2nd order NLCE','DQMC/NLCE'],fontsize=12,loc='upper left') # For E tilde
    elif obs =='doublons':
        axs[0].legend(custom_lines, ['2nd order HTSE', '2nd order NLCE','DQMC/NLCE'],fontsize=12,loc='lower left',labelspacing=0) # For D tilde
    else:
        axs[0].legend(custom_lines, ['2nd order HTSE', '2nd order NLCE','DQMC/NLCE'],fontsize=12,loc='center right',labelspacing=0.2) # For K tilde
    """
    #axs[0].legend(custom_lines, ['2nd order HTSE', '2nd order NLCE','Numerical data'],fontsize=12)
    ##legend1 = axs[0].legend(plot_lines[0], ["algo1", "algo2", "algo3"], loc=1)
    ##axs[0].legend([l[0] for l in plot_lines], parameters, loc=4)
    ###axs[0].gca().add_artist(legend1)

    # Align labels
    fig.align_labels()
    # Just label the outside
    for ax in axs.flat:
        ax.label_outer()
    # Save figure if needed
    if save:
        #fig.savefig(obs + '_NvsT_final_singlefit_cut_newlegend_newconverg.pdf',transparent=True,dpi=300, bbox_inches='tight')
        #fig.savefig(obs + '_NvsT_final_newlegend_newconverg.pdf',transparent=True,dpi=300, bbox_inches='tight')
        #fig.savefig(obs + '_NvsT_final_crop.pdf',transparent=True,dpi=300, bbox_inches='tight')
        #fig.savefig(obs + '_NvsT_final_collapse_werrorbar.pdf',transparent=True,dpi=300, bbox_inches='tight')
        fig.savefig(
            obs +
            '_NvsT_final_collapse_zoom_werrorbar_newlegend_newconverg_refA.pdf',
            transparent=True,
            dpi=300,
            bbox_inches='tight')
    return None