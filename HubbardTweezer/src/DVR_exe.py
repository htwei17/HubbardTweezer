import numpy as np
from DVR.dynamics import (DVRdynamics, DVRdynamics_exe, get_stop_time)
import argparse

dsp = 'Tweezer stroboscopic dynamics by 3D DVR.'
parser = argparse.ArgumentParser(description=dsp)
parser.add_argument("-n",
                    "--number",
                    type=int,
                    help="Number of grid points.",
                    default=0)
parser.add_argument("-f",
                    "--freq",
                    nargs="+",
                    type=float,
                    help="Frequencies: [initial final step] in unit of kHz.",
                    default=[100, 240, 10])
parser.add_argument(
    "-R",
    "--range",
    nargs="+",
    type=float,
    help=
    "Spatial range in dynamics w/o absorption region: [Rx Ry Rz] in unit of w.",
    default=[3, 3, 7.2])
parser.add_argument(
    "-p",
    "--trap",
    nargs="+",
    type=float,
    help="Trap parameters of potential: [V_0 w] in unit of kHz and meter.",
    default=[104.52, 1E-6])
parser.add_argument(
    "-t",
    "--time",
    nargs="+",
    type=float,
    help="Number of steps and stop time in second (1/omega): [step, st]",
    default=[1000.0, 0])
parser.add_argument("-d",
                    "--dim",
                    type=int,
                    help="Dimension of the system.",
                    default=3)
parser.add_argument("-m",
                    "--model",
                    type=str,
                    help="Model to calculate.",
                    default='Gaussian')
parser.add_argument("-sy",
                    "--symmetry",
                    action="store_false",
                    help="Determine to use reflection symmetries or not.",
                    default=True)
parser.add_argument(
    "-o",
    "--sample",
    action="store_false",
    help="Determine to use stop_time/step_no or T as the sampling timespan.",
    default=True)
parser.add_argument("-rt",
                    "--realtime",
                    action="store_true",
                    help="Determine to do realtime dynamics or not.",
                    default=False)
parser.add_argument(
    "-e",
    "--mem_eff",
    action="store_true",
    help=
    "Determine to be memory efficient or not. This has a performance loss.",
    default=False)
parser.add_argument(
    "-a",
    "--absorption",
    action="store_true",
    help="Determine to use absorption potential at boundary or not.",
    default=False)
parser.add_argument("-ap",
                    "--ab_param",
                    nargs="+",
                    type=float,
                    help="Absorption parameters: [V_0I, L].",
                    default=[57.04, 1])
args = parser.parse_args()

# 3D tweezer potential
# Rough estimation on memory:
# grid point = [2n+1 2n+1 4n+3]
# n = 10, size of matrix ~ 3G, time: avg ~3.6h each freq
# n = 15, size of matrix ~ 27G
# n = 20, size of matrix ~ 145G
# n = 30, size of matrix ~ 1561G

# 3D harmonic potential
# Rough estimation on memory:
# grid point = [2n+1 2n+1 2n+1]
# typical n choice n = 20 or 15, dx = 0.4
# n = 15, size of matrix ~ 7G, time: avg ~3.6h each freq
# n = 20, size of matrix ~ 36G
# n = 30, size of matrix ~ 380G

# 3D NEW tweezer potential
# Rough estimation on memory:
# grid point = [2n+1 2n+1 2n+1]
# typical n choice n = 18, dx = 0.15w dz = 0.36w
# n = 6, size of matrix ~ 73M, actual mem usage 480MiB, time: avg ~25s each freq
#         if mem_eff enabled, actual mem usage 111MiB, time: avg ~45s each freq
#         HOWEVER, this is not true at peak in diagonalization, which is not monitored by functions
# n = 12, size of matrix ~ 3G, actual mem usage 13G, time: avg ~28min each freq
# n = 14, size of matrix ~ 9G, actual mem usage 59G, time: avg ~1.3h each freq
# n = 16, size of matrix ~ 19G, time: avg ~3.6h each freq
# n = 18, size of matrix ~ 39G
# n = 20, size of matrix ~ 72G
# n = 30, size of matrix ~ 760G

# Complex number, size of matrix is double.
# A max memory usage is roughly 6x to the matrix size. So matrix size should be restricted to 20GiB.

# WITH SYMMETRY, the effective n is reduced by half in each dimension.
# SO WITH SYMMETRY THE MATRIX SIZE OF n = 12 IS ROUGHLY ORIGINALLY n = 6

# NORMAL WAIST
# V0_SI = 104.52 * 1E3 * 2 * np.pi  # 104.52kHz * h, potential depth, in SI unit, since hbar is set to 1 this should be multiplied by 2pi
# w = 1E-6 / Par.a0  # ~1000nm, waist length, in unit of Bohr radius
# FATTEST WAIST
# V0_SI = 156 * 1E3 * 2 * np.pi  # trap depth for fattest waist
# w = 1.18E-6 / a0  # fattest waist length
# TIGHTEST WAIST
# V0_SI = 76 * 1E3 * 2 * np.pi  # trap depth for tightest waist
# w = 8.61E-7 / a0  # tightest waist length

if len(args.freq) == 1:
    freq_list = np.array(args.freq)
elif len(args.freq) == 3:
    freq_list = np.arange(args.freq[0], args.freq[1] + args.freq[2],
                          args.freq[2])

N = args.number
R = np.array(args.range)
trap = args.trap
step, t = args.time

st = get_stop_time(freq_list, t, trap[0] * 1E3 * 2 * np.pi)

if __debug__:
    print(N)
    print(R)
    print(freq_list)
    print(step, st)
    print(trap)

dvr = DVRdynamics(N, R, freq_list, (step, st), 1 / 2, args.dim, args.model,
                  trap, args.mem_eff, True, args.realtime, (-1, 10),
                  args.symmetry, args.absorption, args.ab_param)

DVRdynamics_exe(dvr)
