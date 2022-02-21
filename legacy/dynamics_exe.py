import numpy as np
from dynamics import *
from DVR3d_exe import DVR3d_exe
import argparse

dsp = 'Tweezer stroboscopic dynamics by 3D DVR.'
parser = argparse.ArgumentParser(description=dsp)
parser.add_argument("-n",
                    "--number",
                    type=int,
                    help="Number of grid points.",
                    default=0)
parser.add_argument(
    "-f",
    "--freq",
    nargs="+",
    type=float,
    help="Frequencies to calculate: [init_freq final_freq freq_step].",
    default=[0.02, 0.1, 0.02])
parser.add_argument(
    "-t",
    "--time",
    type=float,
    help="Stop time of dynamics in unit of V0 (omega). Default 20s (50/omega).",
    default=0)
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
parser.add_argument(
    "-o",
    "--sample",
    action="store_false",
    help="Determine to use stop_time/200 or T as the sampling timespan.",
    default=True)
parser.add_argument("-r",
                    "--realtime",
                    action="store_true",
                    help="Determine to do realtime dynamics or not.",
                    default=False)
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

if len(args.freq) == 1:
    freq_list = args.freq
elif len(args.freq) == 3:
    freq_list = np.arange(args.freq[0], args.freq[1] + args.freq[2],
                          args.freq[2]).tolist()

DVR3d_exe(args.number,
          freq_list,
          stop_time=args.time,
          dim=args.dim,
          model=args.model,
          auto_t_step=args.sample,
          realtime=args.realtime)
