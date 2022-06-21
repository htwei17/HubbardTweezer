from DVR_dynamics import *
from DVR_exe import DVRdynamics_exe

# 3D tweezer potential
print("3D tweezer potential starts.")

# Rough estimation on memory:
# grid point = [2n+1 2n+1 4n+3]
# n = 10, size of matrix ~ 3G
# n = 15, size of matrix ~ 27G
# n = 20, size of matrix ~ 145G
# n = 30, size of matrix ~ 1561G
N = 10
freq_list = [0.02 * 10.**i for i in range(0, 3)]

DVR3d_exe(N, freq_list, model='Gaussian')
