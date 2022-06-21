from DVR_dynamics import *
from DVR_exe import DVRdynamics_exe

# 3D harmonic potential
print("3D harmonic potential starts.")
# Rough estimation on memory:
# grid point = [2n+1 2n+1 2n+1]
# typical n choice n = 20 or 15, dx = 0.4
# n = 15, size of matrix ~ 7G
# n = 20, size of matrix ~ 36G
# n = 30, size of matrix ~ 380G
N = 15
freq_list = [10.**i for i in range(-1, 2)]

DVR3d_exe(N, freq_list, model='sho')
