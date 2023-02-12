# importing the library
import numpy as np

from DVR.core import *
from DVR.mem_scaling import profile

N_list = range(20, 21, 2)
R = 3 * np.array([w, w, 2.4 * w])
freq_list = np.array([.005, .01, .02, .04, .06, .08, .1, .12, .16, .2])
freq_list = freq_list[[2] + list(range(4, 8)) + [-1]]
an = 0
d = 2
length = 1
# st = 2E-0
st = [1E-2 for i in range(2, 6, 2)
      ] + [0.125] + [5E-5 * np.exp(i * .6) for i in range(8, 31, 2)]
# st = st[0:1]
# freq_list = freq_list[0:1]
sn = 1000.0
N = 20


# instantiating the decorator
@profile
def func(n):
    DVR(n, R,
        freq_list=freq_list,
        stop_time=st,
        step_no=sn,
        dim=d,
        model='Gaussian',
        auto_t_step=False,
        realtime=False,
        absorber=True)


func(N)
