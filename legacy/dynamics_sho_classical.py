import numpy as np
from scipy.integrate import solve_ivp


w = 1


def force(x):
    return -w**2 * x


def f(t, T, y, x0):
    t1 = T / 2
    t2 = t % T

    def firstfun(x):
        return force(x-x0)

    def lastfun(x):
        return force(x+x0)

    if t2 >= t1:
        dydt = lastfun(y)
    else:
        dydt = firstfun(y)
    return dydt


def EoM(t, T, y, x0):
    dydt = np.zeros((6,))
    dydt[0:3] = y[3:6]
    dydt[3:6] = f(t, T, y[0:3], x0)
    return dydt


def ODE_solver(T, t_list, t_step, x0):
    def dydt(t, y):
        return EoM(t, T, y, x0)

    t_span = [t_list[0], t_list[-1]]
    sol = solve_ivp(dydt, t_span, np.zeros((6,)), max_step=t_step)
    return sol
