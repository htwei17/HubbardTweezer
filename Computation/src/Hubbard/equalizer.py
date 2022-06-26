import numpy as np
import numpy.linalg as la
from typing import Callable, Iterable
from opt_einsum import contract
from pyparsing import Char
from scipy.integrate import romb
from scipy.optimize import minimize

from .core import *


class HubbardParamEqualizer(MLWF):

    def __init__(
            self,
            N,
            equalize=False,  # Homogenize trap or not
            eqtarget='vt',  # Equalization target
            fixed=False,  # Whether to fix target in combined cost function
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        if equalize:
            self.eq_label = 'eq'
            self.homogenize(eqtarget, fixed)

    # def trap_mat(self):
    #     tc = np.zeros((self.Nsite, dim))
    #     fij = np.ones((self.Nsite, self.Nsite))
    #     for i in range(self.Nsite):
    #         tc[i, :2] = self.graph[i] * self.lc
    #         tc[i, 2] = 0
    #         for j in range(i):
    #             # print(*(tc[i] - tc[j]))
    #             fij[i, j] = -super().Vfun(*(tc[i] - tc[j]))
    #             fij[j, i] = fij[i, j]  # Potential is symmetric in distance
    #     return fij

    def homogenize(self, target: str = 'vt', fixed=False):
        # fij = self.trap_mat()

        # def cost_func(V):
        #     return la.norm(fij @ V - 1)

        # Force target to be 2-character string
        if len(target) == 1:
            if target == 't' or target == 'T':
                # Tunneling is varying spacings in default
                target = '0' + target
            else:
                # Other is varying trap depths in default
                target = target + '0'

        cost_func, quantity = self.one_equzlize(target[0], fixed)
        self.Voff = self.depth_equalize(cost_func)
        if quantity != None:
            print(f'{quantity} homogenized by trap depths.\n')

        cost_func, quantity = self.one_equzlize(target[1], fixed)
        self.trap_centers = self.spacing_equalize(cost_func)
        if quantity != None:
            print(f'{quantity} homogenized by trap spacings.\n')

        return self.Voff, self.trap_centers

    def one_equzlize(self, target: str, fixed=False):
        if 'v' in target:
            cost_func = self.v_equalize(u=False)
            quantity = 'Onsite potential'
        elif 'V' in target:
            # Combined cost function for U and V is used
            cost_func = self.v_equalize(u=True, fixed=fixed)
            quantity = 'Onsite potential combining interaction'
        elif 'u' in target:
            cost_func = self.u_equalize()
            quantity = 'Onsite interaction'
        elif 't' in target:
            cost_func = self.t_equalize(v=False)
            quantity = 'Tunneling'
        elif 'T' in target:
            # Combined cost function for t and V is used
            cost_func = self.t_equalize(v=True, fixed=fixed)
            quantity = 'Tunneling combining onsite potential'
        else:
            cost_func = None
            quantity = None
            print('Input target not recognized.')
        return cost_func, quantity

    def v_cost_func(self,
                    offset: np.ndarray,
                    offset_type: str = 'd',
                    target=None,
                    u: bool = False) -> float:
        # If target = None, then U and V are targeted to mean values
        # If target is given, for V it's float value, for U and V it's a tuple
        if offset_type == 'd':
            self.symm_unfold(self.Voff, offset)
            print("\nCurrent trap depths:", offset)
        elif offset_type == 's':
            offset = offset.reshape(self.Nindep, 2)
            self.symm_unfold(self.trap_centers, offset, graph=True)
            self.update_lattice(self.trap_centers)
            print("\nCurrent trap centers:", offset)

        res = self.singleband_solution(u)
        Vtarget = None
        Utarget = None
        if u:
            A, U = res
            if isinstance(target, Iterable):
                Vtarget, Utarget = target
        else:
            A = res
            Vtarget = target
        if not isinstance(Vtarget, (float, int)):
            Vtarget = np.mean(np.real(np.diag(A)))
        c = la.norm(np.real(np.diag(A)) - Vtarget)
        print(f'Onsite potential target={Vtarget}')
        print(f'Onsite potential distance v={c}')
        if u:
            if not isinstance(Utarget, (float, int)):
                Utarget = np.mean(U)
            else:
                print(f'Onsite interaction target fixed to {Utarget}')
            a = abs(Vtarget / Utarget)
            cu = la.norm(U - Utarget)
            print(f'Scale factor a={a} is applied.')
            print(f'Onsite interaction target={Utarget}')
            print(f'Onsite interaction distance u={cu}')
            c += a * cu

        print("Current total cost:", c, "\n")
        return c

    def v_equalize(self, u, fixed=False) -> Callable[[np.ndarray], float]:
        res = self.singleband_solution(u)
        if u:
            A, U = res
        else:
            A = res
        if fixed:
            Utarget = np.mean(U)
        else:
            Utarget = None
        Vtarget = np.mean(np.real(np.diag(A)))

        def cost_func(offset: np.ndarray, offset_type) -> float:
            c = self.v_cost_func(offset, offset_type, (Vtarget, Utarget), u)
            return c

        return cost_func

    def u_cost_func(self,
                    offset: np.ndarray,
                    offset_type: str = 'd',
                    target=None) -> float:

        if offset_type == 'd':
            self.symm_unfold(self.Voff, offset)
            print("\nCurrent trap depths:", offset)
        elif offset_type == 's':
            offset = offset.reshape(self.Nindep, 2)
            self.symm_unfold(self.trap_centers, offset, graph=True)
            self.update_lattice(self.trap_centers)
            print("\nCurrent trap centers:", offset)

        A, U = self.singleband_solution(u=True)
        target = None
        if not isinstance(target, (float, int)):
            target = np.mean(U)
        c = la.norm(U - target)
        print(f'Onsite interaction target={target}')
        print(f'Onsite interaction distance u={c}')
        print("Current total cost:", c, "\n")
        return c

    def u_equalize(self) -> Callable[[np.ndarray], float]:
        # Equalize onsite chemical potential
        A, U = self.singleband_solution(u=True)
        Utarget = np.mean(U)

        def cost_func(offset: np.ndarray, offset_type) -> float:
            c = self.u_cost_func(offset, offset_type, Utarget)
            return c

        return cost_func

    def depth_equalize(self, cost_func) -> np.ndarray:
        # Equalize onsite chemical potential

        if cost_func != None:
            Voff_bak = self.Voff

            v0 = np.ones(self.Nindep)
            # Bound trap depth variation
            bonds = tuple((0.9, 1.1) for i in range(self.Nindep))
            res = minimize(cost_func, v0, 'd', bounds=bonds)
            self.symm_unfold(self.Voff, res.x)
        return self.Voff

    def t_cost_func(self,
                    offset,
                    offset_type: str,
                    links: tuple,
                    target: tuple,
                    v: bool = False) -> float:
        xlinks, ylinks = links
        nntx, nnty = target[:2]

        if offset_type == 'd':
            self.symm_unfold(self.Voff, offset)
            print("\nCurrent trap depths:", offset)
        elif offset_type == 's':
            offset = offset.reshape(self.Nindep, 2)
            self.symm_unfold(self.trap_centers, offset, graph=True)
            self.update_lattice(self.trap_centers)
            print("\nCurrent trap centers:", offset)

        A = self.singleband_solution()
        nnt = self.nn_tunneling(A)
        dist = abs(nnt[xlinks]) - nntx
        if any(ylinks == True):
            dist = np.concatenate((dist, abs(nnt[ylinks]) - nnty))
        c = la.norm(dist)
        print(f'Onsite potential target=({nntx}, {nnty})')
        print(f'Tunneling distance t={c}')
        if v:
            V = np.real(np.diag(A))
            if len(target) == 3 and isinstance(target[2], (float, int)):
                print(f'Onsite potential target fixed to {Vtarget}')
                Vtarget = target[2]
            else:
                Vtarget = np.mean(V)
            cv = la.norm(V - Vtarget)
            # adjust factor on onsite potential cost function
            a = abs(nntx / Vtarget)
            print(f'Scale factor a={a} is applied.')
            print(f'Onsite potential target={Vtarget}')
            print(f'Onsite potential distance v={cv}')
            c += a * cv

        print("Current total cost:", c, "\n")
        return c

    def t_equalize(self, v, fixed=False) -> Callable[[np.ndarray], float]:
        A = self.singleband_solution()
        nnt = self.nn_tunneling(A)
        xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        Vtarget = None
        if fixed:
            Vtarget = np.mean(np.real(np.diag(A)))

        def cost_func(offset: np.ndarray, offset_type) -> float:
            c = self.t_cost_func(offset, offset_type, (xlinks, ylinks),
                                 (nntx, nnty, Vtarget), v)
            return c

        return cost_func

    def spacing_equalize(self, cost_func) -> np.ndarray:
        # Equalize tunneling
        if cost_func != None:
            ls_bak = self.trap_centers

            v0 = self.trap_centers[self.reflection[:, 0]]
            # print('v0', v0)
            # Bound lattice spacing variation
            xbonds = tuple(
                (v0[i, 0] - 0.05, v0[i, 0] + 0.05) for i in range(self.Nindep))
            if self.lattice_dim == 1:
                ybonds = tuple((0, 0) for i in range(self.Nindep))
            else:
                ybonds = tuple((v0[i, 1] - 0.05, v0[i, 1] + 0.05)
                               for i in range(self.Nindep))
            nested = tuple((xbonds[i], ybonds[i]) for i in range(self.Nindep))
            bonds = tuple(item for sublist in nested for item in sublist)
            # print('bounds', bonds)
            res = minimize(cost_func, v0.reshape(-1), 's', bounds=bonds)
            self.symm_unfold(self.trap_centers,
                             res.x.reshape(self.Nindep, 2),
                             graph=True)
            self.update_lattice(self.trap_centers)
        return self.trap_centers