from typing import Iterable
from opt_einsum import contract
from pyrsistent import get_in
from scipy.integrate import romb
from scipy.optimize import minimize
import numpy as np
from wannier import *
import numpy.linalg as la


class Equalizer(Wannier):

    def __init__(
            self,
            N,
            equalize=False,  # Homogenize trap or not
            eqtarget='vt',  # Equalization target
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        if equalize:
            self.eq_label = 'eq'
            self.homogenize(eqtarget)

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

    def homogenize(self, target: str = 'vt'):
        # fij = self.trap_mat()

        # def cost_func(V):
        #     return la.norm(fij @ V - 1)

        if 'v' in target:
            self.Voff = self.v_equalize(u=False)
            print('Onsite potential homogenized.\n')
        elif 'V' in target:
            # Combined cost function for U and V is used
            self.Voff = self.v_equalize(u=True)
            print('Onsite potential homogenized.\n')

        if 'u' in target:
            self.Voff = self.u_equalize()
            print('Onsite interaction homogenized.\n')

        if 't' in target:
            self.trap_centers = self.t_equalize(v=False)
            print('Tunneling homogenized.\n')
        elif 'T' in target:
            # Combined cost function for t and V is used
            self.trap_centers = self.t_equalize(v=True)
            print('Tunneling homogenized.\n')

        return self.Voff, self.trap_centers

    def v_cost_func(self, offset: np.ndarray, target=None, u: bool = False):
        # If target = None, then U and V are targeted to mean values
        # If target is given, for V it's float value, for U and V it's a tuple
        self.symm_unfold(self.Voff, offset)
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
            a = abs(Vtarget / Utarget)
            cu = la.norm(U - Utarget)
            print(f'Scale factor a={a} is applied.')
            print(f'Onsite interaction target={Utarget}')
            print(f'Onsite interaction distance u={cu}')
            c += a * cu
        return c

    def v_equalize(self, u=False):
        # Equalize onsite chemical potential
        Voff_bak = self.Voff

        res = self.singleband_solution(u)
        if u:
            A, U = res
            Utarget = np.mean(U)
        else:
            A = res
            Utarget = None
        Vtarget = np.mean(np.real(np.diag(A)))

        def cost_func(offset: np.ndarray):
            print("\nCurrent trap depths:", offset)
            c = self.v_cost_func(offset, (Vtarget, Utarget), u)
            print("Current total cost:", c, "\n")
            return c

        v0 = np.ones(self.Nindep)
        # Bound trap depth variation
        bonds = tuple((0.9, 1.1) for i in range(self.Nindep))
        res = minimize(cost_func, v0, bounds=bonds)
        self.symm_unfold(self.Voff, res.x)
        return self.Voff

    def u_cost_func(self, offset: np.ndarray, target=None):
        self.symm_unfold(self.Voff, offset)
        A, U = self.singleband_solution(u=True)
        target = None
        if not isinstance(target, (float, int)):
            target = np.mean(U)
        c = la.norm(U - target)
        print(f'Onsite potential target={target}')
        print(f'Onsite interaction distance u={c}')
        return c

    def u_equalize(self):
        # Equalize onsite chemical potential
        Voff_bak = self.Voff

        A, U = self.singleband_solution(u=True)
        Utarget = np.mean(U)

        def cost_func(offset: np.ndarray):
            print("\nCurrent trap depths:", offset)
            c = self.u_cost_func(offset, Utarget)
            print("Current total cost:", c, "\n")
            return c

        v0 = np.ones(self.Nindep)
        # Bound trap depth variation
        bonds = tuple((0.9, 1.1) for i in range(self.Nindep))
        res = minimize(cost_func, v0, bounds=bonds)
        self.symm_unfold(self.Voff, res.x)
        return self.Voff

    def t_cost_func(self,
                    offset,
                    links: tuple,
                    target: tuple,
                    v: bool = False):
        xlinks, ylinks = links
        nntx, nnty = target[:2]
        offset = offset.reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, offset, graph=True)
        self.update_lattice(self.trap_centers)
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
            if len(target) == 3:
                Vtarget = target[-1]
            else:
                Vtarget = np.mean(V)
            cv = la.norm(V - Vtarget)
            # adjust factor on onsite potential cost function
            a = abs(nntx / Vtarget)
            print(f'Scale factor a={a} is applied.')
            print(f'Onsite potential target={Vtarget}')
            print(f'Onsite potential distance v={cv}')
            c += a * cv
        return c

    def t_equalize(self, v: bool = False) -> np.ndarray:
        # Equalize tunneling
        ls_bak = self.trap_centers

        A = self.singleband_solution()
        nnt = self.nn_tunneling(A)
        xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        Vtarget = None
        if v:
            Vtarget = np.mean(np.real(np.diag(A)))

        def cost_func(offset: np.ndarray):
            print("\nCurrent trap centers:", offset)
            c = self.t_cost_func(offset, (xlinks, ylinks),
                                 (nntx, nnty, Vtarget), v)
            print("Current total cost:", c, "\n")
            return c

        v0 = self.trap_centers[self.reflection[:, 0]]
        print('v0', v0)
        # Bound lattice spacing variation
        xbonds = tuple(
            (v0[i, 0] - 0.05, v0[i, 0] + 0.05) for i in range(self.Nindep))
        if self.lattice_dim == 1:
            ybonds = tuple((0, 0) for i in range(self.Nindep))
        else:
            ybonds = tuple(
                (v0[i, 1] - 0.05, v0[i, 1] + 0.05) for i in range(self.Nindep))
        nested = tuple((xbonds[i], ybonds[i]) for i in range(self.Nindep))
        bonds = tuple(item for sublist in nested for item in sublist)
        print('bounds', bonds)
        res = minimize(cost_func, v0.reshape(-1), bounds=bonds)
        self.symm_unfold(self.trap_centers,
                         res.x.reshape(self.Nindep, 2),
                         graph=True)
        self.update_lattice(self.trap_centers)
        return self.trap_centers
