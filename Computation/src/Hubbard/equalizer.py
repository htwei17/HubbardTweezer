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
            eqV=False,  # Equalize V or not
            eqtarget='vt',  # Equalization target
            fixed=False,  # Whether to fix target in combined cost function
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        if equalize:
            self.eq_label = 'eq'
            # self.homogenize(eqtarget, fixed)
            self.equalzie(v=eqV, fixed=fixed)

    def equalzie(self, v: bool = False, fixed: bool = False):
        A, U, V = self.singleband_Hubbard(u=True, output_unitary=True)

        Utarget = np.mean(U)
        nnt = self.nn_tunneling(A)
        xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        if fixed:
            Vtarget = np.mean(np.real(np.diag(A)))
        else:
            Vtarget = None

        Voff_bak = self.Voff
        ls_bak = self.trap_centers
        v0, bonds = self.init_optimize()

        def cost_func(offset: np.ndarray) -> float:
            c = self.cbd_cost_func(offset, (xlinks, ylinks),
                                   (Vtarget, Utarget, nntx, nnty), v, V)
            return c

        res = minimize(cost_func, v0, bounds=bonds, method='Nelder-Mead')

        trap_depth = res.x[:self.Nindep]
        trap_center = res.x[self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.Voff, trap_depth)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)
        return self.Voff, self.trap_centers

    def init_optimize(self):
        v01 = np.ones(self.Nindep)
        v02 = self.trap_centers[self.reflection[:, 0]]
        # Bound trap depth variation
        b1 = list((0.9, 1.1) for i in range(self.Nindep))
        # Bound lattice spacing variation
        xbonds = tuple(
            (v02[i, 0] - 0.05, v02[i, 0] + 0.05) for i in range(self.Nindep))
        if self.lattice_dim == 1:
            ybonds = tuple((0, 0) for i in range(self.Nindep))
        else:
            ybonds = tuple((v02[i, 1] - 0.05, v02[i, 1] + 0.05)
                           for i in range(self.Nindep))
        nested = tuple((xbonds[i], ybonds[i]) for i in range(self.Nindep))
        b2 = list(item for sublist in nested for item in sublist)

        v0 = np.concatenate((v01, v02.reshape(-1)))
        bonds = tuple(b1 + b2)
        return v0, bonds

    def cbd_cost_func(self,
                      offset: np.ndarray,
                      links: tuple[np.ndarray, np.ndarray],
                      target: tuple[float, ...],
                      v: bool = False, unitary=None) -> float:

        trap_depth = offset[:self.Nindep]
        trap_center = offset[self.Nindep:].reshape(self.Nindep, 2)
        print("\nCurrent trap depths:", trap_depth)
        print("\nCurrent trap centers:", trap_center)
        self.symm_unfold(self.Voff, trap_depth)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        A, U = self.singleband_Hubbard(u=True, x0=unitary)

        xlinks, ylinks = links
        Vtarget = None
        Utarget = None
        nntx, nnty = None, None
        if isinstance(target, Iterable):
            Vtarget, Utarget, nntx, nnty = target

        if v:
            cv = self.v_cost_func(A, Vtarget)
        else:
            cv = 0

        ct = self.t_cost_func(A, (xlinks, ylinks), (nntx, nnty))

        cu = self.u_cost_func(U, Utarget)

        c = cv + ct + cu
        print(f"Current total distance: {c}")
        return c

    def v_cost_func(self, A, Vtarget) -> float:
        if Vtarget is None:
            Vtarget = np.mean(np.real(np.diag(A)))
        print(f'Onsite potential target={Vtarget}')
        cv = la.norm(np.real(np.diag(A)) - Vtarget) / abs(Vtarget)
        print(f'Onsite potential normalized distance v={cv}')
        return cv

    def t_cost_func(self, A: np.ndarray, links: tuple[np.ndarray, np.ndarray],
                    target: tuple[float, ...]) -> float:
        nnt = self.nn_tunneling(A)
        if target is None:
            xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        else:
            xlinks, ylinks = links
            nntx, nnty = target

        print(f'Tunneling target=({nntx}, {nnty})')
        dist = (abs(nnt[xlinks]) - nntx) / abs(nntx)
        if any(ylinks == True):
            dist = np.concatenate(
                (dist, (abs(nnt[ylinks]) - nnty) / abs(nntx)))
        ct = la.norm(dist)
        print(f'Tunneling normalized distance t={ct}')
        return ct

    def u_cost_func(self, U, Utarget) -> float:
        if Utarget is None:
            Utarget = np.mean(U)
        print(f'Onsite interaction target fixed to {Utarget}')
        cu = la.norm(U - Utarget) / abs(Utarget)
        print(f'Onsite interaction normalized distance u={cu}')
        return cu


# ================================ TO BE DEPRACATED ================================


    def homogenize(self, target: str = 'vt', fixed=False):
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

    def v_equalize(self, u, fixed=False) -> Callable[[np.ndarray], float]:
        res = self.singleband_Hubbard(u)
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

            res = self.singleband_Hubbard(u)
            if u:
                A, U = res
            else:
                A = res

            c = self.v_cost_func(A, Vtarget)
            if u:
                c += self.u_cost_func(U, Utarget)
            print("Current total cost:", c, "\n")
            return c

        return cost_func

    def u_equalize(self) -> Callable[[np.ndarray], float]:
        # Equalize onsite chemical potential
        A, U = self.singleband_Hubbard(u=True)
        Utarget = np.mean(U)

        def cost_func(offset: np.ndarray, offset_type) -> float:
            if offset_type == 'd':
                self.symm_unfold(self.Voff, offset)
                print("\nCurrent trap depths:", offset)
            elif offset_type == 's':
                offset = offset.reshape(self.Nindep, 2)
                self.symm_unfold(self.trap_centers, offset, graph=True)
                self.update_lattice(self.trap_centers)
                print("\nCurrent trap centers:", offset)

            A, U = self.singleband_Hubbard(u=True)
            c = self.u_cost_func(U, Utarget)
            print("Current total cost:", c, "\n")
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

    def t_equalize(self, v, fixed=False) -> Callable[[np.ndarray], float]:
        A = self.singleband_Hubbard()
        nnt = self.nn_tunneling(A)
        xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        Vtarget = None
        if fixed:
            Vtarget = np.mean(np.real(np.diag(A)))

        def cost_func(offset: np.ndarray, offset_type) -> float:
            if offset_type == 'd':
                self.symm_unfold(self.Voff, offset)
                print("\nCurrent trap depths:", offset)
            elif offset_type == 's':
                offset = offset.reshape(self.Nindep, 2)
                self.symm_unfold(self.trap_centers, offset, graph=True)
                self.update_lattice(self.trap_centers)
                print("\nCurrent trap centers:", offset)

            A = self.singleband_Hubbard()
            c = self.t_cost_func(A, (xlinks, ylinks), (nntx, nnty))
            if v:
                c += self.v_cost_func(A, Vtarget)
            print("Current total cost:", c, "\n")
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
