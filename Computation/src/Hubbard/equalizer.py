import numpy as np
import numpy.linalg as la
from typing import Callable, Iterable, Union
from opt_einsum import contract
from pyparsing import Char
from scipy.integrate import romb
from scipy.optimize import minimize, shgo

from .core import *


class HubbardParamEqualizer(MLWF):

    def __init__(
            self,
            N,
            equalize=False,  # Homogenize trap or not
            eqtarget='Ut',  # Equalization target
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        if equalize:
            self.eq_label = 'eq'
            # self.homogenize(eqtarget, fixed)
            self.equalzie(eqtarget, callback=True)

    def equalzie(self,
                 target: str = 'Ut',
                 weight: np.ndarray = np.ones(3),
                 random: bool = False,
                 callback: bool = False):
        if self.verbosity:
            print(f"Equalizing {target}.")
        u, t, v = False, False, False
        fix_u, fix_v = False, False
        if 'u' in target or 'U' in target:
            u = True
            if 'U' in target:
                # Whether to fix target in combined cost function
                fix_u = True
        if 't' in target:
            t = True
        if 'v' in target or 'V' in target:
            v = True
            if 'V' in target:
                fix_v = True

        res = self.singleband_Hubbard(u=u, output_unitary=True)
        if u:
            A, U, V = res
        else:
            A, V = res
            U = None

        if fix_u:
            Utarget = np.mean(U)
        else:
            Utarget = None
        if t:
            nnt = self.nn_tunneling(A)
            xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        else:
            nnt, xlinks, ylinks, nntx, nnty = None, None, None, None, None
        if fix_v:
            Vtarget = np.mean(np.real(np.diag(A)))
        else:
            Vtarget = None

        Voff_bak = self.Voff
        ls_bak = self.trap_centers
        v0, bounds = self.init_guess(random=random)

        # Decide if each step cost function used the last step's unitary matrix
        # callback can have sometimes very few iteraction steps
        if callback:
            # Pack x0 to be mutable, thus can be updated in each iteration of minimize
            x0 = [V]
        else:
            x0 = None

        def cost_func(offset: np.ndarray, info: Union[dict, None]) -> float:
            c = self.cbd_cost_func(offset, info, (xlinks, ylinks),
                                   (Vtarget, Utarget, nntx, nnty), (u, t, v), weight, x0)

            return c

        info = {'Nfeval': 0,
                'cost': np.array([]).reshape(0, 3),
                'ctot': np.array([]),
                'fval': np.array([]),
                'diff': np.array([]),
                'x': np.array([]).reshape(0, *v0.shape)}
        t0 = time()
        # res = minimize(cost_func, v0, bounds=bounds, method='Nelder-Mead', options={
        #                'disp': True, 'return_all': True, 'adaptive': True})
        res = minimize(cost_func, v0, args=info, bounds=bounds, options={
            'disp': 0, 'ftol': 1e-9})
        t1 = time()
        if self.verbosity:
            print(f"Equalization took {t1 - t0} seconds.")

        trap_depth = res.x[:self.Nindep]
        trap_center = res.x[self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.Voff, trap_depth)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)
        return self.Voff, self.trap_centers, info

    def init_guess(self, random=False) -> tuple[np.ndarray, tuple]:
        v01 = np.ones(self.Nindep)
        v02 = self.trap_centers[self.reflection[:, 0]]
        # Bound trap depth variation
        b1 = list((0.9, 1.1) for i in range(self.Nindep))
        # Bound lattice spacing variation
        xbounds = tuple(
            (v02[i, 0] - 0.05, v02[i, 0] + 0.05) for i in range(self.Nindep))
        if self.lattice_dim == 1:
            ybounds = tuple((0, 0) for i in range(self.Nindep))
        else:
            ybounds = tuple((v02[i, 1] - 0.05, v02[i, 1] + 0.05)
                            for i in range(self.Nindep))
        nested = tuple((xbounds[i], ybounds[i]) for i in range(self.Nindep))
        b2 = list(item for sublist in nested for item in sublist)

        v0 = np.concatenate((v01, v02.reshape(-1)))
        bounds = tuple(b1 + b2)

        if random:
            v0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            trap_depth = v0[:self.Nindep]
            trap_center = v0[self.Nindep:].reshape(self.Nindep, 2)

        if self.verbosity or random:
            print(f"Initial trap depths: {trap_depth}")
            print("Initial trap centers:")
            print(trap_center)

        return v0, bounds

    def cbd_cost_func(self,
                      offset: np.ndarray,
                      info: Union[dict, None],
                      links: tuple[np.ndarray, np.ndarray],
                      target: tuple[float, ...],
                      utv: tuple[bool] = (False, False, False),
                      weight: np.ndarray = np.ones(3),
                      unitary: Union[list, None] = None) -> float:

        trap_depth = offset[:self.Nindep]
        trap_center = offset[self.Nindep:].reshape(self.Nindep, 2)
        if self.verbosity:
            print(f"\nCurrent trap depths: {trap_depth}")
            print("Current trap centers:")
            print(trap_center)
        self.symm_unfold(self.Voff, trap_depth)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        if unitary != None and self.lattice_dim > 1:
            x0 = unitary[0]
        else:
            x0 = None

        u, t, v = utv

        res = self.singleband_Hubbard(
            u=u, x0=x0, output_unitary=True)
        if u:
            A, U, x0 = res
        else:
            A, x0 = res
            U = None

        # By accessing element of a list, x0 is mutable and can be updated
        if unitary != None and self.lattice_dim > 1:
            unitary[0] = x0

        xlinks, ylinks = links
        Vtarget = None
        Utarget = None
        nntx, nnty = None, None
        if isinstance(target, Iterable):
            Vtarget, Utarget, nntx, nnty = target

        w = weight.copy()
        cu = 0
        if u:
            # U is different, as calculating U costs time
            cu = self.u_cost_func(U, Utarget)

        ct = self.t_cost_func(A, (xlinks, ylinks), (nntx, nnty))
        if not t:
            # Force t to have no effect on cost function
            w[1] = 0

        cv = self.v_cost_func(A, Vtarget)
        if not v:
            # Force V to have no effect on cost function
            w[2] = 0

        cvec = np.array((cu, ct, cv))
        c = w @ cvec
        if self.verbosity:
            print(f"Current total distance: {c}\n")

        # Keep revcord
        if info != None:
            info['Nfeval'] += 1
            info['x'] = np.append(info['x'], offset[None], axis=0)
            info['cost'] = np.append(info['cost'], cvec[None], axis=0)
            ctot = np.sum(cvec)
            info['ctot'] = np.append(info['ctot'], ctot)
            info['fval'] = np.append(info['fval'], c)
            diff = info['fval'][len(info['fval'])//2] - c
            info['diff'] = np.append(info['diff'], diff)
            # display information
            if info['Nfeval'] % 50 == 0:
                print(
                    f'i={info["Nfeval"]}\tc={cvec}\tc_i={c}\tc_i//2-c_i={diff}')

        return c

    def v_cost_func(self, A, Vtarget) -> float:
        if Vtarget is None:
            Vtarget = np.mean(np.real(np.diag(A)))
        cv = la.norm(np.real(np.diag(A)) - Vtarget) / \
            abs(Vtarget * np.sqrt(len(A)))
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Onsite potential target={Vtarget}')
            print(f'Onsite potential normalized distance v={cv}')
        return cv

    def t_cost_func(self, A: np.ndarray, links: tuple[np.ndarray, np.ndarray],
                    target: tuple[float, ...]) -> float:
        nnt = self.nn_tunneling(A)
        if target is None:
            xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        elif isinstance(target, Iterable):
            xlinks, ylinks = links
            nntx, nnty = target
            if nntx is None:
                xlinks, ylinks, nntx, nnty = self.xy_links(nnt)

        dist = (abs(nnt[xlinks]) - nntx) / (nntx * np.sqrt(len(xlinks)))
        if nnty != None:
            dist = np.concatenate(
                (dist, (abs(nnt[ylinks]) - nnty) / (nnty * np.sqrt(len(ylinks)))))
        ct = la.norm(dist)
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Tunneling target=({nntx}, {nnty})')
            print(f'Tunneling normalized distance t={ct}')
        return ct

    def u_cost_func(self, U, Utarget) -> float:
        if Utarget is None:
            Utarget = np.mean(U)
        cu = la.norm(U - Utarget) / abs(Utarget * np.sqrt(len(U)))
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Onsite interaction target fixed to {Utarget}')
            print(f'Onsite interaction normalized distance u={cu}')
        return cu


# ===================== TO BE DEPRECATED =====================================

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
