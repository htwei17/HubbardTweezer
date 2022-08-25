from tkinter import E
import numpy as np
import numpy.linalg as la
from typing import Iterable, Union
from scipy.optimize import minimize, least_squares
from configobj import ConfigObj

from .core import *
from .output import *


class HubbardParamEqualizer(MLWF):

    def __init__(
            self,
            N,
            equalize=False,  # Homogenize trap or not
            eqtarget='uvt',  # Equalization target
            method: str = 'SLSQP',  # Minimize algorithm method
            waist='x',  # Waist to vary, None means no waist change
            random: bool = False,  # Random initial guess
            iofile=None,  # Input/output file
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        self.waist_dir = None
        self.eqinfo = {}

        if equalize:
            self.eq_label = eqtarget
            self.waist_dir = waist
            if self.lattice_dim > 1 and self.waist_dir != None \
                    and self.waist_dir != 'xy':
                self.waist_dir = 'xy'

            __, __, __, self.eqinfo = self.equalize(
                eqtarget, random=random, callback=False, method=method, iofile=iofile)

    def equalize(self,
                 target: str = 'uvt',
                 weight: np.ndarray = np.ones(3),
                 random: bool = False,
                 nobounds: bool = False,
                 method: str = 'SLSQP',
                 callback: bool = False,
                 iofile: ConfigObj = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        print(f"Varying waist direction: {self.waist_dir}.")
        print(f"Equalization method: {method}")
        print(f"Equalization target: {target}\n")
        u, t, v, fix_u, fix_t, fix_v = self.str_to_flags(target)

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
            xlinks, ylinks, txTarget, tyTarget = self.xy_links(nnt)
            if not fix_t:
                txTarget, tyTarget = None, None
        else:
            nnt, xlinks, ylinks, txTarget, tyTarget = None, None, None, None, None
        if fix_v:
            Vtarget = np.mean(np.real(np.diag(A)))
        else:
            Vtarget = None

        # Voff_bak = self.Voff
        # ls_bak = self.trap_centers
        # w_bak = self.waists
        v0, bounds = self.init_guess(random=random)
        if nobounds:
            bounds = None

        self.eqinfo = {'Nfeval': 0,
                       'cost': np.array([]).reshape(0, 3),
                       'ctot': np.array([]),
                       'fval': np.array([]),
                       'diff': np.array([]),
                       'x': np.array([]).reshape(0, *v0.shape)}

        # Decide if each step cost function used the last step's unitary matrix
        # callback can have sometimes very few iteraction steps
        # But since unitary optimize time cost is not large in larger systems
        # it is not recommended
        if callback:
            # Pack x0 to be mutable, thus can be updated in each iteration of minimize
            x0 = [V]
        else:
            x0 = None

        def cost_func(offset: np.ndarray, info: Union[dict, None]) -> float:
            c = self.cbd_cost_func(offset, info, (xlinks, ylinks),
                                   (Vtarget, Utarget, txTarget, tyTarget), (u, t, v), weight, x0, report=iofile)
            return c

        t0 = time()
        # Method-specific options
        if method == 'Nelder-Mead':
            options = {
                'disp': True, 'return_all': True, 'adaptive': False, 'xatol': 1e-7, 'fatol': 1e-9}
        elif method == 'SLSQP':
            options = {'disp': True, 'ftol': 1e-9}

        res = minimize(cost_func, v0, args=self.eqinfo,
                       bounds=bounds, method=method, options=options)
        t1 = time()
        print(f"Equalization took {t1 - t0} seconds.")

        self.eqinfo['termination_reason'] = res.message
        self.eqinfo['exit_status'] = res.status

        trap_depth = res.x[:self.Nindep]
        self.symm_unfold(self.Voff, trap_depth)

        if self.waist_dir != None:
            trap_waist_ratio = res.x[self.Nindep:3 *
                                     self.Nindep].reshape(self.Nindep, 2)
            self.symm_unfold(self.waists, trap_waist_ratio)

        trap_center = res.x[-2 * self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        return self.Voff, self.waists, self.trap_centers, self.eqinfo

    def str_to_flags(self, target: str) -> tuple[bool, bool, bool, bool, bool, bool]:
        u, t, v = False, False, False
        fix_u, fix_t, fix_v = False, False, False
        if 'u' in target or 'U' in target:
            u = True
            if 'U' in target:
                # Whether to fix target in combined cost function
                fix_u = True
        if 't' in target or 'T' in target:
            t = True
            if 'T' in target:
                fix_t = True
        if 'v' in target or 'V' in target:
            v = True
            if 'V' in target:
                fix_v = True
        return u, t, v, fix_u, fix_t, fix_v

    def init_guess(self, random=False) -> tuple[np.ndarray, tuple]:
        # Trap depth variation inital guess and bounds
        v01 = np.ones(self.Nindep)
        b1 = list((0.9, 1.1) for i in range(self.Nindep))

        # Waist variation inital guess and bounds
        if self.waist_dir == None:
            v02 = np.array([])
            b2 = []
        else:
            v02 = np.ones(2 * self.Nindep)
            if 'x' in self.waist_dir:
                b2x = list((0.9, 1.1) for i in range(self.Nindep))
            else:
                b2x = list((1, 1) for i in range(self.Nindep))
            if 'y' in self.waist_dir:
                b2y = list((0.9, 1.1) for i in range(self.Nindep))
            else:
                b2y = list((1, 1) for i in range(self.Nindep))
            n2 = tuple((b2x[i], b2y[i]) for i in range(self.Nindep))
            b2 = list(item for sublist in n2 for item in sublist)

        # Lattice spacing variation inital guess and bounds
        v03 = self.tc0[self.reflection[:, 0]]
        b3x = tuple((v03[i, 0] - 0.1, v03[i, 0] + 0.1) if self.inv_coords[i, 0]
                    == False else (0, 0) for i in range(self.Nindep))
        if self.lattice_dim == 1:
            b3y = tuple((0, 0) for i in range(self.Nindep))
        else:
            b3y = tuple((v03[i, 1] - 0.1, v03[i, 1] + 0.1)
                        if self.inv_coords[i, 1]
                        == False else (0, 0) for i in range(self.Nindep))
        n3 = tuple((b3x[i], b3y[i]) for i in range(self.Nindep))
        b3 = list(item for sublist in n3 for item in sublist)

        bounds = tuple(b1 + b2 + b3)

        if random:
            v0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        else:
            v0 = np.concatenate((v01, v02, v03.reshape(-1)))

        trap_depth = v0[:self.Nindep]
        if self.waist_dir != None:
            trap_waist = v0[self.Nindep:3 *
                            self.Nindep].reshape(self.Nindep, 2)
        trap_center = v0[-2 * self.Nindep:].reshape(self.Nindep, 2)

        if self.verbosity or random:
            print(f"Initial trap depths: {trap_depth}")
            if self.waist_dir != None:
                print(f"Initial waists:")
                print(trap_waist)
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
                      unitary: Union[list, None] = None,
                      report: ConfigObj = None) -> float:

        trap_depth = offset[:self.Nindep]
        self.symm_unfold(self.Voff, trap_depth)

        if self.waist_dir != None:
            trap_waist = offset[self.Nindep:3 *
                                self.Nindep].reshape(self.Nindep, 2)
            self.symm_unfold(self.waists, trap_waist)

        trap_center = offset[- 2 * self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        if self.verbosity:
            print(f"\nCurrent trap depths: {trap_depth}")
            if self.waist_dir != None:
                print(f"Current waists:")
                print(trap_waist)
            print("Current trap centers:")
            print(trap_center)

        if unitary != None and self.lattice_dim > 1:
            x0 = unitary[0]
        else:
            x0 = None

        u, t, v = utv

        # A, U, x0 = self.singleband_Hubbard(
        #     u=True, x0=x0, output_unitary=True)
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
            if info['Nfeval'] % 10 == 0:
                if isinstance(report, ConfigObj):
                    write_equalize_log(report, info, final=False)
                    write_trap_params(report, self)
                    write_singleband(report, self)
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


# ================ TEST LEAST_SQUARE =====================

    def equalize_lsq(self,
                     target: str = 'uvt',
                     waists: str = None,
                     weight: np.ndarray = np.ones(3),
                     random: bool = False,
                     nobounds: bool = False,
                     method: str = 'trf',
                     callback: bool = False,
                     iofile: ConfigObj = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        print(f"Varying waist direction: {self.waist_dir}.")
        print(f"Equalization method: {method}")
        print(f"Equalization target: {target}\n")
        u, t, v, fix_u, fix_t, fix_v = self.str_to_flags(target)

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
            xlinks, ylinks, txTarget, tyTarget = self.xy_links(nnt)
            if not fix_t:
                txTarget, tyTarget = None, None
        else:
            nnt, xlinks, ylinks, txTarget, tyTarget = None, None, None, None, None
        if fix_v:
            Vtarget = np.mean(np.real(np.diag(A)))
        else:
            Vtarget = None

        # Voff_bak = self.Voff
        # ls_bak = self.trap_centers
        # w_bak = self.waists
        v0, bounds = self.init_guess(random=random)
        if nobounds:
            bounds = None

        self.eqinfo = {'Nfeval': 0,
                       'cost': np.array([]).reshape(0, 3),
                       'ctot': np.array([]),
                       'fval': np.array([]),
                       'diff': np.array([]),
                       'x': np.array([]).reshape(0, *v0.shape)}

        # Decide if each step cost function used the last step's unitary matrix
        # callback can have sometimes very few iteraction steps
        # But since unitary optimize time cost is not large in larger systems
        # it is not recommended
        if callback:
            # Pack x0 to be mutable, thus can be updated in each iteration of minimize
            x0 = [V]
        else:
            x0 = None

        def res_func(offset: np.ndarray, info: Union[dict, None]):
            c = self.cbd_res_func(offset, info, (xlinks, ylinks),
                                  (Vtarget, Utarget, txTarget, tyTarget), (u, t, v), weight, x0, report=iofile)
            return c

        def rho(offset: np.ndarray):
            sqrt = np.sqrt(offset)
            r = np.array([sqrt, 1/(2*sqrt), -1/(4*sqrt**3)])
            return r

        t0 = time()
        res = least_squares(res_func, v0, loss=rho, args=(self.eqinfo,),
                            method=method, verbose=2,
                            xtol=np.finfo(float).eps, ftol=np.finfo(float).eps, gtol=np.finfo(float).eps)
        t1 = time()
        print(f"Equalization took {t1 - t0} seconds.")

        self.eqinfo['termination_reason'] = res.message
        self.eqinfo['exit_status'] = res.status

        trap_depth = res.x[:self.Nindep]
        self.symm_unfold(self.Voff, trap_depth)

        if self.waist_dir != None:
            trap_waist_ratio = res.x[self.Nindep:3 *
                                     self.Nindep].reshape(self.Nindep, 2)
            self.symm_unfold(self.waists, trap_waist_ratio)

        trap_center = res.x[-2 * self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        return self.Voff, self.waists, self.trap_centers, self.eqinfo

    def cbd_res_func(self,
                     offset: np.ndarray,
                     info: Union[dict, None],
                     links: tuple[np.ndarray, np.ndarray],
                     target: tuple[float, ...],
                     utv: tuple[bool] = (False, False, False),
                     weight: np.ndarray = np.ones(3),
                     unitary: Union[list, None] = None,
                     report: ConfigObj = None) -> float:

        trap_depth = offset[:self.Nindep]
        self.symm_unfold(self.Voff, trap_depth)

        if self.waist_dir != None:
            trap_waist = offset[self.Nindep:3 *
                                self.Nindep].reshape(self.Nindep, 2)
            self.symm_unfold(self.waists, trap_waist)

        trap_center = offset[- 2 * self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        if self.verbosity:
            print(f"\nCurrent trap depths: {trap_depth}")
            if self.waist_dir != None:
                print(f"Current waists:")
                print(trap_waist)
            print("Current trap centers:")
            print(trap_center)

        if unitary != None and self.lattice_dim > 1:
            x0 = unitary[0]
        else:
            x0 = None

        u, t, v = utv

        # A, U, x0 = self.singleband_Hubbard(
        #     u=True, x0=x0, output_unitary=True)
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
        cu = np.zeros(U.shape)
        if u:
            # U is different, as calculating U costs time
            cu = self.u_res_func(U, Utarget)

        ct = self.t_res_func(A, (xlinks, ylinks), (nntx, nnty))
        if not t:
            # Force t to have no effect on cost function
            w[1] = 0

        cv = self.v_res_func(A, Vtarget)
        if not v:
            # Force V to have no effect on cost function
            w[2] = 0

        cvec = np.array([la.norm(cu), la.norm(ct), la.norm(cv)])
        cw = [w[0] * cu, w[1] * ct, w[2] * cv]
        c = np.concatenate(cw)
        if self.verbosity:
            print(f"Current total distance: {c}\n")

        # Keep revcord
        if info != None:
            info['Nfeval'] += 1
            info['x'] = np.append(info['x'], offset[None], axis=0)
            info['cost'] = np.append(info['cost'], cvec[None], axis=0)
            ctot = np.sum(cvec)
            info['ctot'] = np.append(info['ctot'], ctot)
            fval = la.norm(c)
            info['fval'] = np.append(info['fval'], fval)
            diff = info['fval'][len(info['fval'])//2] - fval
            info['diff'] = np.append(info['diff'], diff)
            # display information
            if info['Nfeval'] % 10 == 0:
                if isinstance(report, ConfigObj):
                    write_equalize_log(report, info, final=False)
                    write_trap_params(report, self)
                    write_singleband(report, self)
                if self.verbosity:
                    print(
                        f'i={info["Nfeval"]}\tc={cvec}\tc_i={c}\tc_i//2-c_i={diff}')

        return c

    def v_res_func(self, A, Vtarget):
        if Vtarget is None:
            Vtarget = np.mean(np.real(np.diag(A)))
        cv = (np.real(np.diag(A)) - Vtarget) / \
            abs(Vtarget * np.sqrt(len(A)))
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Onsite potential target={Vtarget}')
            print(f'Onsite potential normalized distance v={cv}')
        return cv

    def t_res_func(self, A: np.ndarray, links: tuple[np.ndarray, np.ndarray],
                   target: tuple[float, ...]):
        nnt = self.nn_tunneling(A)
        if target is None:
            xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
        elif isinstance(target, Iterable):
            xlinks, ylinks = links
            nntx, nnty = target
            if nntx is None:
                xlinks, ylinks, nntx, nnty = self.xy_links(nnt)

        ct = (abs(nnt[xlinks]) - nntx) / (nntx * np.sqrt(len(xlinks)))
        if nnty != None:
            ct = np.concatenate(
                (ct, (abs(nnt[ylinks]) - nnty) / (nnty * np.sqrt(len(ylinks)))))
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Tunneling target=({nntx}, {nnty})')
            print(f'Tunneling normalized distance t={ct}')
        return ct

    def u_res_func(self, U, Utarget):
        if Utarget is None:
            Utarget = np.mean(U)
        cu = (U - Utarget) / abs(Utarget * np.sqrt(len(U)))
        if self.verbosity:
            if self.verbosity > 1:
                print(f'Onsite interaction target fixed to {Utarget}')
            print(f'Onsite interaction normalized distance u={cu}')
        return cu

# ================ TEST OVER =====================
