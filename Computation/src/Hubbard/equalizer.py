import numpy as np
import numpy.linalg as la
from typing import Callable, Iterable, Union
from opt_einsum import contract
from pyparsing import Char
from scipy.integrate import romb
from scipy.optimize import minimize, shgo

# from mystic.solvers import DifferentialEvolutionSolver
# from mystic.termination import ChangeOverGeneration, VTR
# from mystic.strategy import Best1Exp, Best1Bin, Rand1Exp
# from mystic.monitors import VerboseMonitor
# from mystic.bounds import Bounds

from .core import *


class HubbardParamEqualizer(MLWF):

    def __init__(
            self,
            N,
            equalize=False,  # Homogenize trap or not
            eqtarget='uvt',  # Equalization target
            waist='x',  # Waist to vary, None means no waist change
            *args,
            **kwargs):
        super().__init__(N, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = 'neq'
        self.waist_dir = waist
        if self.lattice_dim > 1 and self.waist_dir != None \
                and self.waist_dir != 'xy':
            self.waist_dir = 'xy'
        if equalize:
            self.eq_label = eqtarget
            # self.homogenize(eqtarget, fixed)
            self.equalzie(eqtarget, random=True, callback=True)

    def equalzie(self,
                 target: str = 'uvt',
                 weight: np.ndarray = np.ones(3),
                 random: bool = False,
                 nobounds: bool = False,
                 callback: bool = False):
        if self.verbosity:
            print(f"Equalizing {target}.")
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

        # Decide if each step cost function used the last step's unitary matrix
        # callback can have sometimes very few iteraction steps
        if callback:
            # Pack x0 to be mutable, thus can be updated in each iteration of minimize
            x0 = [V]
        else:
            x0 = None

        def cost_func(offset: np.ndarray, info: Union[dict, None]) -> float:
            c = self.cbd_cost_func(offset, info, (xlinks, ylinks),
                                   (Vtarget, Utarget, txTarget, tyTarget), (u, t, v), weight, x0)

            return c

        info = {'Nfeval': 0,
                'cost': np.array([]).reshape(0, 3),
                'ctot': np.array([]),
                'fval': np.array([]),
                'diff': np.array([]),
                'x': np.array([]).reshape(0, *v0.shape)}

        t0 = time()
        if nobounds:
            res = minimize(cost_func, v0, args=info)
        else:
            # res = minimize(cost_func, v0, args=info, bounds=bounds, method='Nelder-Mead', options={
            #                'disp': True, 'return_all': True, 'adaptive': False, 'xatol': 1e-8, 'fatol': 1e-10})
            res = minimize(cost_func, v0, args=info, bounds=bounds, method='SLSQP', options={
                           'disp': True, 'ftol': 1e-9})
        t1 = time()
        if self.verbosity:
            print(f"Equalization took {t1 - t0} seconds.")

        trap_depth = res.x[:self.Nindep]
        self.symm_unfold(self.Voff, trap_depth)

        if self.waist_dir != None:
            trap_waist_ratio = res.x[self.Nindep:3 *
                                     self.Nindep].reshape(self.Nindep, 2)
            self.symm_unfold(self.waists, trap_waist_ratio)

        trap_center = res.x[-2 * self.Nindep:].reshape(self.Nindep, 2)
        self.symm_unfold(self.trap_centers, trap_center, graph=True)
        self.update_lattice(self.trap_centers)

        return self.Voff, self.waists, self.trap_centers, info

# # ================ TEST MYSTIC =====================

#     def equalzie_mystic(self,
#                         target: str = 'UT',
#                         weight: np.ndarray = np.ones(3),
#                         random: bool = False,
#                         callback: bool = False):
#         if self.verbosity:
#             print(f"Equalizing {target}.")
#         u, t, v, fix_u, fix_t, fix_v = self.str_to_flags(target)

#         res = self.singleband_Hubbard(u=u, output_unitary=True)
#         if u:
#             A, U, V = res
#         else:
#             A, V = res
#             U = None

#         if fix_u:
#             Utarget = np.mean(U)
#         else:
#             Utarget = None
#         if t:
#             nnt = self.nn_tunneling(A)
#             xlinks, ylinks, txTarget, tyTarget = self.xy_links(nnt)
#             if not fix_t:
#                 txTarget, tyTarget = None, None
#         else:
#             nnt, xlinks, ylinks, txTarget, tyTarget = None, None, None, None, None
#         if fix_v:
#             Vtarget = np.mean(np.real(np.diag(A)))
#         else:
#             Vtarget = None

#         # Voff_bak = self.Voff
#         # ls_bak = self.trap_centers
#         # w_bak = self.waists
#         v0, bounds = self.init_guess(random=random)
#         bounds = Bounds(tuple(i[0] for i in bounds),
#                         tuple(i[1] for i in bounds))

#         # Decide if each step cost function used the last step's unitary matrix
#         # callback can have sometimes very few iteraction steps
#         if callback:
#             # Pack x0 to be mutable, thus can be updated in each iteration of minimize
#             x0 = [V]
#         else:
#             x0 = None

#         def cost_func(offset: np.ndarray, info: Union[dict, None]) -> float:
#             if not isinstance(offset, np.ndarray):
#                 offset = np.array(offset)
#             c = self.cbd_cost_func(offset, info, (xlinks, ylinks),
#                                    (Vtarget, Utarget, txTarget, tyTarget), (u, t, v), weight, x0)

#             return c

#         info = {'Nfeval': 0,
#                 'cost': np.array([]).reshape(0, 3),
#                 'ctot': np.array([]),
#                 'fval': np.array([]),
#                 'diff': np.array([]),
#                 'x': np.array([]).reshape(0, *v0.shape)}

#         t0 = time()

#         ND = v0.shape[0]
#         NP = 10 * ND
#         MAX_GENERATIONS = ND * NP

#         solver = DifferentialEvolutionSolver(ND, NP)
#         solver.SetRandomInitialPoints(min=[0]*ND, max=[2]*ND)
#         solver.SetEvaluationLimits(generations=MAX_GENERATIONS)
#         solver.SetGenerationMonitor(VerboseMonitor(30))
#         strategy = Best1Exp

#         solver.Solve(cost_func, ExtraArgs=(info,), bounds=bounds, termination=VTR(0.01), strategy=strategy,
#                      CrossProbability=0.9, ScalingFactor=0.8)

#         res = solver.Solution()

#         t1 = time()
#         if self.verbosity:
#             print(f"Equalization took {t1 - t0} seconds.")

#         # trap_depth = res.x[:self.Nindep]
#         # trap_waist_ratio = res.x[self.Nindep:3 *
#         #                          self.Nindep].reshape(self.Nindep, 2)
#         # trap_center = res.x[3 * self.Nindep:].reshape(self.Nindep, 2)
#         # self.symm_unfold(self.Voff, trap_depth)
#         # self.symm_unfold(self.waists, trap_waist_ratio)
#         # self.symm_unfold(self.trap_centers, trap_center, graph=True)
#         # self.update_lattice(self.trap_centers)
#         # return self.Voff, self.waists, self.trap_centers, info
#         return res, info

# # ================ TEST OVER =====================

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
                      unitary: Union[list, None] = None) -> float:

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

        A, U, x0 = self.singleband_Hubbard(
            u=True, x0=x0, output_unitary=True)
        # res = self.singleband_Hubbard(
        #     u=u, x0=x0, output_unitary=True)
        # if u:
        #     A, U, x0 = res
        # else:
        #     A, x0 = res
        #     U = None

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
        # cu = 0
        # if u:
        #     # U is different, as calculating U costs time
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


# # ===================== TO BE DEPRECATED =====================================

#     def homogenize(self, target: str = 'vt', fixed=False):
#         # Force target to be 2-character string
#         if len(target) == 1:
#             if target == 't' or target == 'T':
#                 # Tunneling is varying spacings in default
#                 target = '0' + target
#             else:
#                 # Other is varying trap depths in default
#                 target = target + '0'

#         cost_func, quantity = self.one_equzlize(target[0], fixed)
#         self.Voff = self.depth_equalize(cost_func)
#         if quantity != None:
#             print(f'{quantity} homogenized by trap depths.\n')

#         cost_func, quantity = self.one_equzlize(target[1], fixed)
#         self.trap_centers = self.spacing_equalize(cost_func)
#         if quantity != None:
#             print(f'{quantity} homogenized by trap spacings.\n')

#         return self.Voff, self.trap_centers

#     def one_equzlize(self, target: str, fixed=False):
#         if 'v' in target:
#             cost_func = self.v_equalize(u=False)
#             quantity = 'Onsite potential'
#         elif 'V' in target:
#             # Combined cost function for U and V is used
#             cost_func = self.v_equalize(u=True, fixed=fixed)
#             quantity = 'Onsite potential combining interaction'
#         elif 'u' in target:
#             cost_func = self.u_equalize()
#             quantity = 'Onsite interaction'
#         elif 't' in target:
#             cost_func = self.t_equalize(v=False)
#             quantity = 'Tunneling'
#         elif 'T' in target:
#             # Combined cost function for t and V is used
#             cost_func = self.t_equalize(v=True, fixed=fixed)
#             quantity = 'Tunneling combining onsite potential'
#         else:
#             cost_func = None
#             quantity = None
#             print('Input target not recognized.')
#         return cost_func, quantity

#     def v_equalize(self, u, fixed=False) -> Callable[[np.ndarray], float]:
#         res = self.singleband_Hubbard(u)
#         if u:
#             A, U = res
#         else:
#             A = res
#         if fixed:
#             Utarget = np.mean(U)
#         else:
#             Utarget = None
#         Vtarget = np.mean(np.real(np.diag(A)))

#         def cost_func(offset: np.ndarray, offset_type) -> float:
#             # If target = None, then U and V are targeted to mean values
#             # If target is given, for V it's float value, for U and V it's a tuple
#             if offset_type == 'd':
#                 self.symm_unfold(self.Voff, offset)
#                 print("\nCurrent trap depths:", offset)
#             elif offset_type == 's':
#                 offset = offset.reshape(self.Nindep, 2)
#                 self.symm_unfold(self.trap_centers, offset, graph=True)
#                 self.update_lattice(self.trap_centers)
#                 print("\nCurrent trap centers:", offset)

#             res = self.singleband_Hubbard(u)
#             if u:
#                 A, U = res
#             else:
#                 A = res

#             c = self.v_cost_func(A, Vtarget)
#             if u:
#                 c += self.u_cost_func(U, Utarget)
#             print("Current total cost:", c, "\n")
#             return c

#         return cost_func

#     def u_equalize(self) -> Callable[[np.ndarray], float]:
#         # Equalize onsite chemical potential
#         A, U = self.singleband_Hubbard(u=True)
#         Utarget = np.mean(U)

#         def cost_func(offset: np.ndarray, offset_type) -> float:
#             if offset_type == 'd':
#                 self.symm_unfold(self.Voff, offset)
#                 print("\nCurrent trap depths:", offset)
#             elif offset_type == 's':
#                 offset = offset.reshape(self.Nindep, 2)
#                 self.symm_unfold(self.trap_centers, offset, graph=True)
#                 self.update_lattice(self.trap_centers)
#                 print("\nCurrent trap centers:", offset)

#             A, U = self.singleband_Hubbard(u=True)
#             c = self.u_cost_func(U, Utarget)
#             print("Current total cost:", c, "\n")
#             return c

#         return cost_func

#     def depth_equalize(self, cost_func) -> np.ndarray:
#         # Equalize onsite chemical potential

#         if cost_func != None:
#             Voff_bak = self.Voff

#             v0 = np.ones(self.Nindep)
#             # Bound trap depth variation
#             bonds = tuple((0.9, 1.1) for i in range(self.Nindep))
#             res = minimize(cost_func, v0, 'd', bounds=bonds)
#             self.symm_unfold(self.Voff, res.x)
#         return self.Voff

#     def t_equalize(self, v, fixed=False) -> Callable[[np.ndarray], float]:
#         A = self.singleband_Hubbard()
#         nnt = self.nn_tunneling(A)
#         xlinks, ylinks, nntx, nnty = self.xy_links(nnt)
#         Vtarget = None
#         if fixed:
#             Vtarget = np.mean(np.real(np.diag(A)))

#         def cost_func(offset: np.ndarray, offset_type) -> float:
#             if offset_type == 'd':
#                 self.symm_unfold(self.Voff, offset)
#                 print("\nCurrent trap depths:", offset)
#             elif offset_type == 's':
#                 offset = offset.reshape(self.Nindep, 2)
#                 self.symm_unfold(self.trap_centers, offset, graph=True)
#                 self.update_lattice(self.trap_centers)
#                 print("\nCurrent trap centers:", offset)

#             A = self.singleband_Hubbard()
#             c = self.t_cost_func(A, (xlinks, ylinks), (nntx, nnty))
#             if v:
#                 c += self.v_cost_func(A, Vtarget)
#             print("Current total cost:", c, "\n")
#             return c

#         return cost_func

#     def spacing_equalize(self, cost_func) -> np.ndarray:
#         # Equalize tunneling
#         if cost_func != None:
#             ls_bak = self.trap_centers

#             v0 = self.trap_centers[self.reflection[:, 0]]
#             # print('v0', v0)
#             # Bound lattice spacing variation
#             xbonds = tuple(
#                 (v0[i, 0] - 0.05, v0[i, 0] + 0.05) for i in range(self.Nindep))
#             if self.lattice_dim == 1:
#                 ybonds = tuple((0, 0) for i in range(self.Nindep))
#             else:
#                 ybonds = tuple((v0[i, 1] - 0.05, v0[i, 1] + 0.05)
#                                for i in range(self.Nindep))
#             nested = tuple((xbonds[i], ybonds[i]) for i in range(self.Nindep))
#             bonds = tuple(item for sublist in nested for item in sublist)
#             # print('bounds', bonds)
#             res = minimize(cost_func, v0.reshape(-1), 's', bounds=bonds)
#             self.symm_unfold(self.trap_centers,
#                              res.x.reshape(self.Nindep, 2),
#                              graph=True)
#             self.update_lattice(self.trap_centers)
#         return self.trap_centers
