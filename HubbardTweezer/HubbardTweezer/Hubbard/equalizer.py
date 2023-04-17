import numpy as np
import numpy.linalg as la
from numbers import Number
from typing import Callable, Iterable, Union
from scipy.optimize import minimize, least_squares, root, OptimizeResult
from configobj import ConfigObj
from time import time

import nlopt

from .core import *
from .io import *
from .ghost import GhostTrap


def str_to_flags(target: str) -> tuple[bool, bool, bool, bool, bool, bool]:
    u, t, v = False, False, False
    fix_u, fix_t, fix_v = False, False, False
    if "u" in target or "U" in target:
        u = True
        if "U" in target:
            # Whether to fix target in combined cost function
            fix_u = True
    if "t" in target or "T" in target:
        t = True
        if "T" in target:
            fix_t = True
    if "v" in target or "V" in target:
        v = True
        if "V" in target:
            fix_v = True
    return u, t, v, fix_u, fix_t, fix_v


def _set_uv(uv, target, factor):
    if target is None:
        target = np.mean(uv)
    if factor is None:
        # Avoid division by zero
        factor = abs(target)
        if factor < 1e-1:
            factor = 1e-1
    return target, factor


class HubbardEqualizer(MLWF):
    """
    HubbardEqualizer: equalize trap parameters to generate Hubbard model parameters in fermionic tweezer array

    Args:
    -----------------

    """

    def __init__(
        self,
        N,
        equalize=False,  # Homogenize trap or not
        eqtarget="UvT",  # Equalization target
        scale_factor=None,  # Scale factor for cost function
        Ut: float = None,  # Interaction target in unit of tx
        eqmethod: str = None,  # Minimize algorithm method
        nobounds: bool = False,  # Whether to use bounds or not
        waist="x",  # Waist to vary, None means no waist change
        ghost: bool = False,  # Whether to use ghost atoms or not
        ghost_penalty=(0, 0),  # Ghost threshold & penalty weight
        random: bool = False,  # Random initial guess
        iofile=None,  # Input/output file
        write_log: bool = False,  # Whether to write detailed log into iofile
        x0: np.ndarray = None,  # Initial value for minimization to start from
        *args,
        **kwargs,
    ):
        # Set ghost lattice shape
        lattice_shape = kwargs.get("shape", "square")
        kwargs.pop("shape", None)
        ghost_shape = lattice_shape
        if ghost and lattice_shape == "Lieb":
            # If use ghost traps,
            # Lieb lattice has ghost sites in the interior
            # But its shape is square
            print("Equalize: Lieb lattice ghost sites.")
            print("Set shape to square for total system.")
            lattice_shape = "square"
        super().__init__(N, shape=lattice_shape, *args, **kwargs)

        # set equalization label in file output
        self.eq_label = eqtarget
        self.waist_dir = waist
        self.eqinfo = EqulizeInfo()
        if eqmethod is None:
            self.eqmethod = "Nelder-Mead" if waist == None else "trf"
        else:
            self.eqmethod = "Nelder-Mead" if eqmethod == "NM" else eqmethod
        self.log = False if not equalize else write_log
        if isinstance(scale_factor, Number):
            self.sf = scale_factor
        else:
            print("Equalize: scale_factor is not a number. Set to None.")
            self.sf = None

        # Set target to be already limited in the bulk
        self.ghost = GhostTrap(self.lattice, ghost_shape, *ghost_penalty)
        if ghost:
            self.ghost.set_mask(self.lattice)

        if self.ghost.Nsite == 1:
            raise ValueError(
                "Equalize: only one site in the system, equalization is not valid."
            )

        if equalize:
            if (
                self.lattice.dim > 1
                and self.waist_dir != None
                and self.waist_dir != "xy"
            ):
                self.waist_dir = "xy"

            if not isinstance(x0, np.ndarray):
                print("Illegal x0 provided. Use no initial guess.")
                x0 = None

            eig_callback = kwargs.get("eig_callback", True)
            if eig_callback:
                print("Equalize: eig_callback is True.")
            unitary_callback = kwargs.get("unitary_callback", False)

            self.equalize(
                target=eqtarget,
                Ut=Ut,
                x0=x0,
                random=random,
                nobounds=nobounds,
                eig_callback=eig_callback,
                unitary_callback=unitary_callback,
                iofile=iofile,
            )

    def equalize(
        self,
        target: str = "UvT",
        Ut: float = None,  # Target onsite interaction in unit of tx
        x0: np.ndarray = None,
        weight: np.ndarray = np.ones(3),
        random: bool = False,
        nobounds: bool = False,
        eig_callback: bool = False,
        unitary_callback: bool = False,
        iofile: ConfigObj = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        print(f"Equalize: varying waist direction = {self.waist_dir}.")
        print(f"Equalize: method = {self.eqmethod}")
        print(f"Equalize: quantities = {target}\n")
        u, t, v, fix_u, fix_t, fix_v = str_to_flags(target)
        # Force corresponding factor to be 0 if flags u,t,v are false
        weight: np.ndarray = np.array([u, t, v]) * np.array(weight.copy())

        if eig_callback:
            W0 = []
        else:
            W0 = None

        A, U, V = self.singleband_Hubbard(u=u, W0=W0, offset=True)

        maskedA = self.ghost.mask_quantity(A)
        maskedU = self.ghost.mask_quantity(U) if u else None
        links = self.xy_links(self.ghost.links)
        target = self._set_targets(Ut, fix_u, fix_t, links, maskedA, maskedU)

        v0, bounds = self.initialize(random, nobounds)
        init_simplx = None

        if isinstance(x0, np.ndarray):
            try:
                if self.eqmethod == "Nelder-Mead" and x0.shape == (
                    len(v0) + 1,
                    len(v0),
                ):
                    v0 = x0[0]
                    init_simplx = x0
                    print("Equalize: external initial simplex is passed to NM.")
                elif len(x0) == len(v0):
                    v0 = x0  # Use passed initial guess
                    print("Equalize: external initial guess is passed.")
            except:  # x0 is None or other cases
                print("Equalize: external initial guess is not passed.")
                pass

        self.eqinfo.create_log(v0, target)

        # Decide if each step cost function used the last step's unitary matrix
        # callback can have sometimes very few iteraction steps
        # But since unitary optimize time cost is not large in larger systems
        # it is not recommended
        # Pack U0 to be mutable, thus can be updated in each iteration of minimize
        U0 = [V] if unitary_callback else None

        if self.eqmethod in ["trf", "dogbox"]:
            mode = "res"
        elif self.eqmethod in [
            "Nelder-Mead",
            "Powell",
            "bfgs",
            "L-BFGS-B",
            "cobyla",
            "SLSQP",
        ]:
            mode = "cost"
        elif self.eqmethod in ["hybr"]:
            mode = "func"
        elif self.eqmethod in ["bobyqa", "praxis", "subplex", "direct", "crs2"]:
            mode = "nlopt"
            goptim = False
            if self.eqmethod == "bobyqa":
                self.eqmethod = nlopt.LN_BOBYQA
            elif self.eqmethod == "praxis":
                self.eqmethod = nlopt.LN_PRAXIS
            elif self.eqmethod == "subplex":
                self.eqmethod = nlopt.LN_SBPLX
            elif self.eqmethod == "direct":
                goptim = True
                self.eqmethod = nlopt.GN_DIRECT_L
            elif self.eqmethod == "crs2":
                goptim = True
                self.eqmethod = nlopt.GN_CRS2_LM
            else:
                self.eqmethod = nlopt.LN_COBYLA
            opt = nlopt.opt(self.eqmethod, len(v0))
        else:
            mode = "res"
            self.eqmethod = "trf"
            print(
                f"Equalize WARNING: unknown optimization method: {self.eqmethod}. Set to trf."
            )

        def opt_target(point: np.ndarray, info: Union[EqulizeInfo, None]):
            return self.opt_func(
                point,
                info,
                links,
                target,
                weight,
                self.sf,
                W0,
                unitary=U0,
                mode=mode,
                report=iofile,
            )

        t0 = time()
        if mode == "res":
            res = self._min_res_mode(v0, bounds, opt_target)
        elif mode == "cost":
            res = self._min_cost_mode(v0, bounds, init_simplx, opt_target)
        elif mode == "func":
            # FIXME: not working if variable number of unknowns
            # are smaller than number of equations
            res = root(opt_target, v0, args=self.eqinfo, method=self.eqmethod, tol=1e-7)
        elif mode == "nlopt":
            xopt = self._nlopt_min(v0, bounds, opt, opt_target)
            if goptim:
                self.eqmethod = "trf"

                def _res_target(point: np.ndarray, info: Union[EqulizeInfo, None]):
                    return self.opt_func(
                        point,
                        info,
                        links,
                        target,
                        weight,
                        self.sf,
                        W0,
                        unitary=U0,
                        mode="res",
                        report=iofile,
                    )

                res = self._min_res_mode(xopt, bounds, _res_target)
            else:
                res = self._nlopt_ressult(opt, xopt)

        t1 = time()
        print(f"Equalization took {t1 - t0} seconds.")

        self.eqinfo.update_log_final(res, self.sf)
        return self.param_unfold(res.x, "final")

    def _nlopt_min(self, v0, bounds, opt: nlopt.opt, opt_target):
        ba = np.array(bounds)
        lb, ub = ba[:, 0], ba[:, 1]
        tol = 1e-8
        f = lambda x, grad: opt_target(x, self.eqinfo)
        opt.set_min_objective(f)
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_ftol_abs(tol)
        xopt = opt.optimize(v0)
        return xopt

    def _nlopt_ressult(self, opt: nlopt.opt, xopt):
        opt_val = opt.last_optimum_value()
        result = opt.last_optimize_result()
        message = opt.get_stopval()
        res = OptimizeResult(
            x=xopt, fun=opt_val, status=result, success=result >= 0, message=message
        )
        return res

    def _min_cost_mode(self, v0, bounds, init_simplx, opt_target):
        # Method-specific options
        if self.eqmethod == "Nelder-Mead":
            adp = self.lattice.Nindep > 3
            options = {
                "disp": True,
                "initial_simplex": init_simplx,
                "adaptive": adp,
                "xatol": 1e-6,
                "fatol": 1e-7,
                "maxiter": 500 * self.lattice.Nindep,
            }
        elif self.eqmethod == "SLSQP":
            options = {
                "disp": True,
                "ftol": 1e-7,
                "nfev": 500 * self.lattice.Nindep,
            }
        elif self.eqmethod == "Powell":
            options = {
                "disp": True,
                "xtol": 1e-6,
                "maxiter": 500 * self.lattice.Nindep,
            }
        else:
            options = {
                "disp": True,
                "fatol": 1e-7,
                "maxiter": 500 * self.lattice.Nindep,
            }
        res = minimize(
            opt_target,
            v0,
            bounds=bounds,
            args=self.eqinfo,
            method=self.eqmethod,
            options=options,
        )
        return res

    def _min_res_mode(self, v0, bounds, opt_target):
        # Convert to tuple of (lb, ub)
        ba = np.array(bounds)
        bounds = (ba[:, 0], ba[:, 1])
        res = least_squares(
            opt_target,
            v0,
            bounds=bounds,
            args=(self.eqinfo,),
            method=self.eqmethod,
            verbose=2,
            xtol=1e-6,
            ftol=1e-7,
            gtol=1e-7,
            max_nfev=500 * self.lattice.Nindep,
        )
        return res

    def _set_targets(self, Ut, fix_u, fix_t, links, A, U):
        nnt = self.nn_tunneling(A)
        # Set tx, ty target to be small s.t.
        # lattice spacing is not too close and WF collapses
        txTarget, tyTarget = self.txy_target(nnt, links, np.min)
        # Energy scale factor, set to be of avg initial tx
        if not isinstance(self.sf, Number):
            self.sf = np.min([txTarget, tyTarget]) if tyTarget != None else txTarget
        if not fix_t:
            txTarget, tyTarget = None, None

        if fix_u:
            if Ut is None:
                # Set target interaction to be max of initial interaction.
                # This is to make traps not that localized in the middle to equalize U.
                # As to achieve larger U traps depths seem more even.
                Utarget = np.max(U)
                Ut = Utarget / self.sf
            else:
                Utarget = Ut * self.sf
        else:
            Utarget = None

        # If V is shifted to zero then fix_v has no effect
        # Vtarget = np.mean(np.real(np.diag(A))) if fix_v else None
        Vtarget = 0

        print(f"Equalize: scale factor = {self.sf}")
        print(f"Equalize: target tunneling = {txTarget}, {tyTarget}")
        print(f"Equalize: target interaction = {Utarget}")
        print(f"Equalize: target onsite potential = {Vtarget}")
        return Vtarget, Utarget, txTarget, tyTarget

    def xy_links(self, links=None):
        # Distinguish x and y n.n. bonds and target t_x t_y values
        # FIXME: Check all possible cases
        if links is None:
            links = self.lattice.links
        if not self.isotropic and self.lattice.shape in [
            "square",
            "Lieb",
            "triangular",
            "honeycomb",
            "kagome",
        ]:
            xlinks = abs(links[:, 0] - links[:, 1]) == 1
        else:
            xlinks = np.tile(True, links.shape[0])
        ylinks = np.logical_not(xlinks)
        return xlinks, ylinks

    def nn_tunneling(self, A: np.ndarray):
        # Pick up nearest neighbor tunnelings
        # Not limited to specific geometry
        if self.lattice.N == 1:
            nnt = np.zeros(1)
        elif self.lattice.dim == 1:
            nnt = np.diag(A, k=1)
        else:
            nnt = A[self.ghost.links[:, 0], self.ghost.links[:, 1]]
        return nnt

    def eff_dof(self):
        # Record all free DoFs in the function
        self.Voff_dof = np.ones(self.lattice.Nindep).astype(bool)

        if self.waist_dir == None:
            self.w_dof = None
        else:
            wx = np.tile("x" in self.waist_dir, self.lattice.Nindep)
            wy = np.tile("y" in self.waist_dir, self.lattice.Nindep)
            self.w_dof = np.array([wx, wy]).T.reshape(-1)

        tcx = np.array(
            [not self.lattice.inv_coords[i, 0] for i in range(self.lattice.Nindep)]
        )
        if self.lattice.dim == 1:
            tcy = np.tile(False, self.lattice.Nindep)
        else:
            tcy = np.array(
                [not self.lattice.inv_coords[i, 1] for i in range(self.lattice.Nindep)]
            )
        self.tc_dof = np.array([tcx, tcy]).T.reshape(-1)

        return self.Voff_dof, self.w_dof, self.tc_dof

    def initialize(self, random=False, nobounds=False) -> tuple[np.ndarray, tuple]:
        # Mark effective DoFs
        self.eff_dof()

        # Trap depth variation inital guess and bounds
        # s1 = np.inf if nobounds else 0.1
        # v01 = np.ones(self.lattice.Nindep)
        v01 = symm_fold(self.lattice.reflect, self.Voff)
        if nobounds:
            b1 = list((-np.inf, np.inf) for i in range(self.lattice.Nindep))
        else:
            b1 = list((0, 2) for i in range(self.lattice.Nindep))

        # Waist variation inital guess and bounds
        # UB from resolution limit; LB by wavelength
        if nobounds:
            s2 = (-np.inf, np.inf)
        else:
            s2 = (self.l / self.w, 1.2)
        if self.waist_dir == None:
            v02 = np.array([])
            b2 = []
        else:
            # v02 = np.ones(2 * self.lattice.Nindep)
            v02 = symm_fold(self.lattice.reflect, self.waists).flatten()
            b2 = list(s2 for i in range(2 * self.lattice.Nindep) if self.w_dof[i])
            v02 = v02[self.w_dof]

        # Lattice spacing variation inital guess and bounds
        # Must be separated by at least 1 waist
        if nobounds:
            s3 = (-np.inf, np.inf)
        else:
            s3 = (1 - 1 / self.lc[0]) / 2
        v03 = symm_fold(self.lattice.reflect, self.trap_centers).flatten()
        b3 = list(
            (v03[i] - s3, v03[i] + s3)
            for i in range(2 * self.lattice.Nindep)
            if self.tc_dof[i]
        )
        v03 = v03[self.tc_dof]

        bounds = tuple(b1 + b2 + b3)

        if random:
            v0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        else:
            v0 = np.concatenate((v01, v02, v03))

        self._set_trap_params(v0, self.verbosity or random, "Intial")

        return v0, bounds

    def _set_trap_params(self, v0: np.ndarray, verb, status):
        trap_depth = v0[: self.lattice.Nindep]
        if self.waist_dir != None:
            trap_waist = np.ones((self.lattice.Nindep, 2))
            trap_waist[self.w_dof.reshape(self.lattice.Nindep, 2)] = v0[
                self.lattice.Nindep : np.sum(self.w_dof) + self.lattice.Nindep
            ]
        else:
            trap_waist = None
        trap_center = np.zeros((self.lattice.Nindep, 2))
        trap_center[self.tc_dof.reshape(self.lattice.Nindep, 2)] = v0[
            -np.sum(self.tc_dof) :
        ]
        if verb:
            print(f"\nEqualize: {status} trap depths: {trap_depth}")
            if self.waist_dir != None:
                print(f"Equalize: {status} waists:")
                print(trap_waist)
            print(f"Equalize: {status} trap centers:")
            print(trap_center)
        return trap_depth, trap_waist, trap_center

    def txy_target(self, nnt, links, func: Callable = np.min):
        xlinks, ylinks = links
        nntx = func(abs(nnt[xlinks]))  # Find x direction links
        # Find y direction links, if lattice is 1D this is nan
        nnty = func(abs(nnt[ylinks])) if any(ylinks == True) else None
        return nntx, nnty

    def _set_t(self, A, links, target):
        links = self.xy_links() if links is None else links
        nnt = self.nn_tunneling(A)
        # Mostly not usable if not directly call this function
        if target is None:
            txTarget, tyTarget = self.txy_target(nnt, links)
        elif isinstance(target, Iterable):
            txTarget, tyTarget = target
            if txTarget is None:
                txTarget, tyTarget = self.txy_target(nnt, links)
        xlinks, ylinks = links
        return nnt, txTarget, tyTarget, xlinks, ylinks

    def param_unfold(self, point: np.ndarray, status: str = "current"):
        td, tw, tc = self._set_trap_params(point, self.verbosity, status)
        self.symm_unfold(self.Voff, td)
        if self.waist_dir != None:
            self.symm_unfold(self.waists, tw)
        self.symm_unfold(self.trap_centers, tc, graph=True)
        self.update_lattice(self.trap_centers)
        return self.Voff, self.waists, self.trap_centers, self.eqinfo

    def opt_func(
        self,
        point: np.ndarray,
        info: Union[EqulizeInfo, None],
        links: tuple[np.ndarray, np.ndarray],
        target: tuple[float, ...],
        weight: np.ndarray = np.ones(3),
        scale_factor: float = None,
        eig_vec: list[np.ndarray] = None,
        unitary: Union[list, None] = None,
        mode: str = "cost",
        report: ConfigObj = None,
    ) -> float:
        self.param_unfold(point, "current")

        # By accessing element of a list, x0 is mutable and can be updated
        x0 = unitary[0] if unitary != None and self.lattice.dim > 1 else None
        u = weight[0] != 0

        A, U, __ = self.singleband_Hubbard(u=u, x0=x0, W0=eig_vec, offset=True)
        # x0 is used to update unitary[0] in the next iteration

        # Print out Hubbard parameters
        if self.verbosity > 1:
            print(f"scale_factor = {scale_factor}")
            print(f"V = {np.diag(A)}")
            print(f"t = {abs(self.nn_tunneling(A))}")
            print(f"U = {U}")

        if not isinstance(target, Iterable):
            target = (None, None, None, None)

        maskedU = self.ghost.mask_quantity(U) if u else None

        if mode in ["cost", "nlopt"]:
            return self._cost_func(
                point,
                info,
                scale_factor,
                report,
                (A, maskedU),
                links,
                target,
                weight,
            )
        elif mode in ["res", "func"]:
            return self._res_func(
                point,
                info,
                scale_factor,
                report,
                (A, maskedU),
                links,
                target,
                weight,
            )
        else:
            raise ValueError(f"Equalize: mode {mode} not supported.")

    # ================= GENERAL MINIMIZATION =================

    def _cost_func(
        self, point, info: EqulizeInfo, scale_factor, report, res, links, target, w
    ):
        Vtarget, Utarget, txTarget, tyTarget = target
        A, maskedU = res
        maskedA =self.ghost.mask_quantity(A)

        # U is different, as calculating U costs time
        cu = self.u_cost_func(maskedU, Utarget, scale_factor) if w[0] else 0
        cv = self.v_cost_func(A, Vtarget, scale_factor)
        ct = self.t_cost_func(maskedA, links, (txTarget, tyTarget), scale_factor)

        cvec = np.array((cu, ct, cv))
        c = w @ cvec
        cvec = np.sqrt(cvec)
        fval = np.sqrt(c)
        info.update_log(self, point, report, target, cvec, fval)
        return c

    def v_cost_func(
        self, A, Vtarget: float, Vfactor: float = None
    ) -> float:
        Vdiff = self.v_res_func(A, Vtarget, Vfactor)
        cv = np.sum(Vdiff**2)
        if self.verbosity > 1:
            print(f"Onsite potential cost cv^2 = {cv}")
        return cv

    def t_cost_func(
        self,
        maskedA: np.ndarray,
        links: tuple[np.ndarray, np.ndarray],
        target: tuple[float, ...],
        tfactor: float,
    ) -> float:
        tdiff = self.t_res_func(maskedA, links, target, tfactor)
        ct = np.sum(tdiff**2)
        if self.verbosity > 1:
            print(f"Tunneling cost ct^2 = {ct}")
        return ct

    def u_cost_func(self, maskedU, Utarget: float, Ufactor: float = None) -> float:
        Udiff = self.u_res_func(maskedU, Utarget, Ufactor)
        cu = np.sum(Udiff**2)
        if self.verbosity > 1:
            print(f"Onsite interaction cost cu^2 = {cu}")
        return cu

    # ==================== LEAST SQUARES & ROOT FINDING ====================

    def _res_func(
        self, point, info: EqulizeInfo, scale_factor, report, res, links, target, w
    ):
        Vtarget, Utarget, txTarget, tyTarget = target
        A, maskedU = res
        maskedA = self.ghost.mask_quantity(A)

        cu = (
            self.u_res_func(maskedU, Utarget, scale_factor)
            if w[0]
            else np.zeros(self.lattice.N)
        )
        cv = self.v_res_func(A, Vtarget, scale_factor)
        ct = self.t_res_func(maskedA, links, (txTarget, tyTarget), scale_factor)

        cvec = np.array([la.norm(cu), la.norm(ct), la.norm(cv)])
        # Weighted cost function, weight is in front of each squared term
        c = np.concatenate([np.sqrt(w[0]) * cu, np.sqrt(w[1]) * ct, np.sqrt(w[2]) * cv])
        # The cost func val in least_squares is fval**2 / 2
        fval = la.norm(c)
        info.update_log(self, point, report, target, cvec, fval)
        return c

    def v_res_func(
        self,
        A,
        Vtarget: float,
        Vfactor: float = None    ):
        V = np.real(np.diag(A))
        if len(V) == self.ghost.Nsite:
            maskedV = V
        else:
            maskedV = self.ghost.mask_quantity(V)
        Vtarget, Vfactor = _set_uv(maskedV, Vtarget, Vfactor)

        Vdist = V - Vtarget
        self.ghost.penalty(Vdist)
        cv = Vdist / (Vfactor * np.sqrt(len(maskedV)))
        if self.verbosity > 1:
            print(f"Onsite potential target = {Vtarget}")
            if self.verbosity > 2:
                print(f"Onsite potential residue cv = {cv}")
        return cv

    def t_res_func(
        self,
        maskedA: np.ndarray,
        links: tuple[np.ndarray, np.ndarray],
        target: tuple[float, ...],
        tfactor: float,
    ) -> np.ndarray:
        nnt, txTarget, tyTarget, xlinks, ylinks = self._set_t(maskedA, links, target)
        if tfactor is None:
            tfactor = np.min([txTarget, tyTarget]) if tyTarget != None else txTarget
        ct = (abs(nnt[xlinks]) - txTarget) / (tfactor * np.sqrt(np.sum(xlinks)))
        if tyTarget != None:
            ct = np.concatenate(
                (
                    ct,
                    (abs(nnt[ylinks]) - tyTarget) / (tfactor * np.sqrt(np.sum(ylinks))),
                )
            )
        if self.verbosity > 1:
            print(f"Tunneling target = {txTarget}, {tyTarget}")
            if self.verbosity > 2:
                print(f"Tunneling residue ct = {ct}")
        return ct

    def u_res_func(self, maskedU, Utarget: float, Ufactor: float = None):
        Utarget, Ufactor = _set_uv(maskedU, Utarget, Ufactor)
        cu = (maskedU - Utarget) / (Ufactor * np.sqrt(len(maskedU)))
        if self.verbosity > 1:
            print(f"Onsite interaction target = {Utarget}")
            if self.verbosity > 2:
                print(f"Onsite interaction residue cu = {cu}")
        return cu
