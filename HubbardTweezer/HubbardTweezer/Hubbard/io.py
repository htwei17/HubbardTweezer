from numbers import Number
from typing import Iterable, Union
from configobj import ConfigObj
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult

from ..tools import reportIO as rep

# from .core import MLWF


class EqulizeInfo(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self["Nfeval"] = 0
        self["cost"] = np.empty((0, 3))
        self["ctot"] = np.array([])
        self["fval"] = np.array([])
        self["diff"] = np.array([])

    def create_log(self, v0: np.ndarray, target: Iterable):
        self["x"] = v0[None]
        self._update_target(target)

    def _update_target(self, target: Iterable):
        Vtarget, Utarget, txTarget, tyTarget = target
        self["Utarget"] = Utarget
        self["ttarget"] = (txTarget, tyTarget)
        self["Vtarget"] = Vtarget

    def _update_simplex(self):
        N = self["x"].shape[-1]
        # Pick the last 2N points, and select N+1 lowest cost points
        idx = np.argsort(self["fval"][-2 * N :])
        self["simplex"] = np.take(self["x"][-2 * N :], idx[: N + 1], axis=0)

    def update_log(self, G, point, report, target, cvec, fval, io_freq: int = 10):
        # Keep revcord
        ctot = la.norm(cvec)
        self["Nfeval"] += 1
        self["x"] = np.append(self["x"], point[None], axis=0)
        self.update_cost(cvec, fval, ctot)
        diff = self["fval"][len(self["fval"]) // 2] - fval
        self["diff"] = np.append(self["diff"], diff)
        # display selfrmation
        if self["Nfeval"] % io_freq == 0:
            if isinstance(report, ConfigObj):
                self._update_target(target)
                if G.eqmethod == "Nelder-Mead":
                    self._update_simplex()
                self["sf"] = G.sf
                self["success"] = False
                self["exit_status"] = -1
                self["termination_reason"] = "Not terminated yet"
                self.write_equalization(report, G.log)
                write_trap_params(report, G)
                write_singleband(report, G)
        if G.verbosity:
            print(f"Cost function by terms = {cvec}")
            print(f"Cost function total value fval = {fval}\n")
        print(f'i={self["Nfeval"]}\tc={cvec}\tc_i={fval}\tc_i//2-c_i={diff}')

    def update_cost(self, cvec, fval, ctot):
        self["cost"] = np.append(self["cost"], cvec[None], axis=0)
        self["ctot"] = np.append(self["ctot"], ctot)
        self["fval"] = np.append(self["fval"], fval)

    def update_log_final(self, res: OptimizeResult, sf: float):
        self["sf"] = sf
        self["success"] = res.success
        self["exit_status"] = res.status
        self["termination_reason"] = res.message
        if res.get("final_simplex", None) is not None:
            self["simplex"] = res["final_simplex"][0]

    def write_equalization(self, report: ConfigObj, write_log: bool = False):
        """
        Record equalization results to the report.
        """
        values = {
            "x": self["x"][-1],
            "cost_func_by_terms": self["cost"][-1],
            "cost_func_value": self["fval"][-1],
            "total_cost_func": self["ctot"][-1],
            "func_eval_number": self["Nfeval"],
            "U_target": self["Utarget"],
            "t_target": self["ttarget"],
            "V_target": self["Vtarget"],
            "scale_factor": self["sf"],
            "success": self["success"],
            "equalize_status": self["exit_status"],
            "termination_reason": self["termination_reason"],
        }
        if self.get("simplex", None) is not None:
            values["simplex"] = self["simplex"]
        if self.get("Ut", None) is not None:
            values["U_over_t"] = self["Ut"]
        rep.create_report(report, "Equalization_Result", **values)
        if write_log:
            values = {
                "x": self["x"],
                "cost_func_by_terms": self["cost"],
                "cost_func_value": self["fval"],
                "total_cost_func": self["ctot"],
            }
            rep.create_report(report, "Equalization_Log", **values)

    def read_equalizatśon_log(self, report: ConfigObj, G, index: int = 0):
        report = rep.get_report(report)
        self["x"] = rep.a(report, "Equalization_Log", "x")
        self["cost"] = rep.a(report, "Equalization_Log", "cost_func_by_terms")
        self["fval"] = rep.a(report, "Equalization_Log", "cost_func_value")
        self["ctot"] = rep.f(report, "Equalization_Log", "total_cost_func")
        G.eff_dof()
        G.param_unfold(
            self["x"][index - 1], f"{index -1 if index <= 0 else index}-th equalization"
        )
        return G


def write_trap_params(report, G):
    values = {
        "V_offset": G.Voff,
        "trap_centers": G.trap_centers,
        "waist_factors": G.waists,
    }
    rep.create_report(report, "Trap_Adjustments", **values)


def write_singleband(report, G):
    # FIXME: If not final result, G.U might be None.
    Vi = np.real(np.diag(G.A))
    tij = abs(np.real(G.A - np.diag(Vi)))
    values = {"t_ij": tij, "V_i": Vi, "U_i": G.U, "wf_centers": G.wf_centers}
    rep.create_report(report, "Singleband_Parameters", **values)


def read_Hubbard(report: ConfigObj, band: int = 1):
    """
    Read parameters from file.
    """
    if isinstance(band, Number):
        if band == 1:
            section = "Singleband_Parameters"
            U = rep.a(report, section, "U_i")
            Vi = rep.a(report, section, "V_i")
            tij = rep.a(report, section, "t_ij")
            wc = rep.a(report, section, "wf_centers")
        else:
            section = "Multiband_Parameters"
            U = rep.a(report, section, f"U_{band}{band}_i")
            Vi = rep.a(report, section, f"V_{band}_i")
            tij = rep.a(report, section, f"t_{band}_ij")
            wc = rep.a(report, section, f"wf_{band}_centers")
        A = np.diag(Vi) + tij
    elif isinstance(band, Iterable):
        section = "Multiband_Parameters"
        U = rep.a(report, section, f"U_{band[0]}{band[1]}_i")
        A = None
        wc = None
    return U, A, wc


def update_saved_data(report: Union[ConfigObj, str], G):
    if isinstance(report, str):
        report = rep.get_report(report)
    read_trap(report, G)
    eig_sol = G.eigen_basis()
    G.singleband_Hubbard(u=True, eig_sol=eig_sol)
    maskedA = G.A[G.mask, :][:, G.mask]
    maskedU = G.U[G.mask]
    links = G.xy_links(G.masked_links)

    nnt = G.nn_tunneling(maskedA)
    if G.sf == None:
        G.sf, __ = G.txy_target(nnt, links, np.min)

    # ====== Write output ======
    write_singleband(report, G)
    write_trap_params(report, G)


def update_tc(report: Union[ConfigObj, str], G):
    # Update trap centers from unit of itself to unit of the waist.
    if isinstance(report, str):
        report = rep.get_report(report)
    read_trap(report, G)
    G.trap_centers = rep.a(report, "Trap_Adjustments", "trap_centers")
    G.trap_centers = G.trap_centers * G.lc
    write_trap_params(report, G)


def read_trap(report: Union[ConfigObj, str], G):
    if isinstance(report, str):
        report = rep.get_report(report)
    G.Voff, G.trap_centers, G.waists, G.sf = read_trap_params(report)
    return G


def read_trap_params(report: Union[ConfigObj, str]):
    """
    Read equalized trap parameters from file.
    """
    if isinstance(report, str):
        report = rep.get_report(report)
    Voff = rep.a(report, "Trap_Adjustments", "V_offset")
    tc = rep.a(report, "Trap_Adjustments", "trap_centers")
    w = rep.a(report, "Trap_Adjustments", "waist_factors")
    sf = rep.f(report, "Equalization_Result", "scale_factor")
    return Voff, tc, w, sf


def read_target(report: Union[ConfigObj, str]):
    """
    Read target parameters from file.
    """
    if isinstance(report, str):
        report = rep.get_report(report)
    Utarget = rep.a(report, "Equalization_Result", "U_target")
    ttarget = rep.a(report, "Equalization_Result", "t_target")
    Vtarget = rep.a(report, "Equalization_Result", "V_target")
    txTarget, tyTarget = ttarget[0], ttarget[1]
    return Utarget, txTarget, tyTarget, Vtarget


def read_file(report: Union[ConfigObj, str], G, band: int = 1):
    if isinstance(report, str):
        report = rep.get_report(report)
    read_trap(report, G)
    G.U, G.A, G.wf_centers = read_Hubbard(report, band=band)
    if G.wf_centers is None:
        G.wf_centers = G.trap_centers
    return G
