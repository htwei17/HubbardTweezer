from typing import Iterable
import tools.reportIO as rep
from configobj import ConfigObj
import numpy as np
import numpy.linalg as la
from scipy.optimize import OptimizeResult

# from .core import MLWF


class EqulizeInfo(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self['Nfeval'] = 0
        self['cost'] = np.empty((0, 3))
        self['ctot'] = np.array([])
        self['fval'] = np.array([])
        self['diff'] = np.array([])

    def create_log(self, v0: np.ndarray, target: Iterable):
        self['x'] = v0[None]
        self._update_target(target)

    def _update_target(self, target: Iterable):
        Vtarget, Utarget, txTarget, tyTarget = target
        self['Utarget'] = Utarget
        self['ttarget'] = (txTarget, tyTarget)
        self['Vtarget'] = Vtarget

    def _update_simplex(self):
        N = self['x'].shape[-1]
        self['simplex'] = self['x'][-N+1:].copy()

    def update_log(self, G, point, report, target, cvec, fval, io_freq: int = 10):
        # Keep revcord
        ctot = la.norm(cvec)
        diff = self['fval'][len(self['fval'])//2] - fval

        self['Nfeval'] += 1
        self['x'] = np.append(self['x'], point[None], axis=0)
        self.update_cost(cvec, fval, ctot)
        self['diff'] = np.append(self['diff'], diff)
        # display selfrmation
        if self['Nfeval'] % io_freq == 0:
            if isinstance(report, ConfigObj):
                self._update_target(target)
                if G.method == 'Nelder-Mead':
                    self._update_simplex()
                self['sf'] = G.sf
                self['success'] = False
                self['exit_status'] = -1
                self['termination_reason'] = "Not terminated yet"
                self.write_equalization(report, G.log)
                write_trap_params(report, G)
                write_singleband(report, G)
        if G.verbosity:
            print(f"Cost function by terms: {cvec}")
            print(f"Total cost function value fval={fval}\n")
            print(
                f'i={self["Nfeval"]}\tc={cvec}\tc_i={fval}\tc_i//2-c_i={diff}')

    def update_cost(self, cvec, fval, ctot):
        self['cost'] = np.append(self['cost'], cvec[None], axis=0)
        self['ctot'] = np.append(self['ctot'], ctot)
        self['fval'] = np.append(self['fval'], fval)

    def update_log_final(self, res: OptimizeResult, sf: float):
        self['sf'] = sf
        self['success'] = res.success
        self['exit_status'] = res.status
        self['termination_reason'] = res.message
        if res.get('final_simplex', None) is not None:
            self['simplex'] = res.items('final_simplex')[0]

    def write_equalization(self, report: ConfigObj, write_log: bool = False):
        """
        Record equalization results to the report.
        """
        values = {"x": self["x"][-1],
                  "cost_func_by_terms": self['cost'][-1],
                  "cost_func_value": self["fval"][-1],
                  "total_cost_func": self["ctot"][-1],
                  "func_eval_number": self["Nfeval"],
                  "U_target": self["Utarget"],
                  "t_target": self["ttarget"],
                  "V_target": self["Vtarget"],
                  "scale_factor": self["sf"],
                  "success": self["success"],
                  "equalize_status": self["exit_status"],
                  "termination_reason": self["termination_reason"]}
        if self.get('simplex', None) is not None:
            values["simplex"] = self['simplex']
        if self.get('Ut', None) is not None:
            values["U_over_t"] = self["Ut"]
        rep.create_report(report, "Equalization_Result", **values)
        if write_log:
            values = {"x": self["x"],
                      "cost_func_by_terms": self['cost'],
                      "cost_func_value": self["fval"],
                      "total_cost_func": self["ctot"]}
            rep.create_report(report, "Equalization_Log", **values)

    def read_equalizatśon_log(self, report: ConfigObj, G, index: int = 0):
        report = rep.get_report(report)
        self['x'] = rep.a(report, "Equalization_Log", "x")
        self['cost'] = rep.a(report, "Equalization_Log", "cost_func_by_terms")
        self['fval'] = rep.a(report, "Equalization_Log", "cost_func_value")
        self['ctot'] = rep.f(report, "Equalization_Log", "total_cost_func")
        G.eff_dof()
        G.param_unfold(self['x'][index - 1],
                       f'{index -1 if index <= 0 else index}-th equalization')
        return G


def write_trap_params(report, G):
    values = {
        "V_offset": G.Voff,
        "trap_centers": G.trap_centers,
        "waist_factors": G.waists
    }
    rep.create_report(report, "Trap_Adjustments", **values)


def write_singleband(report, G):
    # FIXME: If not final result, G.U might be None.
    Vi = np.real(np.diag(G.A))
    tij = abs(np.real(G.A - np.diag(Vi)))
    values = {"t_ij": tij, "V_i": Vi, "U_i": G.U}
    rep.create_report(report, "Singleband_Parameters", **values)


def read_Hubbard(report: ConfigObj):
    """
    Read parameters from file.
    """
    report = rep.get_report(report)
    U = rep.a(report, "Singleband_Parameters", "U_i")
    Vi = rep.a(report, "Singleband_Parameters", "V_i")
    tij = rep.a(report, "Singleband_Parameters", "t_ij")
    A = np.diag(Vi) + tij
    return U, A


def update_saved_data(report: ConfigObj, G):
    G.U, G.A = read_Hubbard(report)
    G.A = G.A - np.eye(G.A.shape[0]) * np.mean(np.diag(G.A))
    Vi = np.real(np.diag(G.A))
    tij = abs(np.real(G.A - np.diag(Vi)))
    values = {"t_ij": tij, "V_i": Vi, "U_i": G.U}
    rep.create_report(report, "Singleband_Parameters", **values)


def read_trap(report: ConfigObj):
    """
    Read equalized trap parameters from file.
    """
    report = rep.get_report(report)
    Voff = rep.a(report, "Trap_Adjustments", "V_offset")
    tc = rep.a(report, "Trap_Adjustments", "trap_centers")
    w = rep.a(report, "Trap_Adjustments", "waist_factors")
    sf = rep.f(report, "Equalization_Result", "scale_factor")
    return Voff, tc, w, sf
