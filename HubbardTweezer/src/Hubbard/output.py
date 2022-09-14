import tools.reportIO as rep
from configobj import ConfigObj
import numpy as np

from .core import MLWF


def write_equalization(report: ConfigObj, info: dict, write_log: bool = False, final: bool = False):
    """
    Record equalization results to the report.
    """
    values = {"x": info["x"][-1],
              "cost_func_by_terms": info['cost'][-1],
              "cost_func_value": info["fval"][-1],
              "total_cost_func": info["ctot"][-1],
              "func_eval_number": info["Nfeval"],
              "scale_factor": info["sf"],
              "success": info["success"],
              "equalize_status": info["exit_status"],
              "termination_reason": info["termination_reason"]}
    if final:
        values["U_over_t"] = info["Ut"]
    rep.create_report(report, "Equalization_Result", **values)
    if write_log:
        values = {"x": info["x"],
                  "cost_func_by_terms": info['cost'],
                  "cost_func_value": info["fval"],
                  "total_cost_func": info["ctot"]}
        rep.create_report(report, "Equalization_Log", **values)


def write_trap_params(report, G: MLWF):
    values = {
        "V_offset": G.Voff,
        "trap_centers": G.trap_centers,
        "waist_factors": G.waists
    }
    rep.create_report(report, "Trap_Adjustments", **values)


def write_singleband(report, G: MLWF):
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


def update_saved_data(report: ConfigObj, G: MLWF):
    G.U, G.A = read_Hubbard(report)
    G.A = G.A - np.eye(G.A.shape[0]) * np.mean(np.diag(G.A))
    Vi = np.real(np.diag(G.A))
    tij = abs(np.real(G.A - np.diag(Vi)))
    values = {"t_ij": tij, "V_i": Vi, "U_i": G.U}
    rep.create_report(report, "Singleband_Parameters", **values)


def read_trap(report: ConfigObj):
    report = rep.get_report(report)
    Voff = rep.a(report, "Trap_Adjustments", "V_offset")
    tc = rep.a(report, "Trap_Adjustments", "trap_centers")
    w = rep.a(report, "Trap_Adjustments", "waist_factors")
    sf = rep.f(report, "Equalization_Info", "final_scale_factor")
    return Voff, tc, w, sf


def read_equalization(report: ConfigObj, G: MLWF):
    """
    Read equalization parameters from file.
    """
    report = rep.get_report(report)
    G.Voff, G.trap_centers, G.waists, G.sf = read_trap(report)
    return G
