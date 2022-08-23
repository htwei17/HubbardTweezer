import tools.reportIO as rep
from configobj import ConfigObj
import numpy as np

from .core import MLWF


def write_equalization(report: ConfigObj, G: MLWF, info: dict, eq: bool = True, final: bool = False):
    """
    Overwrite equalization log to the report.
    """
    if eq:
        values = {"x": info["x"][-1],
                  "cost_func_terms": info['cost'][-1],
                  "min_target_value": info["fval"][-1],
                  "total_cost_func": info["ctot"][-1],
                  "func_evals": info["Nfeval"],
                  }
        if final:
            values["equalize_status"] = info["exit_status"]
            values["termination_reason"] = info["termination_reason"]
        rep.create_report(report, "Equalization_Info", **values)

    values = {
        "V_offset": G.Voff,
        "trap_centers": G.trap_centers,
        "waist_factors": G.waists
    }
    rep.create_report(report, "Trap_Adjustments", **values)

    if not final:
        write_singleband(report, G)


def write_singleband(report, G: MLWF):
    Vi = np.real(np.diag(G.A))
    tij = abs(np.real(G.A - np.diag(Vi)))
    values = {"t_ij": tij, "V_i": Vi, "U_i": G.U}
    rep.create_report(report, "Singleband_Parameters", **values)


def read_equalization(report: ConfigObj, G: MLWF):
    """
    Read equalization parameters from file.
    """
    report = rep.get_report(report)
    G.Voff = rep.a(report, "Trap_Adjustments", "V_offset")
    G.trap_centers = rep.a(report, "Trap_Adjustments", "trap_centers")
    G.waists = rep.a(report, "Trap_Adjustments", "waist_factors")
    return G


def read_parameters(report: ConfigObj, G: MLWF):
    """
    Read parameters from file.
    """
    report = rep.get_report(report)
    U = rep.a(report, "Singleband_Parameters", "U_i")
    Vi = rep.a(report, "Singleband_Parameters", "V_i")
    tij = rep.a(report, "Singleband_Parameters", "t_ij")
    A = np.diag(Vi) + tij
    return U, A
