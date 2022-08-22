import tools.reportIO as rep
from configobj import ConfigObj

from .equalizer import HubbardParamEqualizer


def write_equalization(report: ConfigObj, G: HubbardParamEqualizer, info : dict):
    """
    Overwrite equalization log to the report.
    """
    values = {"x": info["x"],
              "min_target_value": info["fval"][-1],
              "total_cost_func": info["ctot"][-1],
              "func_evals": info["Nfeval"],
              "equalize_status": info["exit_status"],
              "termination_reason": info["termination_reason"]
              }
    rep.create_report(report, "Equalization_Info", **values)
    values = {
        "V_offset": G.Voff,
        "trap_centers": G.trap_centers,
        "waist_factors": G.waists
    }
    rep.create_report(report, "Trap_Adjustments", **values)
