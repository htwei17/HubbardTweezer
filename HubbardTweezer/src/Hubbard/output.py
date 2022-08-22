import tools.reportIO as rep
from configobj import ConfigObj


def write_equalization(report: ConfigObj, G, info: dict, final: bool = False):
    """
    Overwrite equalization log to the report.
    """
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
