import reportIO as rep
import numpy as np
import h5py
import glob

path = "/Users/nottforestfc/Library/CloudStorage/OneDrive-RiceUniversity/Documents/Research/Hubbard Tweezer Parameters/output/LinearRegression/samples/"
keys = [
    "V_offset",
    "trap_centers",
    "t_ij",
    "V_i",
    "U_i",
    "wf_centers",
    "total_cost_func",
]
dat = {k: [] for k in keys}

for filename in glob.glob(path + "*.ini"):
    print(filename)
    report = rep.get_report(filename)

    dat["V_offset"].append(
        np.asarray([float(x) for x in report["Trap_Adjustments"]["V_offset"]])
    )
    dat["trap_centers"].append(rep.a(report, "Trap_Adjustments", "trap_centers"))
    tij = rep.a(report, "Singleband_Parameters", "t_ij")
    tij += np.diag(rep.a(report, "Singleband_Parameters", "V_i"))
    dat["t_ij"].append(tij)
    dat["U_i"].append(rep.a(report, "Singleband_Parameters", "U_i"))
    dat["wf_centers"].append(rep.a(report, "Singleband_Parameters", "wf_centers"))
    dat["total_cost_func"].append(
        rep.f(report, "Equalization_Result", "total_cost_func")
    )

output = "LinearRegression/Learning_TB_Haotian/alldata.hdf5"
with h5py.File(output, "w") as f:
    print(f"Writing to file {output}...")
    for k in keys:
        f[k] = np.asarray(dat[k])
    print("Done!")
