import reportIO as rep
import numpy as np
import h5py
import glob

path = "/Users/nottforestfc/Library/CloudStorage/OneDrive-RiceUniversity/Documents/Research/Hubbard Tweezer Parameters/output/LinearRegression/samples_new/"
keys = [
    "V_offset",
    "trap_centers",
    "t_ij",
    # "V_i", # STORED IN t_ij DIAGONALS
    "U_i",
    "wf_centers",
    "wf_cost",
    "total_cost_func",
    "dir",
]
dat = {k: [] for k in keys}

for idx in range(3001):
    filename = path + f"3D_4x4_square_None_neq_{idx}.ini"
    print(filename)
    report = rep.get_report(filename)

    dat["V_offset"].append(rep.a(report, "Trap_Adjustments", "V_offset"))
    dat["trap_centers"].append(rep.a(report, "Trap_Adjustments", "trap_centers"))
    tij = rep.a(report, "Singleband_Parameters", "t_ij")
    tij += np.diag(rep.a(report, "Singleband_Parameters", "V_i"))
    dat["t_ij"].append(tij)
    dat["U_i"].append(rep.a(report, "Singleband_Parameters", "U_i"))
    dat["wf_centers"].append(rep.a(report, "Singleband_Parameters", "wf_centers"))
    dat["wf_cost"].append(rep.a(report, "Singleband_Parameters", "wf_cost"))
    dat["total_cost_func"].append(
        rep.f(report, "Equalization_Result", "total_cost_func")
    )
    dat["dir"].append(idx)

output = "LinearRegression/Learning_TB_Haotian/alldata.hdf5"
with h5py.File(output, "w") as f:
    print(f"Writing to file {output}...")
    for k in keys:
        f[k] = np.asarray(dat[k])
