import reportIO as rep
import numpy as np
import h5py
import glob

L = 4

path = f"/Users/nottforestfc/Library/CloudStorage/OneDrive-RiceUniversity/Documents/Research/Hubbard Tweezer Parameters/output/LinearRegression/sample/{L}x{L}/"
keys = [
    "V_offset",
    "V_tot",
    "trap_centers",
    "t_ij",
    "V_i",
    "U_i",
    "wf_centers",
    "wf_cost",
    "total_cost_func",
    "dir",
]
dat = {k: [] for k in keys}

for idx in range(3003):
    filename = path + f"3D_4x4_square_None_neq_{idx}.ini"
    print(filename)
    report = rep.get_report(filename)

    dat["V_offset"].append(rep.a(report, "Trap_Adjustments", "V_offset"))
    dat["trap_centers"].append(rep.a(report, "Trap_Adjustments", "trap_centers"))
    tij = rep.a(report, "Singleband_Parameters", "t_ij")
    # tij += np.diag(rep.a(report, "Singleband_Parameters", "V_i"))
    dat["t_ij"].append(tij)
    dat["V_i"].append(rep.a(report, "Singleband_Parameters", "V_i"))
    dat["U_i"].append(rep.a(report, "Singleband_Parameters", "U_i"))
    dat["wf_centers"].append(rep.a(report, "Singleband_Parameters", "wf_centers"))
    dat["wf_cost"].append(rep.a(report, "Singleband_Parameters", "wf_cost"))
    dat["total_cost_func"].append(
        rep.f(report, "Equalization_Result", "total_cost_func")
    )
    dat["dir"].append(idx)

wxy = np.ones(2)
width = 4
height = 4
N = width * height
dim = 3


def Vfun(x, y):
    dxy = x**2 + y**2
    V = -np.exp(-2 * dxy)
    return V


def tot_trap_depth(V0, trap_centers):
    # tc = np.zeros((N, dim))
    tc = np.zeros((N, 2))
    vij = np.ones((N, N))
    for i in range(N):
        # tc[i, :] = np.append(trap_centers[i], 0)
        tc[i, :] = trap_centers[i]
        for j in range(i):
            vij[i, j] = -Vfun(*(tc[i] - tc[j]))
            vij[j, i] = vij[i, j]  # Potential is symmetric in distance
    vtot = vij @ V0
    return vtot


dat["V_tot"] = np.array(list(map(tot_trap_depth, dat["V_offset"], dat["trap_centers"])))
# vtot_rel = vtot - np.mean(vtot, axis=1)[:, None]

output = f"LinearRegression/HubbardFit/{L}x{L}_alldata.hdf5"
with h5py.File(output, "w") as f:
    print(f"Writing to file {output}...")
    for k in keys:
        f[k] = np.asarray(dat[k])

# Mask the data
mask = np.min(np.asarray(dat["U_i"]), axis=1) > 0.8
output = f"LinearRegression/HubbardFit/{L}x{L}_alldata_masked.hdf5"
with h5py.File(output, "w") as f:
    print(f"Writing to file {output}...")
    for k in dat.keys():
        f[k] = np.asarray(dat[k])[mask]
