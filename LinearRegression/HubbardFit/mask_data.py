import numpy as np
import h5py

input = "LinearRegression/HubbardFit/alldata.hdf5"
with h5py.File(input, "r") as f:
    dat = {k: f[k][:] for k in f.keys()}

mask = np.min(dat["U_i"], axis=1) > 0.8
# mask = np.ones(dat["U_i"].shape[0], dtype=bool)

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


vtot = np.array(
    list(map(tot_trap_depth, dat["V_offset"], dat["trap_centers"]))
)
vtot_rel = vtot - np.mean(vtot, axis=1)[:, None]

output = "LinearRegression/HubbardFit/alldata_masked.hdf5"
with h5py.File(output, "w") as f:
    print(f"Writing to file {output}...")
    for k in dat.keys():
        if k != "V_i":
            f[k] = np.asarray(dat[k][mask])
    f["V_i"] = np.asarray(dat["t_ij"][mask].diagonal(axis1=1, axis2=2))
    f["V_tot_rel"] = np.asarray(vtot_rel[mask])