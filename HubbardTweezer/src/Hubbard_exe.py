import numpy as np
from Hubbard.plot import (HubbardGraph, eigen_basis, optimize, interaction)
import tools.reportIO as rep
import sys

# ====== Read input ======
inFile = sys.argv[1]
# outFile = sys.argv[2]

report = rep.get_report(inFile)

N = rep.i(report, "Parameters", "N", 20)
L0 = rep.a(report, "Parameters", "L0", np.array([3, 3, 7.2]))
lattice = rep.a(report, "Parameters", "lattice", np.array([4])).astype(int)
lc = tuple(rep.a(report, "Parameters", "lattice_const", np.array([1520,
                                                                  1690])))
a_s = rep.f(report, "Parameters", "scattering_length", 1000)
V0 = rep.f(report, "Parameters", "V0", 104.52)
w = rep.a(report, "Parameters", "waist", np.array([1000, 1000]))
m = rep.f(report, "Parameters", "atom_mass", 6.015122)
zR = rep.f(report, "Parameters", "zR", None)
l = rep.f(report, "Parameters", "laser_wavelength", 780)
wd = rep.s(report, "Parameters", "waist_dir", None)
band = rep.i(report, "Parameters", "band", 1)
dim = rep.i(report, "Parameters", "dimension", 1)
avg = rep.i(report, "Parameters", "average", 1)
sp = rep.b(report, "Parameters", "sparse", True)
eq = rep.b(report, "Parameters", "equalize", False)
eqt = rep.s(report, "Parameters", "equalize_target", 'vt')
symm = rep.b(report, "Parameters", "symmetry", True)
verb = rep.i(report, "Parameters", "verbosity", 0)

# ====== Equalize ======
G = HubbardGraph(
    N,
    R0=L0,
    lattice=lattice,
    lc=lc,
    ascatt=a_s,
    band=band,
    dim=dim,
    avg=avg,
    model='Gaussian',  # Tweezer potetnial
    trap=(V0, w),  # 2nd entry in array is (wx, wy), in number is (w, w)
    atom=m,  # Atom mass, in amu. Default Lithium-6
    laser=l,  # Laser wavelength
    zR=zR,  # Rayleigh range input by hand
    waist=wd,  # Waist varying directions
    sparse=sp,
    equalize=eq,
    eqtarget=eqt,
    symmetry=symm,
    verbosity=verb)
eig_sol = eigen_basis(G)
A, U = G.singleband_Hubbard(u=True, eig_sol=eig_sol)
# G.draw_graph('adjust', A, U)
# G.draw_graph(A=A, U=U)
Vi = np.real(np.diag(A))
tij = abs(np.real(A - np.diag(Vi)))

# ====== Write output ======
values = {"t_ij": tij, "V_i": Vi, "U": U}
rep.create_report(report, "Singleband_Parameters", **values)

values = {
    "V_offset": G.Voff,
    "trap_centers": G.trap_centers,
    "waist_factors": G.waists
}
rep.create_report(report, "Trap_Adjustments", **values)

if G.band > 1:
    A, U = optimize(G, *eig_sol)
    values = {}
    for i in range(band):
        Vi = np.real(np.diag(A[i]))
        tij = abs(np.real(A[i] - np.diag(Vi)))
        values[f"t_{i+1}_ij"] = tij
        values[f"V_{i+1}_i"] = Vi

    V = interaction(G, U, *eig_sol[1:])
    for i in range(band):
        for j in range(band):
            values[f"U_{i+1}{j+1}_i"] = V[i, j]

    rep.create_report(report, "Multiband_Parameters", **values)
