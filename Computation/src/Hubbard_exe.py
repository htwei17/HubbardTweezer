from Hubbard_plot import *
import reportIO as rep
import sys

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
band = rep.i(report, "Parameters", "band", 1)
dim = rep.i(report, "Parameters", "dimension", 1)
avg = rep.i(report, "Parameters", "average", 1)
sp = rep.b(report, "Parameters", "sparse", True)
eq = rep.b(report, "Parameters", "equalize", False)
eqt = rep.s(report, "Parameters", "equalize_target", 'vt')
symm = rep.b(report, "Parameters", "symmetry", True)

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
    zR=zR,  # Rayleigh range input by han
    sparse=sp,
    equalize=eq,
    eqtarget=eqt,
    symmetry=symm)
eig_sol = eigen_basis(G)
A, U = G.singleband_solution(u=True, eig_sol=eig_sol)
# G.draw_graph('adjust', A, U)
# G.draw_graph(A=A, U=U)
Vi = np.diag(A)
tij = A - np.diag(Vi)

values = {"t_ij": np.real(tij), "V_i": np.real(Vi), "U": U}
rep.create_report(report, "Singleband_Parameters", **values)

values = {
    "V_offset": G.Voff,
    "trap_centers": G.trap_centers,
    # "edge_lengths": G.edge_label
}
rep.create_report(report, "Trap_Adjustments", **values)

if band > 1:
    A, U = optimize(G, *eig_sol)
    values = {}
    for i in range(band):
        Vi = np.diag(A[i])
        tij = A[i] - np.diag(Vi)
        values[f"t_{i+1}_ij"] = np.real(tij)
        values[f"V_{i+1}_i"] = np.real(Vi)

    V = interaction(G, U, *eig_sol[1:])
    for i in range(band):
        for j in range(band):
            values[f"U_{i+1}{j+1}_i"] = V[i, j]

    rep.create_report(report, "Multiband_Parameters", **values)
