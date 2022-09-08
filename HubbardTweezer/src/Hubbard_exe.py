import numpy as np
from Hubbard.output import *
from Hubbard.plot import HubbardGraph
from Hubbard.equalizer import *
import tools.reportIO as rep
import sys

# ====== Read arguments ======
inFile = sys.argv[1]
# outFile = sys.argv[2]

# ====== Read parameters ======
report = rep.get_report(inFile)

# ====== DVR parameters ======
N = rep.i(report, "Parameters", "N", 20)
L0 = rep.a(report, "Parameters", "L0", np.array([3, 3, 7.2]))
dim = rep.i(report, "Parameters", "dimension", 1)

# ====== Create lattice ======
lattice = rep.a(report, "Parameters", "lattice", np.array([4])).astype(int)
lc = tuple(rep.a(report, "Parameters", "lattice_const", np.array([1520,
                                                                  1690])))
shape = rep.s(report, "Parameters", "shape", 'square')

# ====== Physical parameters ======
a_s = rep.f(report, "Parameters", "scattering_length", 1000)
V0 = rep.f(report, "Parameters", "V_0", 104.52)
w = rep.a(report, "Parameters", "waist", np.array([1000, 1000]))
m = rep.f(report, "Parameters", "atom_mass", 6.015122)
zR = rep.f(report, "Parameters", "zR", None)
l = rep.f(report, "Parameters", "laser_wavelength", 780)
avg = rep.f(report, "Parameters", "average", 1)

# ====== Equalization ======
eq = rep.b(report, "Parameters", "equalize", False)
eqt = rep.s(report, "Parameters", "equalize_target", 'vt')
ut = rep.f(report, "Parameters", "U/t", None)
wd = rep.s(report, "Parameters", "waist_direction", None)
band = rep.i(report, "Parameters", "band", 1)
meth = rep.s(report, "Parameters", "method", 'trf')
nb = rep.b(report, "Parameters", "no_bounds", False)
r = rep.b(report, "Parameters", "random_init_guess", False)
x0 = rep.a(report, "Equalization_Info", "x", None)

# ====== DVR settings ======
s = rep.b(report, "Parameters", "sparse", True)
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
    shape=shape,  # lattice geometries
    waist=wd,  # Waist varying directions
    sparse=s,  # Sparse matrix
    equalize=eq,
    eqtarget=eqt,
    Ut=ut,
    random=r,
    x0=x0,
    method=meth,
    nobounds=nb,
    symmetry=symm,
    iofile=report,
    verbosity=verb)

eig_sol = eigen_basis(G)
G.singleband_Hubbard(u=True, eig_sol=eig_sol)
G.draw_graph('adjust', G.A, G.U)
G.draw_graph(A=G.A, U=G.U)

# ====== Write output ======
write_singleband(report, G)
write_trap_params(report, G)
eqt = 'uvt' if eqt == 'neq' else eqt
u, t, v, __, __, __ = G.str_to_flags(eqt)
nnt = G.nn_tunneling(G.A)
xlinks, ylinks, txTarget, tyTarget = G.xy_links(nnt)
ct = G.t_cost_func(G.A, (xlinks, ylinks), (txTarget, tyTarget))
cv = G.v_cost_func(G.A, None, txTarget)
cu = G.u_cost_func(G.U, None, txTarget)
cvec = np.sqrt(np.array((cu, ct, cv)))
w = np.array([u, t, v])

if eq:
    G.eqinfo['cost'][-1] = cvec
    G.eqinfo['fval'][-1] = la.norm(w @ cvec)
    G.eqinfo['ctot'][-1] = la.norm(cvec)
else:
    v0, __ = G.init_guess(random=False)
    G.eqinfo['x'] = v0[None]
    G.eqinfo['Nfeval'] = 0
    G.eqinfo['cost'] = cvec[None]
    G.eqinfo['fval'] = np.array([la.norm(w @ cvec)])
    G.eqinfo['ctot'] = np.array([la.norm(cvec)])
    G.eqinfo["exit_status"] = -1
    G.eqinfo["termination_reason"] = "Not equalized"
write_equalize_log(report, G.eqinfo, final=True)

if G.bands > 1:
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
