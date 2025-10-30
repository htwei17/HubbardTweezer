import numpy as np
import sys

# import h5py
from os.path import exists

from HubbardTweezer.Hubbard.io import *

# from HubbardTweezer.Hubbard.plot import HubbardGraph
from HubbardTweezer.Hubbard.equalizer import *
import HubbardTweezer.tools.reportIO as rep


def help_message(s=2):
    # print help message and exit
    print(
        """
    Usage: python Hubbard_exe.py <input ini file path>
    
        The program reads input sections from the supplied INI file and appends
        results to that same file. Duplicate keys within a section raise an error.

        # Input Sections
        ----------------

        ## `[DVR_Parameters]`
        - `N` (default 20)
        - `L0` (default 3,3,7.2)
        - `DVR_dimension` (default 1)
        - Optional: `sparse` (default True), `DVR_symmetry` (default True)

        ## `[Lattice_Parameters]`
        - `shape` (default square; supports square, Lieb, triangular, honeycomb,
            defecthoneycomb, kagome, ring, zigzag, Penrose, custom)
        - `lattice_size` (default 4,)
        - `lattice_const` (default 1520,1690 nm)
        - `lattice_symmetry` (default True)
        - `potential_model` (default Gaussian)
            * when `custom`, provide both `custom_potential_grid` and
                `custom_potential_value`
            * when `shape = custom`, supply `site_locations` and optional `bond_links`

        ## `[Trap_Parameters]`
        - `scattering_length` (default 1000 a₀)
        - `V0` (default 104.52 kHz)
        - `waist` (default 1000,1000 nm)
        - `atom_mass` (default 6.015122 amu)
        - `zR` (default None, derived from waist and wavelength)
        - `laser_wavelength` (default 780 nm)
        - `average` (default 1)

        ## `[Hubbard_Settings]`
        - `band` (default 1)
        - `zero_average_V` (default True)
        - `calculate_U` (default True)
        - `Nintgrl_grid` (default 200)
        - `offdiagonal_U` (default False)

        ## `[Equalization_Parameters]`
        - `equalize` (default False)
        - `equalize_item` (default vT; lowercase updates targets every iteration,
            uppercase locks to the initial guess)
        - `balance_V0` (default False)
        - `waist_direction` (default None)
        - `method` (default trf; supports dogbox, Nelder-Mead/NM, Powell, bfgs,
            L-BFGS-B, cobyla, SLSQP, and NLopt methods bobyqa/praxis/subplex/direct/crs2)
        - `no_bounds` (default False)
        - `random_initial_guess` (default False)
        - `ghost_sites` (default False) and `ghost_penalty` (default 1,1)
        - Targets: `U_target`, `t_target`, `V_target` (all default None)
        - Scaling: `scale_factor` (default None)

        ## `[Verbosity]`
        - `write_log` (default False)
        - `verbosity` (default 0)
        - `output_lowest_wannier` (default False)

        ## `[Equalization_Result]`
        - Optional: `x` (initial guess) or `simplex` (for Nelder-Mead)
        - Optional: `U_over_t`

        # Output Sections
        -----------------

        ## `[Singleband_Parameters]`
        Stores single-band Hubbard results (t₍ᵢⱼ₎, Vᵢ, Uᵢ, wf_centers). When
        `calculate_U` is disabled, Uᵢ may be omitted.

        ## `[Trap_Adjustments]`
        Contains the trap offsets, centers (in waist units), and waist scale factors.

        ## `[Equalization_Result]`
        Records optimizer progress: optimized `x`, per-term costs, total cost,
        function-evaluation count, scale factor, success flag, status code, and
        termination message.

        ## `[Equalization_Log]` (optional)
        Populated when `write_log=True`; captures the history of points and costs.

        ## `[Multiband_Parameters]` (optional)
        Available when `band > 1`; mirrors the single-band section with band
        indices. Off-diagonal interactions appear when `offdiagonal_U=True`.
    """
    )
    sys.exit(s)


# ====== Read argument and file ======
try:
    inFile = sys.argv[1]
    # outFile = sys.argv[2]

    if inFile == "--help" or inFile == "-h":
        help_message(2)
    elif exists(inFile):
        report = rep.get_report(inFile)
    else:
        raise FileNotFoundError("Hubbard_exe: input file not found")
except FileNotFoundError as ferr:
    print(ferr)
    print("Usage: python Hubbard_exe.py <input ini file path>")
    print("use -h or --help for help")
    sys.exit(1)

# ====== DVR parameters ======
N = rep.i(report, "DVR_Parameters", "N", 20)
L0 = rep.a(report, "DVR_Parameters", "L0", np.array([3, 3, 7.2]), output_arr=True)
dimension = rep.i(report, "DVR_Parameters", "DVR_dimension", 1)
s = rep.b(report, "DVR_Parameters", "sparse", True)
symm = rep.b(report, "DVR_Parameters", "DVR_symmetry", True)

# ====== Create lattice ======
shape = rep.s(report, "Lattice_Parameters", "shape", "square")
ls = rep.b(report, "Lattice_Parameters", "lattice_symmetry", True)
lc = tuple(rep.a(report, "Lattice_Parameters", "lattice_const", [1520, 1690]))
lsize = rep.a(
    report, "Lattice_Parameters", "lattice_size", np.array([4]), output_arr=True
).astype(int)
nodes = None
links = None
model = rep.s(report, "Lattice_Parameters", "potential_model", "Gaussian")
custom_potential = None
if model in ["Gaussian", "optical_lattice"]:
    if model == "optical_lattice" and shape != "square":
        raise ValueError(
            "optical_lattice model only supports square lattice shape, please use Gaussian model for other shapes."
        )
    if shape == "custom":
        nodes = rep.a(report, "Lattice_Parameters", "site_locations", None)
        links = rep.a(report, "Lattice_Parameters", "bond_links", None)
elif model == "custom":
    custom_potential_grid = rep.a(
        report, "Lattice_Parameters", "custom_potential_grid", None
    )
    custom_potential_value = rep.a(
        report, "Lattice_Parameters", "custom_potential_value", None
    )
    if custom_potential_grid is not None and custom_potential_value is not None:
        custom_potential = (custom_potential_grid, custom_potential_value)
    else:
        custom_potential = None
        print("Custom potential grid or value not provided, using default potential.")

# ====== Physical trap parameters ======
a_s = rep.f(report, "Trap_Parameters", "scattering_length", 1000)
V0 = rep.f(report, "Trap_Parameters", "V0", 104.52)
w = rep.a(report, "Trap_Parameters", "waist", np.array([1000, 1000]), output_arr=True)
m = rep.f(report, "Trap_Parameters", "atom_mass", 6.015122)
zR = rep.f(report, "Trap_Parameters", "zR", None)
l = rep.f(report, "Trap_Parameters", "laser_wavelength", 780)
avg = rep.f(report, "Trap_Parameters", "average", 1)

# ====== Hubbard settings ======
band = rep.i(report, "Hubbard_Settings", "band", 1)
zero_avgV = rep.b(report, "Hubbard_Settings", "zero_average_V", True)
calculate_U = rep.b(report, "Hubbard_Settings", "calculate_U", True)
Nintgrl_grid = rep.i(report, "Hubbard_Settings", "Nintgrl_grid", 200)
offdiag_U = rep.b(report, "Hubbard_Settings", "offdiagonal_U", False)

ut = rep.f(report, "Equalization_Result", "U_over_t", None)

# ====== Equalization ======
eq = rep.b(report, "Equalization_Parameters", "equalize", False)
if eq and model == "custom":
    raise ValueError("Equalization not supported for custom potential profile.")
eqt = rep.s(report, "Equalization_Parameters", "equalize_item", "vT")
balance_V0 = rep.b(report, "Equalization_Parameters", "balance_V0", False)
wd = rep.s(report, "Equalization_Parameters", "waist_direction", None)
meth = rep.s(report, "Equalization_Parameters", "method", "trf")
nb = rep.b(report, "Equalization_Parameters", "no_bounds", False)
gho = rep.b(report, "Equalization_Parameters", "ghost_sites", False)
ghopen = rep.a(
    report,
    "Equalization_Parameters",
    "ghost_penalty",
    np.array([1, 1]),
    output_arr=True,
)
r = rep.b(report, "Equalization_Parameters", "random_initial_guess", False)
Utarget = rep.a(report, "Equalization_Parameters", "U_target", None, output_arr=True)
tTarget = rep.a(report, "Equalization_Parameters", "t_target", None)
Vtarget = rep.a(report, "Equalization_Parameters", "V_target", None, output_arr=True)
if any([Utarget is not None, tTarget is not None, Vtarget is not None]):
    if tTarget is None:
        txTarget, tyTarget = None, None
    elif len(tTarget) == 1:
        txTarget, tyTarget = tTarget[0], None
    elif len(tTarget) == 2:
        txTarget, tyTarget = tTarget
    else:
        raise ValueError(
            "t_target must be a single value/array or a tuple of two values/arrays for tx and ty."
        )
    target_values = (Vtarget, Utarget, txTarget, tyTarget)
else:
    target_values = None
sf = rep.f(report, "Equalization_Parameters", "scale_factor", None)
# Try to read existing equalization result as initial guess for next equalization
meth = "Nelder-Mead" if meth == "NM" else meth
# Try to read initial guess for equalization
if meth == "Nelder-Mead":
    # Try to read simplex first, then x0
    x0 = rep.a(
        report,
        "Equalization_Result",
        "simplex",
        rep.a(report, "Equalization_Result", "x", None),
    )
else:
    x0 = rep.a(report, "Equalization_Result", "x", None)
print("x0", x0)

# ====== Verbosity & Plotting ======
log = rep.b(report, "Verbosity", "write_log", False)
verb = rep.i(report, "Verbosity", "verbosity", 0)
# plot = rep.b(report, "Verbosity", "plot", False)
output_wf = rep.b(report, "Verbosity", "output_lowest_wannier", False)
# savefmt = rep.s(report, "Verbosity", "save_format", "ini")

# ====== Lattice parameters ======
lattice = Lattice(
    shape=shape,  # lattice geometries
    lattice_symmetry=ls,  # lattice reflection symmetry
    lattice=lsize,  # lattice size
    lc=lc,  # lattice constant in nm
    nodes=nodes,  # custom lattice site positions, in unit of lc
    links=links,  # custom lattice links
    isotropic=False,  # check if the lattice is isotropic
    ghost=gho,
    ghost_penalty=ghopen,
)

# ====== Equalize ======
G = HubbardEqualizer(
    N,
    R0=L0,
    dim=dimension,
    lattice=lattice,  # Lattice object
    custom_potential=custom_potential,  # Custom trapping potential
    ascatt=a_s,
    band=band,
    avg=avg,
    model=model,  # Trapping potetnial type
    trap=(V0, w),  # 2nd entry in array is (wx, wy), in number is (w, w)
    atom=m,  # Atom mass, in amu. Default Lithium-6
    laser=l,  # Laser wavelength
    zR=zR,  # Rayleigh range input by hand
    variable_waist=wd,  # Waist varying directions
    sparse=s,  # Sparse matrix
    zero_avgV=zero_avgV,  # Shift V to zero average
    equalize=eq,
    eqitem=eqt,
    balance_V0=balance_V0,  # Balance trap depths V0 for all traps first, useful for two-band calculation
    Ut=ut,
    target_values=target_values,  # U, t, V target values
    Nintgrl_grid=Nintgrl_grid,
    random=r,
    x0=x0,
    scale_factor=sf,
    eqmethod=meth,
    nobounds=nb,
    symmetry=symm,
    iofile=report,
    write_log=log,
    verbosity=verb,
)

# ====== Adjust Voff if just do Hubbard parameter calculation ======
if not eq:
    G.Voff = rep.a(report, "V_offset", "Trap_Adjustments", G.Voff)

eig_sol = G.eigen_basis()
__, __, WF = G.singleband_Hubbard(u=calculate_U, eig_sol=eig_sol)
maskedA = G.lattice.ghost.mask_quantity(G.A)
if calculate_U:
    maskedU = G.lattice.ghost.mask_quantity(G.U)
links = G.xy_links(G.lattice.ghost.links)

nnt = G.lattice.nn_tunneling(maskedA)
if G.sf == None:
    G.sf, __ = G.txy_target(nnt, links, np.mean)
# Print out Hubbard parameters
if G.verbosity > 1:
    print(f"scale_factor = {G.sf}")
    print(f"V = {np.diag(G.A)}")
    print(f"t = {abs(G.lattice.nn_tunneling(G.A))}")
    print(f"U = {G.U}")
# if plot:
#     G.draw_graph("adjust", A=G.A, U=G.U)
#     G.draw_graph(A=G.A, U=G.U)

# ====== Write singleband, trap and Wannier parameters ======
write_singleband(report, G)
write_wannier(report, G, output_wf, eig_sol[1][0], eig_sol[2][0], WF)
# Off-diagonal elements of U
if G.bands == 1 and calculate_U and offdiag_U:
    print("Singleband off-diagonal U calculation.")
    __, W, __, __ = G.multiband_WF(*eig_sol)
    U = interaction(G, W, *eig_sol[1:], onsite=False)[0][0]
    values = {"U_ijkl": U}
    rep.create_report(report, "Singleband_Parameters", **values)
write_trap_params(report, G)

# ====== Calculate Hubbard parameter variances ======
eqt = "uvt" if eqt == "neq" else eqt
u, t, v, __, __, __ = str_to_flags(eqt)
w = np.array([u, t, v])
if not target_values:
    Vtarget = np.mean(np.real(np.diag(maskedA)))
    tTarget = G.txy_target(nnt, links, np.mean)
else:
    Vtarget, Utarget, txTarget, tyTarget = target_values
    tTarget = [txTarget, tyTarget]
ct = G.t_cost_func(maskedA, links, tTarget, G.sf)
cv = G.v_cost_func(maskedA, Vtarget, G.sf)
if calculate_U:
    if not target_values:
        Utarget = np.mean(maskedU)
    cu = G.u_cost_func(maskedU, Utarget, G.sf)
else:
    Utarget = 0
    cu = 0
cvec = np.array((cu, ct, cv))
c = w @ cvec
cvec = np.sqrt(cvec)
fval = np.sqrt(c)
ctot = la.norm(cvec)
G.eqinfo["sf"] = G.sf
# Final U/t, so is determined by average values
G.eqinfo["Ut"] = np.max(Utarget) / np.min(tTarget[0])

if eq:
    G.eqinfo.update_cost(cvec, fval, ctot)
else:
    v0, __ = G.init_v0_and_bound(random=False)
    G.eqinfo.create_log(v0, (Vtarget, Utarget, *tTarget))
    G.eqinfo.update_cost(cvec, fval, ctot)
    G.eqinfo["success"] = False
    G.eqinfo["exit_status"] = -1
    G.eqinfo["termination_reason"] = "Not equalized"
G.eqinfo.write_equalization(report, write_log=log)

# if savefmt == "h5":
#     outFile = inFile[:-4] + ".h5"  # remove .ini and add .h5
#     tij = abs(G.A)
#     # remove diagonal elements, replace with V
#     tij += np.diag(np.diag(G.A) - np.diag(tij))
#     dat = {
#         "t_ij": tij,
#         "U_i": G.U,
#         "V_offset": G.Voff,
#         "trap_centers": G.trap_centers,
#         "wf_centers": G.wf_centers,
#         "wf_cost": G.wf_cost,
#         "total_cost_func": ctot,
#     }
#     with h5py.File(outFile, "w") as f:
#         print(f"Writing to h5 file {outFile} ...")
#         for k in dat.keys():
#             f[k] = np.asarray(dat[k])
#         print("Done!")

# ====== Write multiband output ======
if G.bands > 1:
    maskedA, W, wf_centers, wf_costs = G.multiband_WF(*eig_sol)
    values = {}
    for i in range(band):
        Vi = np.real(np.diag(maskedA[i]))
        tij = abs(np.real(maskedA[i] - np.diag(Vi)))
        values[f"t_{i+1}_ij"] = tij
        values[f"V_{i+1}_i"] = Vi
        values[f"wf_{i+1}_centers"] = wf_centers[i]
        values[f"wf_{i+1}_cost"] = wf_costs[i]

    if calculate_U:
        U = interaction(G, W, *eig_sol[1:])
        for i in range(band):
            for j in range(band):
                values[f"U_{i+1}{j+1}_i"] = U[i, j]

    rep.create_report(report, "Multiband_Parameters", **values)

sys.exit(0)  # Exit with no error
