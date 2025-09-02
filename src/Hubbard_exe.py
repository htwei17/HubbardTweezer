import numpy as np
import sys
import h5py
from os.path import exists

from HubbardTweezer.Hubbard.io import *
from HubbardTweezer.Hubbard.plot import HubbardGraph
from HubbardTweezer.Hubbard.equalizer import *
import HubbardTweezer.tools.reportIO as rep


def help_message(s=2):
    # print help message and exit
    print(
        """
    Usage: python Hubbard_exe.py <input ini file path>
    
    The program will read [Parameters] setion in the input file
    and generate output setions in the same file. Detail see below.
    WARNING: multiple definitions of the same item will raise error.
    
    # Items to input the file
    ---------------------------

    ## `[Parameters]`

    ### DVR hyperparameters:

    * N:  DVR half grid point number (default: 20)
    * L0: DVR grid half-size in unit of x_waist (default: 3, 3, 7.2)
    * DVR_dimension:   DVR dimension (default: 1)

    ### DVR calculation settings:

    * sparse: (optional) use sparse matrix or not (default: True)
    * symmetry:   (optional) use symmetry in DVR calculation or not (default: True)

    ### Lattice parameters:

    * lattice_size:   lattice size (default: 4,)
    * lattice_constant:   lattice spacing in unit of nm
                        if one number eg. 1500, means a_x=a_y (default: 1520, 1690)
    * shape:  lattice shape (default: square)
    * lattice_symmetry:   use lattice reflection symmetry or not (default: True)

    ### Physical parameters:

    * scattering_length:  scattering length in unit of a_0 (default: 1770)
    * V0:    trap depth in unit of kHz (default: 104.52)
    * waist: xy waist in unit of nm (default: 1000, 1000)
    * atom_mass:  atom mass in unit of amu (default: 6.015122)
    * zR:    (optional) Rayleigh range in unit of nm
            None means calculated from laser wavelength (default: None)
    * laser_wavelength:   laser wavelength in unit of nm (default: 780)
    * average:    coefficient in front of trap depth, meaning the actual trap depth = `average * V0` (default: 1)

    * Hubbard parameter calculation:
    * band:   number of bands to calculate Hubbard parameters (default: 1)
    * U_over_t:   Hubbard U/t ratio (default: None)
                None means avg U / avg t_x calculated in initial guess

    ### Hubbard parameter hyperparameters:

    * Nintgrl_grid:   number of grid points in integration (default: 257)

    ### Hubbard parameter equalization:

    * equalize:   equalize Hubbard parameters or not (default: False)
    * equalize_item:    determine which Hubbard parameters to be equalized (default: `vT`)
                        see `Hubbard.equalizer` for more details
    * method:     optimization algorithm to equalize Hubbard parameters (default: `trf`)
                see `scipy.optimize.minimize`, `least_squares`, and `nlopt` documentations for more details
    * no_bounds:  (optional) do not use bounds in optimization (default: False)
    * random_initial_guess:   (optional) use random initial guess (default: False)
    * scale_factor:   (optional) energy scale factor to make cost function dimensionless
                    None means avg t_x calculated in initial guess
                    in unit of kHz (default: None)
    * write_log:  (optional) print parameters of every step to log file or not (default: False).
                See `[Equalization_Log]` in output file
    * plot:   plot Hubbard parameter graphs or not (default: False)

    * verbosity:  (optional) 0~3, levels of how much information to print (default: 0)

    ## `[Equalization_Result]`

    * x:  (optional) initial free trap parameters for equalization as 1D array

    # Items output by the program
    ---------------------------

    Here N is the number of sites, and k is the number of bands.

    ## `[Singleband_Parameters]`

    The Hubbard parameters for the single-band Hubbard model, unit kHz.

    * t_ij:   NxN array, tunneling matrix between sites i and j
    * V_i:    Nx1 array, on-site potential at site i
    * U_i:    Nx1 array, on-site Hubbard interaction at site i
    * wf_centers:    Nx2 array, calculated Wannier orbital center positions

    ## `[Trap_Adjustments]`

    The factors to adjust traps to equalize Hubbard parameters.

    * V_offset:   Nx1 array, factor to scale trap depth, true depth = V_offset *V_0
    * trap_centers:   Nx2 array, trap center position in unit of waist_x and waist_y
    * waist_factors:  Nx2 array, factor to scale trap waist, true waist_x/y = waist_factors_x/y* waist_x/y

    ## `[Equalization_Result]`

    This section lists the equalization status and result.

    * x:  optimized free trap parameters as minimization function input
    * cost_func_by_terms:  cost function values C_U, C_t, C_V by terms of U, t, and V
    * cost_func_value: cost function value feval to be minimized
                        `feval = w_1 * C_U + w_2 * C_t + w_3 * C_V`
    * total_cost_func:    total cost function value `C = C_U + C_t + C_V`
    * func_eval_number:   number of cost function evaluations
    * scale_factor:   energy scale factor to make cost function dimensionless.
                    See scale_factor in `[Parameters]`
    * success:    minimization success or not
    * equalize_status:    minimization status given by scipy.optimize.minimize
    * termination_reason: termination message given by scipy.optimize.minimize
    * U_over_t:   Hubbard U/t ratio

    ## `[Equalization_Log]` (optional)

    Log of equalization process, turn on/off by `write_log`. Each item is an array of values introduced in `[Equalization_Result]`, which each row shows one step.

    ## `[Multiband_Parameters]` (optional)

    Multiband Hubbard parameters, unit kHz.
    Each item is similar to `[Singleband_Parameters]` with band indices added.
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
L0 = rep.a(report, "DVR_Parameters", "L0", np.array([3, 3, 7.2]))
dimension = rep.i(report, "DVR_Parameters", "DVR_dimension", 1)
s = rep.b(report, "DVR_Parameters", "sparse", True)
symm = rep.b(report, "DVR_Parameters", "DVR_symmetry", True)

# ====== Create lattice ======
model = rep.s(report, "Lattice_Parameters", "potential_model", "Gaussian")
custom_potential = None
if model in ["Gaussian", "optical_lattice"]:
    shape = rep.s(report, "Lattice_Parameters", "shape", "square")
    if model == "optical_lattice" and shape != "square":
        raise ValueError(
            "optical_lattice model only supports square lattice shape, please use Gaussian model for other shapes."
        )
    if shape == "custom":
        nodes = rep.a(report, "Lattice_Parameters", "site_locations", None)
        links = rep.a(report, "Lattice_Parameters", "bond_links", None)
    else:
        nodes = None
        links = None
        lsize = rep.a(
            report, "Lattice_Parameters", "lattice_size", np.array([4])
        ).astype(int)
        lc = tuple(
            rep.a(report, "Lattice_Parameters", "lattice_const", np.array([1520, 1690]))
        )
        ls = rep.b(report, "Lattice_Parameters", "lattice_symmetry", True)
elif model == "custom":
    model = "custom"
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
w = rep.a(report, "Trap_Parameters", "waist", np.array([1000, 1000]))
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
eqt = rep.s(report, "Equalization_Parameters", "equalize_item", "vT")
balance_V0 = rep.b(report, "Equalization_Parameters", "balance_V0", False)
wd = rep.s(report, "Equalization_Parameters", "waist_direction", None)
meth = rep.s(report, "Equalization_Parameters", "method", "trf")
nb = rep.b(report, "Equalization_Parameters", "no_bounds", False)
gho = rep.b(report, "Equalization_Parameters", "ghost_sites", False)
ghopen = rep.a(report, "Equalization_Parameters", "ghost_penalty", np.array([1, 1]))
r = rep.b(report, "Equalization_Parameters", "random_initial_guess", False)
Utarget = rep.a(report, "Equalization_Parameters", "U_target", None)
tTarget = rep.a(report, "Equalization_Parameters", "t_target", None)
Vtarget = rep.a(report, "Equalization_Parameters", "V_target", None)
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
plot = rep.b(report, "Verbosity", "plot", False)
output_wf = rep.b(report, "Verbosity", "output_wannier", False)
savefmt = rep.s(report, "Verbosity", "save_format", "ini")

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
G = HubbardGraph(
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
if output_wf:
    edge = 2
    x = np.linspace(
        lattice.trap_centers[0][0] - edge,
        lattice.trap_centers[-1][0] + edge,
        G.Nintgrl_grid,
    )
    y = np.linspace(
        lattice.trap_centers[0][1] - edge,
        lattice.trap_centers[-1][1] + edge,
        G.Nintgrl_grid,
    )
    z = 0
    wfval = wannier_func((x, y, z), WF, G, *eig_sol[1:])
if plot:
    G.draw_graph("adjust", A=G.A, U=G.U)
    G.draw_graph(A=G.A, U=G.U)

# ====== Write singleband and trap parameters ======
write_singleband(report, G)
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

if savefmt == "h5":
    outFile = inFile[:-4] + ".h5"  # remove .ini and add .h5
    tij = abs(G.A)
    # remove diagonal elements, replace with V
    tij += np.diag(np.diag(G.A) - np.diag(tij))
    dat = {
        "t_ij": tij,
        "U_i": G.U,
        "V_offset": G.Voff,
        "trap_centers": G.lattice.trap_centers,
        "wf_centers": G.wf_centers,
        "wf_cost": G.wf_cost,
        "total_cost_func": ctot,
    }
    with h5py.File(outFile, "w") as f:
        print(f"Writing to h5 file {outFile} ...")
        for k in dat.keys():
            f[k] = np.asarray(dat[k])
        print("Done!")

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
