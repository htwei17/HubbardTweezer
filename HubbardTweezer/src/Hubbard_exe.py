import numpy as np
import sys
from os.path import exists

from Hubbard.output import *
from Hubbard.plot import HubbardGraph
from Hubbard.equalizer import *
import tools.reportIO as rep


def help_message(s=2):
    # print help message and exit
    print('''
    Usage: python Hubbard_exe.py <input ini file path>
    
    The program will read [Parameters] setion in the input file
    and generate output setions in the same file. Detail see below.
    WARNING: multiple definitions of the same item will raise error.
    
    Items to input the file:
    ----------------------------------------
    
    [Parameters]
    DVR hyperparameters:
    N:  DVR half grid point number (default: 20)
    L0: DVR grid half-size in unit of x_waist (default: 3, 3, 7.2)
    dimensin:   DVR dimension (default: 1)

    DVR calculation settings:
    sparse: (optional) use sparse matrix or not (default: True)
    symmetry:   (optional) use symmetry in DVR calculation or not (default: True)

    Lattice parameters:
    lattice_size:   lattice size (default: 4,)
    lattice_constant:   lattice spacing in unit of nm
                        if one number eg. 1500, means a_x=a_y (default: 1520, 1690)
    shape:  lattice shape (default: square)
    lattice_symmetry:   use lattice reflection symmetry or not (default: True)
    
    Physical parameters:
    scattering_length:  scattering length in unit of a_0 (default: 1770)
    V_0:    trap depth in unit of kHz (default: 104.52)
    waist: xy waist in unit of nm (default: 1000, 1000)
    atom_mass:  atom mass in unit of amu (default: 6.015122)
    zR:    (optional) Rayleigh range in unit of nm
            None means calculated from laser wavelength (default: None)
    laser_wavelength:   laser wavelength in unit of nm (default: 780)
    average:    coefficient in front of trap depth, used for strobed trap (default: 1)
    
    Hubbard parameter calculation:
    band:   number of bands to calculate Hubbard parameters (default: 1)
    U_over_t:   Hubbard U/t ratio (default: None)
                None means avg U / avg t_x calculated in initial guess

    Hubbard parameter equalization:
    equalize:   equalize Hubbard parameters or not (default: False)
    equalize_target:    target Hubbard parameters to be equalized (default: vT)
                        see Hubbard.equalizer for more details
    method:     optimization algorithm to equalize Hubbard parameters (default: 'trf')
                see scipy.optimize.minimize and least_squares for more details
    no_bounds:  (optional) do not use bounds in optimization (default: False)
    random_initial_guess:   (optional) use random initial guess (default: False)
    scale_factor:   (optional) energy scale factor to make cost function dimensionless
                    None means avg t_x calculated in initial guess
                    in unit of kHz (default: None)
    write_log:  (optional) print parameters of every step to log file or not (default: False).
                See [Equalization_Log] in output file
    plot:   plot Hubbard parameter graphs or not (default: False)

    verbosity:  (optional) 0~3, print more information or not (default: 0)
    
    [Equalization_Info]
    x:  (optional) initial free trap parameters for equalization as 1D array
    
    
    Items output by the program:
    ----------------------------------------
    N is the number of sites. k is the number of bands.
    
    [Singleband_Parameters]
    The Hubbard parameters for the singleband Hubbard model, unit kHz.
    t_ij:   NxN array, tunneling matrix between sites i and j
    V_i:    Nx1 array, on-site potential at site i
    U_i:    Nx1 array, on-site Hubbard interaction at site i
    
    [Trap_Adjustments]
    The factors to adjust traps to equalize Hubbard parameters.
    V_offset:   Nx1 array, factor to scale trap depth, true depth = V_offset * V_0
    trap_centers:   Nx2 array, trap center position in unit of waist_x and waist_y
    waist_factors:  Nx2 array, factor to scale trap waist, true waist_x/y = waist_factors_x/y * waist_x/y
    
    [Equalization_Result]
    The equalization status and result.
    x:  optimized free trap parameters as minimization function input
    cost_func_by_terms:  cost function values C_U, C_t, C_V by terms of U, t, and V
    cost_func_value: cost function value feval to be minimized
                     feval = w_1 * C_U + w_2 * C_t + w_3 * C_V
    total_cost_func:    total cost function value C = C_U + C_t + C_V
    func_eval_number:   number of cost function evaluations
    scale_factor:   energy scale factor to make cost function dimensionless.
                    See scale_factor in [Parameters]
    success:    minimization success or not
    equalize_status:    minimization status given by scipy.optimize.minimize
    termination_reason: termination message given by scipy.optimize.minimize
    U_over_t:   Hubbard U/t ratio
    
    [Equalization_Log]
    (optional) log of equalization process.
    Each item is a list of values introduced in [Equalization_Result] in each step.
    
    [Multiband_Parameters]
    (optional) Hubbard parameters for the multiband Hubbard model, unit kHz.
    Each item is similar to [Singleband_Parameters] with band indices added.
    ''')
    sys.exit(s)


# ====== Read argument and file ======
try:
    inFile = sys.argv[1]
    # outFile = sys.argv[2]

    if inFile == '--help' or inFile == '-h':
        help_message(2)
    elif exists(inFile):
        report = rep.get_report(inFile)
    else:
        raise FileNotFoundError('Hubbard_exe: input file not found')
except FileNotFoundError as ferr:
    print(ferr)
    print("Usage: python Hubbard_exe.py <input ini file path>")
    print("use -h or --help for help")
    sys.exit(1)

# ====== DVR parameters ======
N = rep.i(report, "Parameters", "N", 20)
L0 = rep.a(report, "Parameters", "L0", np.array([3, 3, 7.2]))
dim = rep.i(report, "Parameters", "dimension", 1)

# ====== Create lattice ======
lattice = rep.a(report, "Parameters", "lattice_size",
                np.array([4])).astype(int)
lc = tuple(rep.a(report, "Parameters", "lattice_const",
                 np.array([1520, 1690])))
shape = rep.s(report, "Parameters", "shape", 'square')
ls = rep.b(report, "Parameters", "lattice_symmetry", True)

# ====== Physical parameters ======
a_s = rep.f(report, "Parameters", "scattering_length", 1000)
V0 = rep.f(report, "Parameters", "V_0", 104.52)
w = rep.a(report, "Parameters", "waist", np.array([1000, 1000]))
m = rep.f(report, "Parameters", "atom_mass", 6.015122)
zR = rep.f(report, "Parameters", "zR", None)
l = rep.f(report, "Parameters", "laser_wavelength", 780)
avg = rep.f(report, "Parameters", "average", 1)

# ====== Hubbard parameters ======
band = rep.i(report, "Parameters", "band", 1)
ut = rep.f(report, "Parameters", "U_over_t", None)

# ====== Equalization ======
eq = rep.b(report, "Parameters", "equalize", False)
eqt = rep.s(report, "Parameters", "equalize_target", 'vT')
wd = rep.s(report, "Parameters", "waist_direction", None)
meth = rep.s(report, "Parameters", "method", 'trf')
nb = rep.b(report, "Parameters", "no_bounds", False)
r = rep.b(report, "Parameters", "random_initial_guess", False)
sf = rep.f(report, "Parameters", "scale_factor", None)
log = rep.b(report, "Parameters", "write_log", False)
x0 = rep.a(report, "Equalization_Info", "x", None)

# ====== Plotting ======
plot = rep.b(report, "Parameters", "plot", False)

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
    lattice_symmetry=ls,
    Ut=ut,
    random=r,
    x0=x0,
    scale_factor=sf,
    method=meth,
    nobounds=nb,
    symmetry=symm,
    iofile=report,
    write_log=log,
    verbosity=verb)

eig_sol = eigen_basis(G)
G.singleband_Hubbard(u=True, eig_sol=eig_sol)
if plot:
    G.draw_graph('adjust', A=G.A, U=G.U)
    G.draw_graph(A=G.A, U=G.U)

# ====== Write output ======
write_singleband(report, G)
write_trap_params(report, G)

eqt = 'uvt' if eqt == 'neq' else eqt
u, t, v, __, __, __ = str_to_flags(eqt)
w = np.array([u, t, v])
nnt = G.nn_tunneling(G.A)
xlinks, ylinks, txTarget, tyTarget = G.xy_links(nnt)
if G.sf == None:
    G.sf = txTarget
ct = G.t_cost_func(G.A, (xlinks, ylinks), (txTarget, tyTarget))
cv = G.v_cost_func(G.A, None, G.sf)
cu = G.u_cost_func(G.U, None, G.sf)
cvec = np.array((cu, ct, cv))
c = w @ cvec
cvec = np.sqrt(cvec)
fval = np.sqrt(c)
ctot = la.norm(cvec)
G.eqinfo['sf'] = G.sf
G.eqinfo['Ut'] = np.mean(G.U) / txTarget

if eq:
    G.eqinfo['cost'] = np.append(G.eqinfo['cost'], cvec[None], axis=0)
    G.eqinfo['fval'] = np.append(G.eqinfo['fval'], fval)
    G.eqinfo['ctot'] = np.append(G.eqinfo['ctot'], ctot)
else:
    v0, __ = G.init_guess(random=False)
    G.eqinfo['x'] = v0[None]
    G.eqinfo['Nfeval'] = 0
    G.eqinfo['cost'] = cvec[None]
    G.eqinfo['fval'] = np.array([fval])
    G.eqinfo['ctot'] = np.array([ctot])
    G.eqinfo["success"] = False
    G.eqinfo["exit_status"] = -1
    G.eqinfo["termination_reason"] = "Not equalized"
write_equalization(report, G.eqinfo, write_log=log, final=True)

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

sys.exit(0)
