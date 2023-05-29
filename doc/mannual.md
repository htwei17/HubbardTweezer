# Get started on HubbardTweezer

----------------------------------------

In general, the code is run by feeding an `ini` file in which required parameters are set.

```shell
python Hubbard_exe.py parameters.ini
```

The code calculates, and generates results in the same `ini` file.

## `ini` file structure

`ini` is a text file format. It has a section-property structure. In one file it has one or more sections. In every section, key-value pairs provide properties:

```ini
[Section A]
key1 = value1
key2 = value2
...
[section B]
...
```

What the program does is to read parameters set in `[Parameters]` section and write calculation results to the other sections, such as `[Singleband_Parameters]` for single-band Hubbard parameters, `[Equalization_Result]` for equalization solutions and `[Trap_Adjustments]` for how the traps need to be adjusted in experiment to realize the desired Hubbard parameters.

### Data type: number vs array

Specifically for the program to read a length=1 array, what needs to do is to add a comma after the number:

```ini
number = 2 # 2 read as number
tuple = 2, # (2,) read as a tuple
```

## Parameter definitions

### Items to input the file

#### `[Parameters]`

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

Hubbard parameter hyperparameters:
Nintgrl_grid:   number of grid points in integration (default: 257)

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

#### `[Equalization_Info]`

x:  (optional) initial free trap parameters for equalization as 1D array

### Items output by the program

N is the number of sites. k is the number of bands.

#### `[Singleband_Parameters]`

The Hubbard parameters for the singleband Hubbard model, unit kHz.
t_ij:   NxN array, tunneling matrix between sites i and j
V_i:    Nx1 array, on-site potential at site i
U_i:    Nx1 array, on-site Hubbard interaction at site i

#### `[Trap_Adjustments]`

The factors to adjust traps to equalize Hubbard parameters.
V_offset:   Nx1 array, factor to scale trap depth, true depth = V_offset *V_0
trap_centers:   Nx2 array, trap center position in unit of waist_x and waist_y
waist_factors:  Nx2 array, factor to scale trap waist, true waist_x/y = waist_factors_x/y* waist_x/y

#### `[Equalization_Result]`

The equalization status and result.
x:  optimized free trap parameters as minimization function input
cost_func_by_terms:  cost function values C_U, C_t, C_V by terms of U, t, and V
cost_func_value: cost function value feval to be minimized
                    feval = w_1 *C_U + w_2* C_t + w_3 * C_V
total_cost_func:    total cost function value C = C_U + C_t + C_V
func_eval_number:   number of cost function evaluations
scale_factor:   energy scale factor to make cost function dimensionless.
                See scale_factor in [Parameters]
success:    minimization success or not
equalize_status:    minimization status given by scipy.optimize.minimize
termination_reason: termination message given by scipy.optimize.minimize
U_over_t:   Hubbard U/t ratio

#### `[Equalization_Log]`

(optional) log of equalization process.
Each item is a list of values introduced in [Equalization_Result] in each step.

#### `[Multiband_Parameters]`

(optional) Hubbard parameters for the multiband Hubbard model, unit kHz.
Each item is similar to [Singleband_Parameters] with band indices added.

## Example

### Calculate single-band Hubbard parameters for a 2x2 square lattice

We write the input `2x2.ini` file as below:

```ini
[Parameters]
N = 15
L0 = 3, 3, 7.2
lattice_size = 2, 2
lattice_const = 1550, 1600
V0 = 52.26
waist = 1000,
laser_wavelength = 780
shape = square
scattering_length = 1770
dimension = 3
lattice_symmetry = True
equalize = False
equalize_target = UvT
U_over_t = None
method = neq
random_initial_guess = True
ghost_sites = False
waist_direction = None
write_log = True
no_bounds = False
verbosity = 3
```

Then we run the command (make sure give the program correct paths):

```shell
python Hubbard_exe.py 2x2.ini
```

After calculation result is given by appending contents to the end of `2x2.ini` file:

```ini
[Singleband_Parameters]
t_ij = "[[0.0, 0.18649499553656998, 0.26852005133458734, 0.00342070692730814], [0.18649499553656942, 0.0, 0.0034207662976130366, 0.2685200513997752], [0.26852005133459006, 0.0034207662976127227, 0.0, 0.18649499532128158], [0.0034207069273072236, 0.26852005139977375, 0.18649499532128272, 0.0]]"
V_i = "[3.4677462679155724e-08, -5.102112510257939e-08, -3.122281810874483e-08, 4.7566487637595856e-08]"
U_i = "[1.213458140903253, 1.2134581273579168, 1.2134581305101972, 1.2134581428885043]"
wf_centers = "[[-0.7254249258931982, -0.7620829552187119], [-0.7254249258811576, 0.7620829551109817], [0.7254249258875868, -0.7620829551332057], [0.7254249258867685, 0.7620829552409356]]"
```

The other sections in the file are not what we are interested.

### Equalize Hubbard parameters for a 4-site chain

Here we want to equalize Hubbard parameters for a 4-site chain by `trf` optimization algorithm in `scipy`, without using ghost trap or waist tuning. The input file `4x1_eq.ini` is as below:

```ini
[Parameters]
N = 20
L0 = 3, 3, 7.2
lattice_size = 4,
lattice_const = 1550,
V0 = 52.26
waist = 1000,
laser_wavelength = 780
shape = square
scattering_length = 1770
dimension = 3
lattice_symmetry = True
equalize = True
equalize_target = UvT
waist_direction = None
U_over_t = None
method = trf
write_log = True
no_bounds = False
verbosity = 3
```

## Code structure

The code consists of two modules `DVR` and `Hubbard`. Their main modules are explained below.

1. `DVR`: DYR dynamics

* `DVR.core`: `DVR` base class to calculate DVR spectra
* `DVR.dynamics`: define `dynamics` class and `DVR_exe` function
* `DVR.output`: output storage `.h5` file interal structure definitions
* `DVR_exe.py`: execute script of DVR dynamics on command line

1. `Hubbard`: Hubbard parameter calculations

* The code now supports square/rectangular, Lieb, triangular, honeycomb (defect honeycomb) and kagome lattices
* `Hubbard.core` : `MLWF` class to construct maximally localized Wannier funcitons
* `Hubbard.equalizer` : `HubbardParamEqualizer` class to equalize Hubbard parameters over all lattice sites
* `Hubbard.plot`: `HubbardGraph` class to plot Hubbard parameters on lattice graphs
* `Hubbard_exe.py` : execute script to read inputs and write out Hubbard parameters for given lattice

## `Hubbard.plot`

`Hubbard.plot` is the submodule to print and save Hubbard parameter graphs.
