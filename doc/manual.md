# HubbardTweezer Documentation

----------------------------------------

Here is the introductory manual for the code in [paper]() TODO:fill in the link.

For scientific principles, please refer to the paper main text.

## Dependencies

In order to run the code, you need to install the following packages:

* [`pymanopt`](https://github.com/pymanopt/pymanopt) which depends on [`torch`](https://github.com/pytorch/pytorch)
* [`networkx`](https://github.com/networkx/networkx) which depends on `matplotlib`
* [`opt_einsum`](https://github.com/dgasmith/opt_einsum)
* `scipy` along with `numpy`
* `ortools`
* `configobj`
* `pympler`
* `h5py`

## Get started on HubbardTweezer

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

## Example

### Calculate single-band Hubbard parameters for a 2x2 square lattice

We write the input `2x2.ini` file as below:

```ini
[DVR_Parameters]
N = 20
L0 = 3, 3, 7.2
DVR_dimension = 3
[Trap_Parameters]
V0 = 52.26
waist = 1000,
scattering_length = 1770
laser_wavelength = 780
[Lattice_Parameters]
shape = square
lattice_size = 2, 2
lattice_const = 1550, 1600
lattice_symmetry = True
[Equalization_Parameters]
equalize = False
[Verbosity]
write_log = False
verbosity = 3
```

Then we run the command (make sure give the program correct paths):

```shell
python Hubbard_exe.py 2x2.ini
```

After calculation result is given by appending contents to the end of `2x2.ini` file:

```ini
[Singleband_Parameters]
t_ij = "[[0.0, 0.1864950982522663, 0.26852018226162067, 0.003420732961071802], [0.18649509825226715, 0.0, 0.00342073295455529, 0.2685201822611937], [0.26852018226161994, 0.0034207329545554283, 0.0, 0.1864950982516911], [0.0034207329610705987, 0.268520182261194, 0.18649509825169167, 0.0]]"
V_i = "[-9.315215265814913e-12, 6.394884621840902e-14, 9.301004411099711e-12, -5.684341886080802e-14]"
U_i = "[1.2134538518651798, 1.2134538518666564, 1.2134538518681128, 1.2134538518666398]"
wf_centers = "[[-0.725424845124012, -0.7620829309237793], [-0.7254248451240122, 0.7620829309237923], [0.7254248451240126, -0.7620829309238053], [0.7254248451240124, 0.7620829309237923]]"
[Trap_Adjustments]
V_offset = "[1.0, 1.0, 1.0, 1.0]"
trap_centers = "[[-0.775, -0.7999999999999999], [-0.775, 0.7999999999999999], [0.775, -0.7999999999999999], [0.775, 0.7999999999999999]]"
waist_factors = "[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]"
[Equalization_Result]
x = "[1.0, -0.775, -0.7999999999999999]"
cost_func_by_terms = "[5.560385638393359e-12, 1.9205510638480715e-12, 3.52929827971391e-11]"
cost_func_value = 3.534519983120769e-11
total_cost_func = 3.5779897142888145e-11
func_eval_number = 0
U_target = 1.2134538518666473
t_target = 0.18649509825197869, 0.26852018226140717
V_target = -1.7763568394002505e-15
scale_factor = 0.1864950982516911
success = False
equalize_status = -1
termination_reason = Not equalized
U_over_t = 6.506625982346819
```

The other sections in the file are not what we are interested in.

### Example 2: Equalize Hubbard parameters for a 4-site chain

Here we want to equalize Hubbard parameters for a 4-site chain by `trf` optimization algorithm in `scipy`, without using ghost trap or waist tuning. The input file `4x1_eq.ini` is as below:

```ini
[DVR_Parameters]
N = 20
L0 = 3, 3, 7.2
DVR_dimension = 3
[Trap_Parameters]
V0 = 52.26
waist = 1000,
scattering_length = 1770
laser_wavelength = 780
[Lattice_Parameters]
shape = square
lattice_size = 4,
lattice_const = 1550,
lattice_symmetry = True
[Equalization_Parameters]
equalize = True
equalize_target = UvT
waist_direction = None
U_over_t = None
method = trf
no_bounds = False
[Verbosity]
write_log = False
verbosity = 3
```

The main difference is in `[Equalization_Parameters]` section. By running the same command as above
```shell
python Hubbard_exe.py 4x1_eq.ini
```
we get the result:

```ini
[Equalization_Result]
x = "[1.027480937300892, 1.0083650796908388, -2.354612841530081, -0.7875721977039343]"
cost_func_by_terms = "[0.25916330530145887, 0.07763234841109246, 0.010500917429445717]"
cost_func_value = 0.27074465756771354
total_cost_func = 0.27074465756771354
func_eval_number = 40
U_target = 1.3045697992761218
t_target = 0.20750237974990482, None
V_target = 0
scale_factor = 0.20750237974990482
success = True
equalize_status = 2
termination_reason = `ftol` termination condition is satisfied.
U_over_t = 6.653609207877738
[Trap_Adjustments]
V_offset = "[1.027480937300892, 1.0083650796908388, 1.0083650796908388, 1.027480937300892]"
trap_centers = "[[-2.354612841530081, -0.0], [-0.7875721828027731, -0.0], [0.7875721828027731, -0.0], [2.354612841530081, -0.0]]"
waist_factors = "[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]"
[Singleband_Parameters]
t_ij = "[[0.0, 0.2069326134671067, 0.007771384980274503, 0.0005508604690104194], [0.20693261346711186, 0.0, 0.17276048246199027, 0.0077713849802729185], [0.007771384980273734, 0.17276048246199152, 0.0, 0.20693261346710737], [0.0005508604690078773, 0.007771384980273034, 0.2069326134671076, 0.0]]"
V_i = "[0.002178965356165463, -0.002178965356165463, -0.0021789653561725686, 0.002178965356165463]"
U_i = "[1.3548364097993686, 1.2472824046115605, 1.2472824046115616, 1.3548364097993688]"
wf_centers = "[[-2.3110489105373313, 0.0], [-0.7903690147813551, 0.0], [0.790369014781355, 0.0], [2.3110489105373317, 0.0]]"
```

## Parameter definitions

### Items to input the file

#### `[DVR_Parameters]`

* `N`:  DVR grid point number parameter from the outermost trap center to the box edges (default: 20)
* `L0`: DVR grid half-size in unit of $x$ direction waist $w_x$ (default: 3, 3, 7.2)
* `DVR_dimension`:   DVR grid spatial dimension (default: 1)
* `sparse`: (optional) use sparse matrix or not (default: True)
* `DVR_symmetry`:   (optional) use reflection symmetry sector in DVR calculation or not (default: True)

> ##### Explain reflection symmetry
>
>The property above determines whether to use reflection symmetry in DVR calculation. If `True`, DVR Hamiltonian is solved in several reflection symmetry sectors. If `False`, DVR Hamiltonian is solved in the whole space at a time. The information of which reflection symmetries are used is in `[Lattice_Parameters]` section.

#### `[Lattice_Parameters]`

* `lattice_size`:  number of traps in each lattice dimension (default: 4,)
* `lattice_constant`:   lattice spacing in unit of nm
                    if one number eg. 1500, means $a_x=a_y$ (default: 1520, 1690)
* `shape`:  lattice shape. Supported values: `square`, `Lieb`, `triangular`, `honeycomb`, `defecthoneycomb` and `kagome` (default: `square`)
* `lattice_symmetry`:   use lattice reflection symmetry or not (default: True)

> ##### Difference from lattice_symmetry to DVR_symmetry
>
> `DVR_symmetry` determine if these symmetries are used in DVR eigenstate calculation. And `lattice_symmetry` determine if there are $x$ and $y$ reflection symmetries in the lattice.
> 
> With `DVR_symmetry` to be `True`, if only lowest band is solved, then the $z$-even sector is calculated. But if more than one band is calculated, then the $z$-odd sector is also calculated.
>
> If `DVR_symmetry` is `False`, then no matter `lattice_symmetry` is `True` or not, no symmetries are used in calculation. But if `DVR_symmetry` is `True`, then the reflection symmetry sectors are used based on `lattice_symmetry` and the bands to solve. In this case, if `lattice_symmetry` is `True`, the $x$ and $y$ reflection symmetry sectors are used in the calculation, no matter what lattice shape is.

#### `[Trap_Parameters]`

* `scattering_length`:  scattering length in unit of Bohr radius $a_0$ (default: 1770)
* `V0`:    trap depth in unit of kHz (default: 104.52)
* `waist`: (w_x, w_y) waist tuple in unit of nm. If only one is set it means w_x=w_y (default: 1000, 1000)
* `atom_mass`:  atom mass in unit of amu (default: 6.015122)
* `zR`:    (optional) (zR_x, zR_y) Rayleigh range tuple in unit of nm
        None means calculated from laser wavelength (default: None)
* `laser_wavelength`:   laser wavelength in unit of nm (default: 780)
<!-- * `average`:    coefficient in front of trap depth, meaning the actual trap depth = `average * V0` (default: 1) -->

#### `[Hubbard_Settings]`

* `Nintgrl_grid`:   number of grid points in integration (default: 200)
* `band`:   number of bands to be calculated in Hubbard model (default: 1)
* `U_over_t`:   Hubbard $U/t$ ratio (default: None)
            None means $\mathrm{avg} U / \mathrm{avg} t_x$ calculated in initial guess

#### input in `[Trap_Adjustment]` section

* `V_offset`:   factor to scale trap depth, true depth = $V_\text{offset} \times V_0$
                    Only used when `equalize` is `False`
                    For `equalize` is `True`, use `x` as input, see details below
                    None means $V_\text{offset} = 1$ over the entire lattice (default: None)

#### `[Equalization_Parameters]`

* `equalize`:   equalize Hubbard parameters or not (default: False)
* `equalize_target`:    target Hubbard parameters to be equalized (default: `vT`)
                    see `Hubbard.equalizer` for more details

> ##### Explain equalization target
>
> 1. `u`,`v`,`t`: Hubbard parameters to equalize without setting target values, meaning the program minimizes the variance of Hubbard parameters
> 2. `U`, `V`, `T`: Hubbard parameters to equalize to target values, meaning the program minimizes the difference between Hubbard parameters and target values

* `method`:     optimization algorithm to equalize Hubbard parameters (default: `trf`)
            see `scipy.optimize.minimize`, `scipy.optimize.least_squares`, and `nlopt` documentations for more details
* `no_bounds`:  (optional) do not use bounds in optimization (default: False)
* `random_initial_guess`:   (optional) use random initial guess (default: False)
* `scale_factor`:   (optional) energy scale factor to make cost function dimensionless
                None means $\min t$ calculated in initial guess
                in unit of kHz (default: None)

##### Equalization proposal: adjust waist

* `waist_direction`:  (optional) direction of waist adjustment. `x`, `y`, `xy` are supported
                    None means no waist adjustment (default: None)

##### Equalization proposal: ghost trap

* `ghost_sites`:   (optional) add ghost sites to the lattice or not (default: False)
* `ghost_penalty`: (optional) 2-entry tuple (penalty, threshold) to determine the ghost penalty added to the cost function
                 threshold is in unit of kHz (default: 1, 1)

> ghost_penalty determines how the penalty is added to the equalization cost function. The formula is as below:
> $\mathrm{penalty} = \mathrm{factor}\times \exp\{-6(q-\mathrm{threshold})\}$

#### `[Verbosity]`

* `write_log`:  (optional) print parameters of every step to log file or not (default: False).
            See `[Equalization_Log]` in output file
* `plot`:   plot Hubbard parameter graphs or not (default: False)
* `verbosity`:  (optional) 0~3, levels of how much information to print (default: 0)

#### input in `[Equalization_Result]` section

* `x`:  (optional) initial trap parameters for equalization as 1D array
            Used as initial guess for equalization

### Items output by the program

Here N is the number of sites, and k is the number of bands.

#### `[Singleband_Parameters]`

The Hubbard parameters for the single-band Hubbard model, unit kHz.

* `t_ij`:   NxN array, tunneling matrix between sites i and j
* `V_i`:    Nx1 array, on-site potential at site i
* `U_i`:    Nx1 array, on-site Hubbard interaction at site i
* `wf_centers`:    Nx2 array, calculated Wannier orbital center positions

#### `[Trap_Adjustments]`

The factors to adjust traps to equalize Hubbard parameters.

* `V_offset`:   Nx1 array, factor to scale trap depth, true depth = $V_\text{offset} \times V_0$
* `trap_centers`:   Nx2 array, trap center position in unit of waist_x and waist_y
* `waist_factors`:  Nx2 array, factor to scale trap waist, true waist_x/y = waist_factors_x/y* waist_x/y

#### output in `[Equalization_Result]`

This section lists the equalization status and result.

* `x`:  optimized free trap parameters as minimization function input
* `cost_func_by_terms`:  cost function values $C_U$, $C_t$, $C_V$ by terms of $U$, t, and V
* `cost_func_value`: cost function value feval to be minimized
                    $\mathrm{feval} = w_1\times C_U + w_2\times C_t + w_3\times C_V$
* `total_cost_func`:    total cost function value $C = C_U + C_t + C_V$
* `func_eval_number`:   number of cost function evaluations
* `scale_factor`:   energy scale factor to make cost function dimensionless.
                See scale_factor in `[Parameters]`
* `success`:    minimization success or not
* `equalize_status`:    minimization status given by scipy.optimize.minimize
* `termination_reason`: termination message given by scipy.optimize.minimize
* `U_over_t`:   Hubbard $U/t$ ratio

#### `[Equalization_Log]` (optional)

Log of equalization process, turn on/off by `write_log`. Each item is an array of values introduced in `[Equalization_Result]`, which each row shows one step.

#### `[Multiband_Parameters]` (optional)

Multiband Hubbard parameters, unit kHz.
Each item is similar to `[Singleband_Parameters]` with band indices added.

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
