# HubbardTweezer Documentation

----------------------------------------

This is an introductory manual for the code on [github](https://github.com/htwei17/HubbardTweezer) and used on the calculations in the [paper](https://arxiv.org/abs/2306.03019). For scientific principles, please refer to the paper main text.

## Dependencies

In order to run the code, you need to install the following packages:

* `scipy` along with `numpy`
* [`pymanopt`](https://github.com/pymanopt/pymanopt) which depends on [`torch`](https://github.com/pytorch/pytorch)
* [`opt_einsum`](https://github.com/dgasmith/opt_einsum)
* [`nlopt`](https://github.com/stevengj/nlopt)
* [`ortools`](https://github.com/google/or-tools)
* `configobj`
<!-- * `pympler` used to monitor memory usage -->
<!-- * [`networkx`](https://github.com/networkx/networkx) which depends on `matplotlib` -->
<!-- * `h5py` -->

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

What the program does is to read parameters set in sections given at input e.g. `[DVR_Parameters]` for parameters of DVR calculation, and write calculation results to the output sections, such as `[Singleband_Parameters]` for single-band Hubbard parameters, `[Equalization_Result]` for equalization solutions and `[Trap_Adjustments]` for how the traps need to be adjusted in experiment to realize the desired Hubbard parameters.

### Data type: number, tuple and array

Specifically for the program to read a length=1 tuple, what needs to do is to add a comma after the number:

```ini
number = 2 # read as number 2
tuple = 2, # read as a tuple (2,)
```

And if we want to input a 1-D or n-D `numpy.array`, we use the following format:

```ini
1d_array = "[1, 2, 3]" # read as a 1-D array
2d_array = "[[1, 2], [3, 4]]" # read as a 2-D array
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

In this section, `Nsite` is the number of trap sites.

#### `[DVR_Parameters]`

* `N`:  (integer) number of DVR grid points from the outermost trap center to the box edges (default: `20`)
* `L0`:  (3-entry array) $x$, $y$ and $z$ direction distances from the outermost trap center to the box edges in unit of $x$ direction waist $w_x$ (default: `3, 3, 7.2`)
* `DVR_dimension`:   (integer) DVR grid spatial dimension (default: `1`)
<!-- * `sparse`:   (optional, bool) use sparse matrix (default: `True`) -->
<!-- * `DVR_symmetry`:   (optional) use reflection symmetries in DVR calculation (default: True) -->

<!-- > ##### Reflection symmetry
>
> If `DVR_symmetry` is `False`, the DVR Hamiltonian is solved without block-diagonalizing the reflection symmetry sectors. If `True`, it solves the DVR Hamiltonian in symmetry sectors specified by the properties `lattice_symmetry` and `band` defined later. -->

#### `[Lattice_Parameters]`

* `shape`:  (string) lattice shape.  
                    Supported strings: `square`, `Lieb`, `triangular`, `honeycomb`, `defecthoneycomb`, `kagome` and `custom` (default: `square`)
* `lattice_constant`:   (tuple or float) the $x$ and $y$ directions lattice spatial scaling, in unit of nm  
                    if `shape` is `custom`, it is the unit for `site_locations`  
                    if `shape` is not `custom`, it is lattice spacing  
                    if only one number is given e.g. `1500`, this means $a_x=a_y$ (default: `1520, 1690`)

If `shape` is not `custom`, the following parameter is read:

* `lattice_size`:  (tuple or integer) the number of traps in each lattice dimension  
                    if only one number is given, this means the lattice is a 1D chain (default: `4,`)

If `shape` is `custom`, the following two parameters are read:

* `site_locations`:  (`Nsite` x 2 array) trap centers in unit of `lattice_constant` (default: `None`)  
                     the `i`-th row is the `(x,y)` coordinate for the `i`-th trap site (`i=0,1,..., Nsite - 1`)  
* `bond_links`:      (number of bonds x 2 array) used in Hubbard parameter equalization to decide which pairs of sites' tunneling will be equalized (default: `None`)  
                     each row is a bond, i.e. link between a pair of sites `(i,j)`, with integers `i` and `j` trap site indices

The next parameter specifies whether to use lattice reflection symmetries in the DVR calculation. If this is enabled, only the `(x<=0, y<=0)` quadrant tweezer array parameters, including the trap center locations and the trap depths are used in the calculation. The other quadrants are overwritten by the copy of the `(x<=0, y<=0)` quadrant. Therefore, if the system is not reflection symmetric, please don't set to `True`.

* `lattice_symmetry`:   (bool) use lattice $x$- and $y$-reflection symmetry (default: `True`)

<!-- > ##### Reflection symmetry
>
> The program generates a list of reflection symmetry sectors for DVR calculation to solve the Hamiltonian, labeled by $x$, $y$ and $z$-reflection parities `[px,py,pz]` with `px`, `py`, `pz` each to be `1`, `-1` or `0`. `1` mean even-parity, `-1` means odd-parity, `0` means no reflection symmetry is used in this direction. The list is generated based on the values of properties `lattice_symmetry` and `band` defined later.
>
> `DVR_symmetry` determine if the reflection symmetries are used in DVR eigenstate calculation. `lattice_symmetry` determine if there are $x$ and $y$ reflection symmetries in the lattice used.
>
> If `DVR_symmetry` is `True`:
> `band` is `1`, then `pz` is fixed to `1`, meaning only even sector is calculated. If `band` is larger than `1`, then `pz=[1,-1]` are both calculated.
>
> If `DVR_symmetry` is `False`, then no matter `lattice_symmetry` is, no reflection symmetries are used in calculation. But if `DVR_symmetry` is `True`, then the reflection symmetry sectors are used based on `lattice_symmetry` and the bands to solve. In this case, if `lattice_symmetry` is `True`, then `px=[1,-1]` and `py=[1,-1]` are calculated. -->

#### `[Trap_Parameters]`

* `scattering_length`:  (float) scattering length in unit of Bohr radius $a_0$ (default: `1770`)
* `waist`:  (tuple or float) $x$ and $y$ direction waist ($w_x$, $w_y$) in unit of nm (default: `1000, 1000`)  
             if only one is set it means $w_x=w_y$
* `atom_mass`:  (float) atom mass in unit of amu (default: `6.015122`)
* `laser_wavelength`:   (float) laser wavelength in unit of nm (default: `780`)
* `zR`:    (tuple or float, optional) $x$ and $y$ direction Rayleigh range ($z_{R,x}$, $z_{R,y}$) in unit of nm (default: `None`)  
        `None` means calculated from `waist` and `laser_wavelength`
<!-- * `average`:    coefficient in front of trap depth, meaning the actual trap depth = `average * V0` (default: 1) -->

##### Set trap depths for each trap
>
> The trap depths of each trap is $\text{trap depth} = V_\text{offset} \times V_0$
> where $V_0$ is a number specifying the frequency scale and $V_\text{offset}$ is an array of scale factors of each trap. They are the two next parameters listed.

* `V0`:    (float) trap depth frequency scale in unit of kHz (default: 104.52)

<!-- <span id="input-in-trap_adjustment"></span> -->

#### input in `[Trap_Adjustment]`

* `V_offset`:   (`Nsite`-entry array) trap depth factors for each trap (default: `None`)  
                if `lattice_symmetry` is `True`, only the `(x<=0,y<=0)` quadrant of the lattice will be used, and the rest of the trap depths input will be overwritten  
                if `equalize` is `True`, `V_offset` information is overridden by `x`, see details in input in `[Equalization_Result]` [section](#input-in-equalization_result)  
                `None` means $V_\text{offset} = 1$ over the entire lattice
  
#### `[Hubbard_Settings]`

* `Nintgrl_grid`:   (integer) number of grid points in each dimension in trapezoidal numerical integration of $U$ (default: `200`)
* `band`:   (integer) number of bands to be calculated in Hubbard model (default: `1`)
* `offdiagonal_U`:   (bool) calculate multi-site interaction $U_{ijkl}$ (default: `False`)  
                     if it is `True`, it calculates and stores a tensor of $N_\text{site}^4$ elements  
                     only `band=1` is supported

<!-- <span id="equalization_parameters"></span> -->

#### `[Equalization_Parameters]`

For the following sections about equalization process, please refer to the [paper](https://arxiv.org/abs/2306.03019) for more details.

* `equalize`:   (bool) whether equalize Hubbard parameters or not (default: `False`)
* `equalize_target`:    (string) target Hubbard parameters to be equalized (default: `vT`)

##### Explain equalization target
>
> The expression of the equalization cost function is the Eq.(16) in the [paper](https://arxiv.org/abs/2306.03019), which is the squared difference from the calculated Hubbard parameters to the target values $\tilde{q}$. The `equalize_target` parameter specifies how the target values are determined for each kind of Hubbard parameters.
>
> 1. Lowercase `u`,`v`,`t`: the target values are changed to the average values of each kind of Hubbard parameter in each iteration of the equalization, meaning the program minimizes the sum of variances of all the Hubbard parameters  
> 2. Uppercase `U`, `V`, `T`: the target values are fixed by their values calculated by the initial physical trap parameters. The target values cannot be set by external input except that the $U/t$ ratio can be set by `U_over_t` parameter in the input in `[Equalization_Result]` [section](#input-in-equalization_result)  
>  i. For `U`, the target value is set to be the maximum value of the calculated Hubbard parameters by the initial physical trap parameters  
>   ii. For `T`, the target value is set to be the minimum value of the calculated Hubbard parameters by the initial physical trap parameters  
>   ii. Since the absolute value of `V` is not important, the case of `V` plays no effect  
> 3. Multiple letters can be used together, e.g. `uT` means to equalize `u` to uniform and `T` to target values determined by the initial physical trap parameters, while the uniformity of `V` is not considered

* `method`:    (string) optimization algorithm to equalize Hubbard parameters (default: `trf`)  
               available algorithms:
               implemented by `scipy.optimize`:`trf`, `Nelder-Mead`, `SLSQP`, `L-BFGS-B` and `cobyla`,
               implemented by `nlopt`: `praxis` and `bobyqa`
<!-- * `no_bounds`:  (optional) do not use bounds in optimization (default: False) -->
<!-- * `random_initial_guess`:   (optional) use random initial guess to equaliz (default: False) -->
* `scale_factor`:   (float, optional) energy scale factor to make cost function dimensionless  
                None means the smallest target value (see [explanation](#explain-equalization-target) above) calculated in initial guess  
                in unit of kHz (default: None)

##### Equalization proposal: adjust waist

* `waist_direction`:  (optional, string) direction of waist adjustment. `x`, `y`, `xy` are supported  
                    `None` means no waist adjustment (default: `None`)

##### Equalization proposal: ghost trap

`shape=custom` is not supported by ghost trap adjustment.

* `ghost_sites`:   (optional, bool) add ghost sites to the lattice (default: `False`)
* `ghost_penalty`: (optional, tuple) 2-entry tuple (factor, threshold) of the ghost penalty added to the cost function (default: `1, 1`)  
                 threshold is in unit of kHz

##### Explain ghost penalty
>
> ghost_penalty determines how the penalty is added to the equalization cost function. The formula is as below:
> $\mathrm{penalty} = \mathrm{factor} \times \exp[-6(q-\mathrm{threshold})]$

#### `[Verbosity]`

* `write_log`:  (optional, bool) print parameters of every step to the `[Equalization_Log]` of the `ini` file  (default: `False`)  
            see `[Equalization_Log]` in [output sections](#equalization_log-optional)
<!-- * `plot`:   plot Hubbard parameter graphs  (default: False) -->
* `verbosity`:  (optional, integer `0~3`) levels of how much information printed, `3` is the most detailed level, `0` means no printed information (default: `0`)

<!-- <span id="input-in-equalization_result"></span> -->

#### input in `[Equalization_Result]`

* `x`:  (optional, 1-D array) initial trap parameters for equalization as a 1-D array  
        used as the initial guess for equalization.
        The structure is `concatenate([V_offset, trap_centers, waist_factors])`
* `U_over_t`:   (float) target Hubbard $U/t$ ratio (default: `None`)  
                `None` means this value is calculated by the ratio of $\mathrm{avg} U / \mathrm{avg} t_x$ in initial guess

### Items output by the program

Here the integer `N` is the number of sites, and the integer `k` is the number of bands.

#### `[Singleband_Parameters]`

The Hubbard parameters for the single-band Hubbard model, unit kHz.

* `t_ij`:   (`N` x `N` array) tunneling matrix between sites `i` and `j`
* `V_i`:    (`N` x 1 array) on-site potential at site `i`
* `U_i`:    (`N` x 1 array) on-site Hubbard interaction at site `i`
* `U_ijkl`:   (`N` x `N` x `N` x `N` array) Hubbard interaction $U_{ijkl}$ among site `i`, `j`, `k` and `l`, calculated only if `offdiagonal_U=True`
* `wf_centers`:    (`N` x 2 array) calculated Wannier orbital center positions

#### output in `[Trap_Adjustment]`

The factors to adjust traps to equalize Hubbard parameters.

* `V_offset`:   (`N` x 1 array) factor to scale individual trap depth, the same item as in the [input section](#input-in-trap_adjustment)  
                resulting trap depth $V_\text{trap} = V_\text{offset} \times V_0$
* `trap_centers`:   (`N` x 2 array) trap center position in unit of `waist_x` and `waist_y`
* `waist_factors`:  (`N` x 2 array) factor to scale trap waist, resulting $x$ and $y$ waist <img src="https://github.com/htwei17/HubbardTweezer/blob/release/doc/wf.png" height="20" style="vertical-align: middle;">.
<!-- $w_{x,y} = \mathrm{waist\_factors}_{x,y} \times w_{x,y}$ -->

#### output in `[Equalization_Result]`

This section lists the equalization status and result. The definitions of $C_q$'s with $q=U$, $t$, or $V$ follow the Eq.(16) in the [paper](https://arxiv.org/abs/2007.02995) as below:
$$ C_q = \frac{1}{N_q \times \text{scale\_factor}}\sum_{i=1}^{N_q} \left(q_i - \tilde{q}\right)^2 $$
where $q_i$ is the Hubbard parameter at $i$-th site/bondand $N_q$ is the number of the parameters in one kind. $\tilde{q}$ is the target value of $q_i$'s, explained [here](#explain-equalization-target), and `scale_factor` is the smallest $\tilde{q}$ among all Hubbard parameters as explained in `[Equalization_Parameters]` [section](#equalization_parameters).

* `x`:  (1-D array) the optimal trap parameters to equalize Hubbard parameters, the same item as in the input part
* `cost_func_by_terms`:  (3-entry array) cost function values $C_U$, $C_t$, $C_V$ by terms of $U$, $t$, and $V$
* `cost_func_value`: (float) weighted cost function value `feval` to be minimized  
                    $\mathrm{feval} = w_1\times C_U + w_2\times C_t + w_3\times C_V$
* `total_cost_func`:    (float) equal-weighted total cost function value $C = C_U + C_t + C_V$
* `func_eval_number`:   (integer) number of cost function evaluations
* `scale_factor`:   (float) energy scale factor to make cost function dimensionless in unit of kHz.  
                See `scale_factor` in `[Equalization_Parameters]` [section](#equalization_parameters) for the definition of energy scale factor.
* `success`:    (bool) minimization success
* `equalize_status`:    (integer) termination status of the optimization algorithm
* `termination_reason`: (string) termination message given by the optimization algorithm
* `U_over_t`:   (float) $U/t$ ratio, the same item as in the [input section](#input-in-equalization_result)

<!-- <span id="equalization_log-optional"></span> -->

#### `[Equalization_Log]` (optional)

Log of equalization process, turn on/off by `write_log`. Each item is an array of values introduced in `[Equalization_Result]` bearing the same key name, of which each row refers to one iteration step.

#### `[Multiband_Parameters]` (optional)

Multiband Hubbard parameters in unit of kHz, turn on if `band > 1`. Parameters have the same format as in `[Singleband_Parameters]`, labeled by band index.

For example, `t_1_ij` is the tunneling matrix between sites `i` and `j` for the 1st band, and `U_12_i` is the on-site Hubbard interaction at site `i` between 1st and 2nd bands.

## Code structure

The code consists of two modules `DVR` and `Hubbard`. Their main modules are explained below.

1. `DVR`: DVR spectra calculations
   * `DVR.core`: `DVR` base class to calculate DVR spectra
   * `DVR.const`: constants used in DVR calculations
   * `DVR.wavefunc`: wavefunction calculations
   <!-- * `DVR.dynamics`: define `dynamics` class and `DVR_exe` function -->
   <!-- * `DVR.output`: output storage `.h5` file interal structure definitions -->
   <!-- * `DVR_exe.py`: execute script of DVR dynamics on command line -->

2. `Hubbard`: Hubbard parameter calculations
   * `Hubbard.core` : `MLWF` class to construct maximally localized Wannier funcitons (MLWFs)
   * `Hubbard.equalizer` : `HubbardParamEqualizer` class to equalize Hubbard parameters over all lattice sites
   * `Hubbard.riemann`: functions for Riemannian manifold optimization in constructing MLWFs
   * `Hubbard.eqinit`: functions to initialize trap parameters for equalization
   * `Hubbard.io`: logger and functions to read and write Hubbard parameters in equalization
   * `Hubbard.lattice`: `Lattice` class to define lattice geometry
   * `Hubbard.ghost`: `GhostTrap` class to add ghost traps to the lattice
   <!-- * `Hubbard.plot`: `HubbardGraph` class to plot Hubbard parameters on lattice graphs -->

3. `tools`: tools for data analysis
   * `tools.integrate`: functions to calculate 3D numerical integrals
   * `tools.point_match`: function to match and label MLWFs to the traps
   * `tools.reportIO`: functions to read and write `ini` files

4. `Hubbard_exe.py` : execute script to read inputs and write out Hubbard parameters for given lattice

<!-- ## `Hubbard.plot`

`Hubbard.plot` is the submodule to print and save Hubbard parameter graphs. -->
