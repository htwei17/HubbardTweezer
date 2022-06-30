# DVR

![](https://img.shields.io/gitlab/v/release/Nottforestfc/DVR?color=brightgreen&sort=semver)

Repo for DVR full/sparse diagonalizaiton using 3d `x,y,z`-reflection symmetry

## Features:
* dynamics of on-off strobe potential
* solve Hubbard parameters of 2d arbitrary geometry

## Functioning scripts:
### DYR dynamics
* `DVR_core.py`: `DVR` base class to calculate DVR spectra
* `DVR_dynamics.py`: define `dynamics` class and `DVR_exe` function
* `DVR_output.py`: output .h5 file structure definitions
* `DVR_exe.py`: execute script of DVR dynamics on command line
### Hubbard parameters
* Prerequisite: [`pymanopt`](https://github.com/pymanopt/pymanopt)
* The code now supports square/rectangular, Lieb, triangular, honeycomb and kagome lattices
* `Hubbard_core.py` : `MLWF` class to construct maximally localized Wannier funcitons
* `Hubbard_plot.py` : `HubbardGraph` class to plot Hubbard parameters on lattice graphs
* `Hubbard_exe.py` : execute script to read inputs and write out Hubbard parameters for given lattice
### Test and executables
* `*.ipynb` are for test use. Most are self-explained by their title cells.

## TODO:
1. Find protocol to equalzie all the Hubbard parameters
2. Make able to equalzie Hubbard parameters for all lattice geometries
