# HubbardTweezer

![](https://img.shields.io/github/v/release/Kvanti17/HubbardTweezer?color=green&include_prereleases)

Repo for Hubbard parameter calculations in 1&2D optical tweezer array systems.

## Features:
* dynamics of on-off strobe potential
* solve Hubbard parameters of 2d arbitrary geometry

## Functioning scripts:
### DYR dynamics
* `DVR/core.py`: `DVR` base class to calculate DVR spectra
* `DVR/dynamics.py`: define `dynamics` class and `DVR_exe` function
* `DVR/output.py`: output .h5 file structure definitions
* `DVR_exe.py`: execute script of DVR dynamics on command line
### Hubbard parameters
* Prerequisite: [`pymanopt`](https://github.com/pymanopt/pymanopt)
* The code now supports square/rectangular, Lieb, triangular, honeycomb and kagome lattices
* `Hubbard/core.py` : `MLWF` class to construct maximally localized Wannier funcitons
* `Hubbard/plot.py` : `HubbardGraph` class to plot Hubbard parameters on lattice graphs
* `Hubbard_exe.py` : execute script to read inputs and write out Hubbard parameters for given lattice
### Test and executables
* `*.ipynb` are for test use. Most are self-explained by their title cells.

## TODO:
1. Find protocol to equalzie all the Hubbard parameters
2. Test Hubbard parameter calculations for all lattice geometries
