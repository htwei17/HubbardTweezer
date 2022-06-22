# DVR

DVR repo for DVR full/sparse diagonalizaiton using 3d reflection symmetry

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
Note: `pymanopt' is needed. However, its current version doesn't support unitary matrix optimization (pending addition into 2.x.x version). So a self-made modified version based on v0.26 is attached.
* `Hubbard_core.py` : `MLWF` class to construct maximally localized Wannier funcitons
* `Hubbard_plot.py` : `HubbardGraph` class to plot Hubbard parameters on lattice graphs
* `Hubbard_exe.py` : execute script to read inputs and write out Hubbard parameters for given lattice
### Test and executables
* `*.ipynb` are for test use. Most are self-explained by their title cells.
