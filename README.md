# DVR

DVR repo for DVR full/sparse diagonalizaiton using 3d reflection symmetry

Features:
* dynamics of on-off strobe potential
* solve Hubbard parameters of 2d arbitrary geometry

Functioning scripts:
* `DVR_full.py`: `DVR` base class to calculate DVR spectra
* `DVR_exe.py`: define `DVR_exe` function in `dynamics_exe.py`
* `dynamics.py`: `dynamics` class, functions to run dynamics
* `dynamics_exe.py`: execute script of DVR dynamics on command line
* `wannier.py` : `Wannier` class to construct Wannier funcitons
* `plot_Hubbard.py` : `Graph` class to plot Hubbard parameters on lattice graphs
* `*.ipynb` are for test use. Most are self-explained by their title cells.
