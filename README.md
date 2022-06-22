# DVR

DVR repo for DVR full/sparse diagonalizaiton using 3d reflection symmetry

### Features:
* dynamics of on-off strobe potential
* solve Hubbard parameters of 2d arbitrary geometry

### Functioning scripts:
* `DVR_core.py`: `DVR` base class to calculate DVR spectra
* `DVR_dynamics.py`: define `dynamics` class and `DVR_exe` function
* `DVR_exe.py`: execute script of DVR dynamics on command line
* `Hubbard_core.py` : `MLWF` class to construct maximally localized Wannier funcitons
* `Hubbard_plot.py` : `HubbardGraph` class to plot Hubbard parameters on lattice graphs
* `*.ipynb` are for test use. Most are self-explained by their title cells.
