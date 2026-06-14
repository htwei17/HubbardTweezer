"""Public package API for HubbardTweezer."""

from importlib import import_module as _import_module

__version__ = "0.6.0_dev"

DVR = _import_module(".DVR", __name__)
Hubbard = _import_module(".Hubbard", __name__)
tools = _import_module(".tools", __name__)

DVRBase = DVR.DVR
DVRConfig = DVR.DVRConfig
DVRGridMetadata = DVR.DVRGridMetadata
DVRMetadata = DVR.DVRMetadata
DVRPhysicsMetadata = DVR.DVRPhysicsMetadata
TrapParameters = DVR.TrapParameters
A0 = DVR.A0
AMU = DVR.AMU
DIM = DVR.DIM
h = DVR.h
psi = DVR.psi
psi_from_grid = DVR.psi_from_grid

Lattice = Hubbard.Lattice
LatticeGrid = Hubbard.LatticeGrid
MLWF = Hubbard.MLWF
wannier_func = Hubbard.wannier_func

__all__ = [
    "__version__",
    "DVR",
    "Hubbard",
    "tools",
    "DVRBase",
    "DVRConfig",
    "DVRGridMetadata",
    "DVRMetadata",
    "DVRPhysicsMetadata",
    "A0",
    "AMU",
    "DIM",
    "psi",
    "psi_from_grid",
    "h",
    "Lattice",
    "LatticeGrid",
    "MLWF",
    "TrapParameters",
    "wannier_func",
]

if getattr(Hubbard, "EqualizeInfo", None) is not None:
    EqualizeInfo = Hubbard.EqualizeInfo
    __all__.append("EqualizeInfo")

if getattr(Hubbard, "EqulizeInfo", None) is not None:
    EqulizeInfo = Hubbard.EqulizeInfo
    __all__.append("EqulizeInfo")

if getattr(Hubbard, "HubbardEqualizer", None) is not None:
    HubbardEqualizer = Hubbard.HubbardEqualizer
    __all__.append("HubbardEqualizer")
