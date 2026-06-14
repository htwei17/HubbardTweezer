"""Public Hubbard API."""

from .core import MLWF, interaction, singleband_interaction, site_sort, wannier_func
from .grid import LatticeGrid
from .io import (
    EqualizeInfo,
    EqulizeInfo,
    read_file,
    read_Hubbard,
    read_target,
    read_trap,
    read_trap_params,
    update_saved_data,
    update_tc,
    write_singleband,
    write_trap_params,
    write_wannier,
)
from .lattice import Lattice

__all__ = [
    "EqualizeInfo",
    "EqulizeInfo",
    "Lattice",
    "LatticeGrid",
    "MLWF",
    "interaction",
    "read_file",
    "read_Hubbard",
    "read_target",
    "read_trap",
    "read_trap_params",
    "singleband_interaction",
    "site_sort",
    "update_saved_data",
    "update_tc",
    "wannier_func",
    "write_singleband",
    "write_trap_params",
    "write_wannier",
]

try:
    from .equalizer import HubbardEqualizer
except ImportError:
    HubbardEqualizer = None
else:
    __all__.append("HubbardEqualizer")
