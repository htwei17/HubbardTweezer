"""Structured configuration and metadata for DVR-based solvers."""

from dataclasses import dataclass, field
from numbers import Number
from typing import Iterable

import numpy as np

# Fundamental constants
A0 = 5.29177e-11  # Bohr radius, in unit of meter
# micron = 1000  # Length scale micrn, in unit of nm
# Eha = 6579.68392E12 * 2 * np.pi  # Hartree energy, in unit of Hz
AMU = 1.66053907e-27  # atomic mass, in unit of kg
h = 6.62607015e-34  # Planck constant
# l = 780E-9 / a0  # 780nm, light wavelength

DIM = 3  # space dimension

# NOTE: 1. Harmonic length \propto sqrt(w)
#       2. All length units are in wx, wy are rep by a factor wy/wx.
#          z direction zRx, zRy are also in unit of wx.


def _axis_tuple(values, dtype, length: int = DIM) -> tuple:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size > length:
        raise ValueError(f"Expected at most {length} axis values, got {arr.size}.")
    full = np.zeros(length, dtype=dtype)
    full[: arr.size] = arr
    return tuple(full.tolist())


def _optional_axis_tuple(values, dtype, length: int = DIM) -> tuple | None:
    if values is None:
        return None
    return _axis_tuple(values, dtype=dtype, length=length)


def _waist_tuple(waist) -> tuple[float, float]:
    if isinstance(waist, Iterable) and not isinstance(waist, (str, bytes)):
        values = tuple(float(item) for item in waist)
        if len(values) == 1:
            return (values[0], values[0])
        if len(values) >= 2:
            return (values[0], values[1])
    if isinstance(waist, Number):
        value = float(waist)
        return (value, value)
    raise TypeError("Trap waist must be a scalar or iterable of one or two values.")


@dataclass(frozen=True)
class TrapParameters:
    depth_khz: float
    waist_nm: tuple[float, float]
    atom_mass_amu: float = 6.015122
    laser_wavelength_nm: float = 780.0
    rayleigh_length_nm: float | tuple[float, float] | None = None

    @classmethod
    def from_legacy(
        cls,
        trap: tuple[float, float | tuple[float, float]],
        atom: float = 6.015122,
        laser: float = 780.0,
        zR=None,
    ) -> "TrapParameters":
        return cls(
            depth_khz=float(trap[0]),
            waist_nm=_waist_tuple(trap[1]),
            atom_mass_amu=float(atom),
            laser_wavelength_nm=float(laser),
            rayleigh_length_nm=zR,
        )

    def to_legacy_tuple(self) -> tuple[float, float | tuple[float, float]]:
        if self.waist_nm[0] == self.waist_nm[1]:
            waist: float | tuple[float, float] = self.waist_nm[0]
        else:
            waist = self.waist_nm
        return self.depth_khz, waist


@dataclass(frozen=True)
class DVRConfig:
    n: tuple[int, int, int]
    R0: tuple[float, float, float]
    avg: float = 1.0
    model: str = "Gaussian"
    trap: TrapParameters = field(
        default_factory=lambda: TrapParameters(104.52, (1000.0, 1000.0))
    )
    symmetry: bool = True
    parity: tuple[int, int, int] | None = None
    absorber: bool = False
    ab_param: tuple[float, float] = (57.04, 1.0)
    sparse: bool = False
    verbosity: int = 2

    @classmethod
    def from_inputs(
        cls,
        n,
        R0,
        avg: float = 1.0,
        model: str = "Gaussian",
        trap: tuple[float, float | tuple[float, float]] = (104.52, 1000),
        atom: float = 6.015122,
        laser: float = 780.0,
        zR=None,
        symmetry: bool = True,
        parity=None,
        absorber: bool = False,
        ab_param: tuple[float, float] = (57.04, 1.0),
        sparse: bool = False,
        verbosity: int = 2,
    ) -> "DVRConfig":
        return cls(
            n=_axis_tuple(n, int),
            R0=_axis_tuple(R0, float),
            avg=float(avg),
            model=model,
            trap=TrapParameters.from_legacy(trap, atom=atom, laser=laser, zR=zR),
            symmetry=bool(symmetry),
            parity=_optional_axis_tuple(parity, int),
            absorber=bool(absorber),
            ab_param=(float(ab_param[0]), float(ab_param[1])),
            sparse=bool(sparse),
            verbosity=int(verbosity),
        )

    def to_init_kwargs(self) -> dict:
        parity = None if self.parity is None else np.array(self.parity, dtype=int)
        return {
            "n": np.array(self.n, dtype=int),
            "R0": np.array(self.R0, dtype=float),
            "avg": self.avg,
            "model": self.model,
            "trap": self.trap.to_legacy_tuple(),
            "atom": self.trap.atom_mass_amu,
            "laser": self.trap.laser_wavelength_nm,
            "zR": self.trap.rayleigh_length_nm,
            "symmetry": self.symmetry,
            "parity": parity,
            "absorber": self.absorber,
            "ab_param": self.ab_param,
            "sparse": self.sparse,
            "verbosity": self.verbosity,
        }


@dataclass(frozen=True)
class DVRGridMetadata:
    n: tuple[int, int, int]
    R0: tuple[float, float, float]
    R: tuple[float, float, float]
    dx: tuple[float, float, float]
    nd: tuple[bool, bool, bool]
    parity: tuple[int, int, int]
    init: tuple[int, int, int]


@dataclass(frozen=True)
class DVRPhysicsMetadata:
    base_model: str
    active_model: str
    avg: float
    dvr_symmetry: bool
    sparse: bool
    absorber: bool
    hbar: float
    mass: float
    V0: float
    kHz: float
    kHz_2p: float
    waist_scale_m: float
    waist_ratio: tuple[float, float]
    rayleigh_range: tuple[float, float] | None
    effective_rayleigh_range: float | None
    omega: tuple[float, float, float]
    harmonic_length: tuple[float, float, float]
    absorber_strength: float
    absorber_width: float
    absorber_strength_scaled: float | None


@dataclass(frozen=True)
class DVRMetadata:
    config: DVRConfig
    grid: DVRGridMetadata
    physics: DVRPhysicsMetadata
