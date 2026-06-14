from numbers import Number
from typing import Iterable, Literal, Union
import sys
from time import time
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import scipy.sparse as sp
import scipy.interpolate as interp
from scipy.sparse.linalg import LinearOperator
from opt_einsum import contract

from ..tools.potentials import tweezer_potential

from .metadata import (
    AMU,
    DIM,
    h,
    DVRConfig,
    DVRGridMetadata,
    DVRMetadata,
    DVRPhysicsMetadata,
)


def get_init(n: np.ndarray, p: np.ndarray) -> np.ndarray:
    init = -n
    init[p == 1] = 0
    init[p == -1] = 1
    return init


def _kinetic_offdiag(T: np.ndarray) -> np.ndarray:
    # Kinetic energy matrix off-diagonal elements
    non0 = T != 0  # To avoid warning on 0-divide-0
    T[non0] = 2 * np.power(-1.0, T[non0]) / T[non0] ** 2
    return T


class DVR:
    """DVR base class

    Args:
    ----------
        n (`np.ndarray[int, int, int]`): DVR grid half-size in each direction.
            The actual grid size is 2n+1.
            If i-th dimension is not calculated, n[i]=0
        R0 (`np.ndarray[float, float]`): Grid halfwidth in each direction
        model (`str`): Trap potential.
            "Gaussian": tweezer potential
            "sho": simple harmonic potential
            "free": free particle, no potential
        avg (`float`): Factor a in Humiltonian H = T + aV
        trap (`tuple[float, float | tuple[float, float]]`): Trap potential parameters.
            trap[0] is the trap potential strength in unit of kHz
            trap[1] is the waist in unit of nm;
            if trap[1] given as a tuple, then it is (wx, wy)
        atom (`float`): Atom mass in unit of amu
        laser ('float'): Laser wavelength in unit of nm
        zR ('float'): Rayleigh length in unit of nm;
            if not given, zR is calculated by \\pi * w^2 / laser
        sparse (`bool`): Whether to use sparse matrix
        symmetry (`bool`): Whether to use symmetry in DVR
        absorber (`bool`): Whether to use absorber
        ab_param (`tuple[float, float]`): Absorber parameters.
            ab_param[0] is the strength of linear imaginary potential in unit of kHz
            ab_param[1] is the width of absorber in unit of wx
    """

    initial_index = staticmethod(get_init)
    kinetic_offdiag = staticmethod(_kinetic_offdiag)

    @staticmethod
    def _default_grid_size(model: str) -> int:
        if model == "sho":
            return 15
        return 10

    @staticmethod
    def _infer_dim(dim, n, R0) -> int:
        if dim is not None:
            return max(1, min(DIM, int(dim)))

        for values in (n, R0):
            if values is None:
                continue
            arr = np.asarray(values).reshape(-1)
            if arr.size > 1:
                nonzero = int(np.count_nonzero(arr[:DIM]))
                return max(1, min(DIM, nonzero or arr.size))
        return DIM

    @staticmethod
    def _coerce_axis_array(values, name: str, dtype, dim: int) -> np.ndarray:
        arr = np.asarray(values, dtype=dtype).reshape(-1)
        if arr.size == 1 and dim > 1:
            arr = np.repeat(arr, dim)
        if arr.size > DIM:
            raise ValueError(f"{name} must have at most {DIM} entries.")

        full = np.zeros(DIM, dtype=dtype)
        full[: arr.size] = arr
        return full

    @classmethod
    def _resolve_grid(
        cls,
        n=None,
        R0=None,
        N=None,
        dim=None,
        model: str = "Gaussian",
    ) -> tuple[np.ndarray, np.ndarray]:
        if R0 is None:
            raise TypeError("DVR grid resolution requires R0.")

        resolved_dim = cls._infer_dim(dim, N if n is None else n, R0)
        R0 = cls._coerce_axis_array(R0, "R0", float, resolved_dim)

        if n is None:
            grid_n = cls._default_grid_size(model) if N in (None, 0) else N
            n = cls._coerce_axis_array(grid_n, "n", int, resolved_dim)
        else:
            scalar_n = np.asarray(n).reshape(-1).size == 1
            n = cls._coerce_axis_array(n, "n", int, resolved_dim)
            if scalar_n and n[0] == 0:
                n = cls._coerce_axis_array(
                    cls._default_grid_size(model), "n", int, resolved_dim
                )

        n[resolved_dim:] = 0
        R0[resolved_dim:] = 0.0
        return n, R0

    def update_n(self, n: np.ndarray, R0: np.ndarray):
        # Change n by fixed R0
        self.n = n.copy()
        self.init[self.nd] = self.initial_index(self.n[self.nd], self.p[self.nd])
        self.R0 = R0.copy()
        self.dx = np.zeros(n.shape)
        self.nd = n != 0
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]
        self.update_absorber()

    def update_R0(self, R0: np.ndarray, dx: np.ndarray):
        # Update R0 by fixed dx
        self.R0 = R0.copy()
        self.dx = dx.copy()
        self.nd = R0 != 0
        self.n[self.nd == 0] = 0
        self.dx[self.nd == 0] = 0
        self.n[self.nd] = (self.R0[self.nd] / self.dx[self.nd]).astype(int)
        self.init[self.nd] = self.initial_index(self.n[self.nd], self.p[self.nd])
        self.update_absorber()

    def update_absorber(self):
        # Update absorber
        if self.verbosity:
            print(f"DVR: dx={self.dx[self.nd]}w is set.")
            print(f"DVR: n={self.n[self.nd]} is set.")
            print(f"DVR: R0={self.R0[self.nd]}w is set.")
        if self.absorber:
            if self.verbosity:
                print(f"DVR: Absorber width LI={self.LI:g}w")
            # if __debug__:
            #     print(self.R0[0])
            #     print(self.dx[0])
            #     a = self.LI / self.dx[self.nd]
            #     print(a[0])
            #     print(np.rint(self.LI / self.dx[self.nd]).astype(int))
            self.n[self.nd] += np.rint(self.LI / self.dx[self.nd]).astype(int)
            self.R[self.nd] = self.n[self.nd] * self.dx[self.nd]
            if self.verbosity > 1:
                print(f"DVR: n is set to {self.n[self.nd]} by adding absorber.")
        else:
            self.R[self.nd] = self.R0[self.nd]
            if self.verbosity > 1:
                print(f"DVR: R={self.R[self.nd]}w is set.")

    def update_parity(self, p):
        # Update parity
        self.p = p
        self.init = self.initial_index(self.n, p)

    def _store_input_metadata(self, model, trap, atom, laser, zR, ab_param):
        self.base_model = model
        self._input_trap = trap
        self._input_atom = atom
        self._input_laser = laser
        self._input_zR = zR
        self._input_ab_param = tuple(ab_param)

    def _initialize_grid_state(self, n: np.ndarray, R0: np.ndarray) -> None:
        self.n = n.copy()
        self.R0 = R0.copy()  # Physical region size, in unit of wx
        self.R = R0.copy()  # Total region size, R = R0 + LI
        self.nd = n != 0
        self.dx = np.zeros(n.shape)
        self.dx[self.nd] = self.R0[self.nd] / n[self.nd]

    def _configure_absorber(
        self, absorber: bool, ab_param: tuple[float, float]
    ) -> None:
        self.absorber = absorber
        if absorber:
            self.VI, self.LI = ab_param
        else:
            self.VI = 0
            self.LI = 0

    def _configure_parity(self, parity) -> None:
        self.p = np.zeros(DIM, dtype=int)
        if self.dvr_symm:
            if parity is None:
                self.p[self.nd] = 1
            else:
                self.p[self.nd] = parity[self.nd].astype(int)
            if self.verbosity:
                axis = np.array(["x", "y", "z"])
                print(f"{axis[self.nd]}-reflection symmetry is used.")
        self.init = self.initial_index(self.n, self.p)

    def _configure_experimental_units(
        self,
        trap: tuple[float, float | tuple[float, float]],
        atom: float,
        laser: float,
        zR,
    ) -> None:
        self.hb = h / (2 * np.pi)  # Reduced Planck constant
        self.m = atom * AMU  # Atom mass, in unit of electron mass
        self.kHz = 1e3
        self.kHz_2p = 2 * np.pi * self.kHz
        self.V0 = trap[0] * self.kHz_2p

        if isinstance(trap[1], Iterable) and len(trap[1]) == 1:
            wx: Number = trap[1][0]
            self.wxy = np.ones(2)
        elif isinstance(trap[1], Iterable):
            wx = trap[1][0]
            self.wxy = np.array(trap[1]) / wx
        elif isinstance(trap[1], Number):
            wx = trap[1]
            self.wxy = np.ones(2)
        else:
            wx = 1000
            self.wxy = np.ones(2)
        self.w = wx * 1e-9

        self.mtV0 = self.m * self.V0
        self.l: Literal[None] = None
        if isinstance(zR, Number):
            self.zR = zR * np.ones(2) / wx
        elif isinstance(zR, Iterable):
            self.zR = np.array(zR) / wx
        else:
            self.l = laser * 1e-9
            self.zR = np.pi * self.w * self.wxy**2 / self.l
        self.zR0 = np.prod(self.zR) / la.norm(self.zR)

        self.omega = np.array([*(np.sqrt(2) / self.wxy), 1 / self.zR0])
        self.omega *= np.sqrt(2 * self.avg * self.hb * self.V0 / self.m) / self.w
        self.hl = np.sqrt(self.hb / (self.m * self.omega))

        if self.verbosity:
            print(f"param_set: trap parameter V0={self.avg * trap[0]}kHz w={trap[1]}nm")

    def _configure_sho_units(self) -> None:
        self.hb = 1.0
        self.omega = np.ones(DIM)
        self.m = 1.0
        self.w = 1.0
        self.wxy = np.ones(2)
        self.mtV0 = self.m
        self.V0 = 1.0
        self.kHz = 1.0
        self.kHz_2p = 1.0
        self.zR = None
        self.zR0 = None
        self.l = None
        self.hl = np.sqrt(self.hb / (self.m * self.omega))

        print(f"param_set: trap parameter V0={self.avg * self.V0} w0={self.w}")

    def _configure_physics(
        self,
        model: str,
        trap: tuple[float, float | tuple[float, float]],
        atom: float,
        laser: float,
        zR,
    ) -> None:
        if model in ["Gaussian", "optical_lattice", "custom"]:
            self._configure_experimental_units(trap, atom, laser, zR)
        elif model == "sho":
            self._configure_sho_units()
        else:
            raise ValueError(f"Unsupported DVR model {model}.")

    def _set_custom_potential(self, custom_potential) -> None:
        self.custom_potential = None
        if custom_potential is None:
            return

        if isinstance(custom_potential, Iterable) and not callable(custom_potential):
            points, values = custom_potential
            self.custom_potential = interp.RegularGridInterpolator(
                points=points, values=values
            )
            if self.verbosity:
                print("DVR: custom potential interpolator is set.")
        elif callable(custom_potential):
            self.custom_potential = custom_potential
            if self.verbosity:
                print("DVR: custom potential function is set.")
        else:
            raise TypeError(
                "Invalid custom potential type. The accepted types are callable or tuple of (grid, values)."
            )

    def evaluate_custom_potential(self, x, y, z):
        if self.custom_potential is None:
            raise ValueError("custom potential model requires custom_potential.")

        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("X, Y, Z must have the same shape.")

        if isinstance(self.custom_potential, interp.RegularGridInterpolator):
            points = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
            return self.custom_potential(points).reshape(x.shape)
        return self.custom_potential(x, y, z)

    def _builtin_Vfun(self, model: str, x, y, z):
        if model == "Gaussian":
            return tweezer_potential(x, y, z, self.wxy, self.zR, self.zR0)
        if model == "sho":
            return (
                self.m
                / 2
                * (
                    self.omega[0] ** 2 * x**2
                    + self.omega[1] ** 2 * y**2
                    + self.omega[2] ** 2 * z**2
                )
            )
        if model == "free":
            return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z)))
        raise ValueError(f"Unsupported potential model {model}.")

    def _apply_axis_masks(self) -> None:
        self.R0 *= self.nd
        self.R *= self.nd
        self.dx *= self.nd
        self.hl[np.logical_not(self.nd)] = 1

        if self.absorber:
            self.LI = self._input_ab_param[1]
            self.VI = self._input_ab_param[0] * self.kHz_2p
            self.VIdV0 = self.VI / self.V0
        else:
            self.VI = 0
            self.LI = 0
            self.VIdV0 = None

    def update_trap(self, trap=None, atom=None, laser=None, zR=None) -> None:
        if trap is not None:
            self._input_trap = trap
        if atom is not None:
            self._input_atom = atom
        if laser is not None:
            self._input_laser = laser
        if zR is not None:
            self._input_zR = zR

        self._configure_physics(
            self.base_model,
            self._input_trap,
            self._input_atom,
            self._input_laser,
            self._input_zR,
        )
        self._apply_axis_masks()

    @property
    def config(self) -> DVRConfig:
        parity = self.p if self.dvr_symm else None
        return DVRConfig.from_inputs(
            n=self.n,
            R0=self.R0,
            avg=self.avg,
            model=self.base_model,
            trap=self._input_trap,
            atom=self._input_atom,
            laser=self._input_laser,
            zR=self._input_zR,
            symmetry=self.dvr_symm,
            parity=parity,
            absorber=self.absorber,
            ab_param=self._input_ab_param,
            sparse=self.sparse,
            verbosity=self.verbosity,
        )

    @property
    def grid_metadata(self) -> DVRGridMetadata:
        return DVRGridMetadata(
            n=tuple(self.n.astype(int).tolist()),
            R0=tuple(self.R0.astype(float).tolist()),
            R=tuple(self.R.astype(float).tolist()),
            dx=tuple(self.dx.astype(float).tolist()),
            nd=tuple(self.nd.astype(bool).tolist()),
            parity=tuple(self.p.astype(int).tolist()),
            init=tuple(self.init.astype(int).tolist()),
        )

    @property
    def physics_metadata(self) -> DVRPhysicsMetadata:
        zR = (
            None
            if self.zR is None
            else tuple(np.asarray(self.zR, dtype=float).tolist())
        )
        zR0 = None if self.zR0 is None else float(self.zR0)
        VIdV0 = None if self.VIdV0 is None else float(self.VIdV0)
        return DVRPhysicsMetadata(
            base_model=self.base_model,
            active_model=self.model,
            avg=float(self.avg),
            dvr_symmetry=bool(self.dvr_symm),
            sparse=bool(self.sparse),
            absorber=bool(self.absorber),
            hbar=float(self.hb),
            mass=float(self.m),
            V0=float(self.V0),
            kHz=float(self.kHz),
            kHz_2p=float(self.kHz_2p),
            waist_scale_m=float(self.w),
            waist_ratio=tuple(np.asarray(self.wxy, dtype=float).tolist()),
            rayleigh_range=zR,
            effective_rayleigh_range=zR0,
            omega=tuple(np.asarray(self.omega, dtype=float).tolist()),
            harmonic_length=tuple(np.asarray(self.hl, dtype=float).tolist()),
            absorber_strength=float(self.VI),
            absorber_width=float(self.LI),
            absorber_strength_scaled=VIdV0,
        )

    @property
    def metadata(self) -> DVRMetadata:
        return DVRMetadata(
            config=self.config,
            grid=self.grid_metadata,
            physics=self.physics_metadata,
        )

    @classmethod
    def from_metadata(cls, metadata: DVRMetadata | DVRConfig, **overrides):
        if isinstance(metadata, DVRMetadata):
            kwargs = metadata.config.to_init_kwargs()
        else:
            kwargs = metadata.to_init_kwargs()
        kwargs.update(overrides)
        return cls(**kwargs)

    def __init__(
        self,
        n: np.ndarray,
        R0: np.ndarray,
        avg: float = 1,
        model: str = "Gaussian",
        custom_potential=None,
        # 2nd entry in array is (wx, wy) in unit of nm
        # if given in single number w it is (w, w)
        trap: tuple[float, Union[float, tuple[float, float]]] = (104.52, 1000),
        atom: float = 6.015122,  # Atom mass, in amu. Default Lithium-6
        laser: float = 780,  # 780nm, laser wavelength in unit of nm
        # Rayleigh range input by hand, in unit of nm
        zR: Union[None, float] = None,
        symmetry: bool = True,
        # Parity of each dimension, used when symmetry is True
        parity: Union[None, np.ndarray] = None,
        absorber: bool = False,
        ab_param: tuple[float, float] = (57.04, 1),
        sparse: bool = False,
        verbosity: int = 2,  # How much information to print
        *args,
        **kwargs,
    ) -> None:
        self._store_input_metadata(model, trap, atom, laser, zR, ab_param)
        self.avg = avg
        self.model = "free" if self.avg == 0 else model
        self.dvr_symm = symmetry
        self.sparse = sparse
        self.verbosity = verbosity if verbosity >= 0 else 0

        self._initialize_grid_state(n, R0)
        self._configure_absorber(absorber, ab_param)
        self.update_absorber()
        self._configure_parity(parity)
        self._configure_physics(model, trap, atom, laser, zR)
        self._set_custom_potential(custom_potential)
        self._apply_axis_masks()

    def Vfun(self, x, y, z):
        # Potential function
        if self.model == "custom":
            return self.evaluate_custom_potential(x, y, z)
        return self._builtin_Vfun(self.model, x, y, z)

    def Vabs(self, x, y, z):
        r = np.array([x, y, z]).transpose(1, 2, 3, 0)
        np.set_printoptions(threshold=sys.maxsize)
        d: np.ndarray = abs(r) - self.R0
        d = (d > 0) * d
        L: np.ndarray = self.R - self.R0
        L[L == 0] = np.inf
        # if __debug__:
        #     print(self.R)
        #     print(self.R0)
        #     print(L)
        Vi: np.ndarray = np.sum(d / L, axis=3)
        V = -1j * self.VIdV0 * Vi  # Energy in unit of V0
        return V

    def Vmat(self) -> tuple[np.ndarray, np.ndarray]:
        # Potential energy tensor, index order [x y z x' y' z']
        # NOTE: here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
        #       potential(x, y, z) is a function handle to be processed as potential function for solving
        x = []
        for i in range(DIM):
            x.append(
                np.arange(self.init[i], self.n[i] + 1) * self.dx[i]
            )  # In unit of micron
        X = np.meshgrid(*x, indexing="ij")
        # 3 index tensor V(x, y, z)
        V: np.ndarray = self.avg * self.Vfun(*X)
        if self.absorber:  # add absorber
            V = V.astype(complex) + self.Vabs(*X)
        no = self.n + 1 - self.init
        # V * identity rank-6 tensor
        if not self.sparse:
            V = np.diag(V.reshape(-1))
            V = V.reshape(*no, *no)
        return V, no

    def _Tmat_1d(self, i: int):
        # Kinetic energy matrix for 1-dim
        n = self.n[i]
        # dx is dimensionless by dividing w,
        # to restore unit we need to multiply w
        dx = self.dx[i] * self.w
        p = self.p[i]

        init = self.initial_index(np.array([n]), np.array([p]))[0]

        # Off-diagonal part
        T0 = np.arange(init, n + 1, dtype=float)[None]
        T = self.kinetic_offdiag(T0 - T0.T)

        # Diagonal part
        T[np.diag_indices(n + 1 - init)] = np.pi**2 / 3
        if p != 0:
            T += p * self.kinetic_offdiag(T0 + T0.T)
            if p == 1:
                T[:, 0] /= np.sqrt(2)
                T[0, :] /= np.sqrt(2)
                T[0, 0] = np.pi**2 / 3
        # get kinetic energy in unit of V0, hb is cancelled as V0 is in unit of angular freq, mtV0 = m * V0
        T *= self.hb / (2 * dx**2 * self.mtV0)
        return T

    def Tmat(self):
        # Kinetic energy tensor, index order [x y z x' y' z']
        # NOTE: 1. here n, dx are 3-element np.array s.t. n = [nx, ny, nz], dx = [dx, dy, dz]
        #       2. p=0, d=-1 means no symmetry applied
        delta = []
        T0 = []
        for i in range(DIM):
            delta.append(np.eye(self.n[i] + 1 - self.init[i]))  # eg. delta_xx'
            if self.n[i]:
                T0.append(self._Tmat_1d(i))  # append p-sector
            # If the systems is set to have only 1 grid point (N = 0) in this direction, ie. no such dimension
            else:
                T0.append(None)

        if self.sparse:
            for i in range(DIM):
                if not isinstance(T0[i], np.ndarray):
                    T0[i] = np.zeros((1, 1))
            return T0
        else:
            # delta_xx' delta_yy' T_zz'
            # delta_xx' T_yy' delta_zz'
            # T_xx' delta_yy' delta_zz'
            T = 0
            for i in range(DIM):
                if isinstance(T0[i], np.ndarray):
                    T += contract(
                        "ij,kl,mn->ikmjln", *delta[:i], T0[i], *delta[i + 1 :]
                    )
            return T

    def H_op(self, T: list, V, no, psi0: np.ndarray):
        # Define Hamiltonian operator for sparse solver

        psi0 = psi0.reshape(*no)
        psi: np.ndarray = V * psi0  # delta_xx' delta_yy' delta_zz' V(x,y,z)
        # T_xx' delta_yy' delta_zz'
        psi += contract("ij,jkl->ikl", T[0], psi0)
        # delta_xx' T_yy' delta_zz'
        psi += contract("jl,ilk->ijk", T[1], psi0)
        # delta_xx' delta_yy' T_zz'
        psi += contract("ij,klj->kli", T[2], psi0)
        return psi.reshape(-1)

    def H_mat(self):
        # Construct Hamiltonian matrix
        self.p *= self.n != 0
        self.dx *= self.n != 0

        # np.set_printoptions(precision=2, suppress=True)
        if self.verbosity:
            print(
                f"H_mat: n={self.n[self.nd]} dx={self.dx[self.nd]}w p={self.p[self.nd]} {self.model} diagonalization starts."
            )
        T = self.Tmat()
        V, no = self.Vmat()
        H = T + V
        del T, V
        N = np.prod(no)
        H = H.reshape((N, N))
        if not self.absorber:
            H = (H + H.T.conj()) / 2
        if self.verbosity:
            print(f"H_mat: H matrix memory usage: {H.nbytes / 2**20:.2f} MiB.")
        return H

    def H_solver(
        self,
        k: int = -1,  # number of eigenvalues to be calculated
        v0: np.ndarray = None,  # initial guess for eigenvectors
    ) -> tuple[np.ndarray, np.ndarray]:
        # Solve Hamiltonian matrix

        if self.sparse:
            if self.verbosity:
                print(
                    f"H_op: n={self.n[self.nd]} dx={self.dx[self.nd]}w "
                    f"p={self.p[self.nd]} {self.model} sparse diagonalization starts. "
                    f"Lowest {k} states are to be calculated."
                )

            self.p *= self.n != 0
            self.dx *= self.n != 0

            T = self.Tmat()
            V, no = self.Vmat()
            # for i in range(3):
            #     print(T[i])
            # print(V)
            if self.verbosity > 2:
                print(
                    f"H_op: n={self.n[self.nd]} dx={self.dx[self.nd]}w "
                    f"p={self.p[self.nd]} {self.model} operator constructed."
                )

            t0 = time()

            def applyH(psi) -> np.ndarray:
                return self.H_op(T, V, no, psi)

            N = np.prod(no)
            H = LinearOperator((N, N), matvec=applyH)

            if v0 is not None:  # Flatten v0
                v0 = v0.reshape(-1)

            if k <= 0:
                k = 10
            if self.absorber:
                if self.verbosity > 2:
                    print("H_solver: diagonalize sparse non-hermitian matrix.")
                    print(f"H_solver: matrix dimension = {N}")
                E, W = ssla.eigs(H, k, which="SA", v0=v0)
            else:
                if self.verbosity > 2:
                    print("H_solver: diagonalize sparse hermitian matrix.")
                    print(f"H_solver: matrix dimension = {N}")
                E, W = ssla.eigsh(H, k, which="SA", v0=v0)
        else:
            # avg factor is used to control the time average potential strength
            H = self.H_mat()
            t0 = time()
            if self.absorber:
                if self.verbosity > 2:
                    print("H_solver: diagonalize non-hermitian matrix.")
                    print(f"H_solver: matrix dimension = {H.shape[0]}")
                E, W = la.eig(H)
            else:
                if self.verbosity > 2:
                    print("H_solver: diagonalize hermitian matrix.")
                    print(f"H_solver: matrix dimension = {H.shape[0]}")
                E, W = la.eigh(H)
            if k > 0:
                E = E[:k]
                W = W[:, :k]

        t1 = time()

        if self.verbosity:
            print(
                f"H_solver: {self.model} Hamiltonian solved. Time spent: {t1 - t0:.2f}s."
            )
            print(f"H_solver: eigenstates memory usage: {W.nbytes/2**20: .2f} MiB.")
        # No absorber, all eigenstates are real
        return E, W
