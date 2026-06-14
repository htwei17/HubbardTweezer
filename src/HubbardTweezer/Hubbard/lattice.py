import cmath
from typing import Iterable
import numpy as np
from numbers import Number

from .grid import LatticeGrid
from .ghost import GhostTrap


TRIANGULAR_LATTICES = [
    "triangular",
    "honeycomvb",
    "defecthoneycomb",
    "kagome",
    "zigzag",
]


class Lattice:

    grid: LatticeGrid
    ghost: GhostTrap

    @property
    def N(self):
        return self.grid.N

    @property
    def dim(self):
        return self.grid.dim

    @property
    def size(self):
        return self.grid.size

    @property
    def shape(self):
        return self.grid.shape

    @property
    def Nindep(self):
        return self.grid.Nindep

    @property
    def mask(self):
        return self.ghost.mask

    def __init__(
        self,
        shape: str = "square",  # Shape of the lattice
        lattice_symmetry: bool = True,  # Whether the lattice has reflection symmetry
        # Square lattice dimensions & lattice constant, in unit of nm
        lattice: np.ndarray = np.array([2], dtype=int),
        lc: tuple[float, float] = (1520, 1690),
        # Custom lattice site positions & lattice links
        nodes: np.ndarray = None,  # custom lattice site positions, in unit of lc
        links: np.ndarray = None,  # custom lattice links
        isotropic: bool = False,  # Check if the lattice is isotropic
        ghost: bool = False,  # Whether to use ghost atoms or not
        ghost_penalty=(0, 0),  # Ghost penalty weight & threshold
    ):
        # Set ghost lattice shape
        ghost_shape = shape
        if ghost and shape == "Lieb":
            # If use ghost traps,
            # Lieb lattice has ghost sites in the interior
            # But its shape is square
            print("Equalize: Lieb lattice ghost sites.")
            print("Set shape to square for total system.")
            shape = "square"

        self.ls = lattice_symmetry
        self.isotropic = isotropic
        if shape == "zigzag":
            self.ls = False

        # graph : each line represents coordinate (x, y) of one lattice site
        self.grid = LatticeGrid(lattice, shape, self.ls, nodes, links)

        # Set target to be already limited in the bulk
        self.ghost = GhostTrap(self.grid, ghost_shape, *ghost_penalty)
        if ghost:
            self.ghost.set_mask(self.grid)

        self.set_lc(lc, shape)  # Convert lc to (lc, lc) and in unit of wx

    def set_lc(self, lc, shape):
        # Convert lc to (lc, lc) or the other if only one number is given
        if isinstance(lc, Iterable) and len(lc) == 1:
            lc: Number = lc[0]
        if isinstance(lc, Number):
            self.isotropic = True
            if shape in TRIANGULAR_LATTICES:
                # For equilateral triangle
                lc: tuple = (lc, np.sqrt(3) / 2 * lc)
            else:
                # For squre and others
                lc: tuple = (lc, lc)
        # Confirm (lc, lc) case that the lattice is isotropic
        if shape not in TRIANGULAR_LATTICES and lc[0] == lc[1]:
            self.isotropic = True
        print(f"Lattice: lattice shape is {shape}; lattice constants set to: {lc}")

        self.lc = np.array(lc, dtype=float)
        # Assume WF are localized at trap centers, location in unit of wx
        self.tc0 = self.grid.nodes * self.lc
        self.trap_centers = self.tc0.copy()

    def nn_tunneling(self, A: np.ndarray):
        # Pick up nearest neighbor tunnelings
        # Not limited to specific geometry
        if self.grid.N == 1:
            nnt = np.zeros(1)
        elif self.grid.dim == 1:
            nnt = np.diag(A, k=1)
        else:
            nnt = A[self.ghost.links[:, 0], self.ghost.links[:, 1]]
        return nnt

    def symm_unfold(self, target: Iterable, info, graph=False):
        # Unfold information to all symmetry sectors
        # No need to output as target is Iterable
        if self.ls:
            parity = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
            for row in range(self.grid.reflect.shape[0]):
                if graph:  # Symmetrize graph node coordinates
                    # NOTE: repeated nodes will be removed
                    info[row][self.grid.inv_coords[row]] = 0
                    target[self.grid.reflect[row, :]] = parity * info[row][None]
                else:  # Symmetrize trap depth
                    target[self.grid.reflect[row, :]] = info[row]
        else:
            target[:] = info
