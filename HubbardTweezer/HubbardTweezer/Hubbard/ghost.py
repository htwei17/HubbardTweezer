import numpy as np

from .lattice import Lattice, squeeze_idx


def _lieb_ghost_sites(Nx, Ny):
    # Generate interior ghost sites for Lieb lattice
    # Note that since the boundary is always ghost,
    # we can safely set all boundary sites to be ghost

    # Generate index pairs for rectangular lattice, column major
    idx_pairs = np.array([[i, j] for i in range(Nx) for j in range(Ny)])
    hole = np.all(idx_pairs % 2 == 0, axis=1)
    hole_idx = np.nonzero(hole)[0]
    return hole_idx


class GhostTrap:
    shape: str = None
    threshold: float
    weight: float
    mask: np.ndarray = None
    links: np.ndarray = None
    is_masked: bool = False
    penfunc: str = "exp"

    def __init__(self, lattice: Lattice, shape, threshold=0, penalty=0, func="exp"):
        self.shape = shape
        self.threshold = threshold
        self.weight = penalty
        self.penfunc = func
        self.mask = np.ones(lattice.N, dtype=bool)
        self.links = lattice.links
        self.Nsite = np.sum(self.mask)
        self.is_masked = False

    def set_mask(self, lattice: Lattice):
        # Set ghost trap for 1D & 2D lattice
        # If trap is ghost, mask is False

        # block = np.zeros(self.lattice.N, dtype=bool)
        err = ValueError("Ghost sites not implemented for this lattice.")

        if lattice.dim == 1:
            self.mask[[0, -1]] = False
        elif lattice.dim == 2:
            if self.shape in ["square", "triangular", "Lieb"]:
                Nx, Ny = lattice.size
                extra = np.array([], dtype=int)
                if self.shape in ["square", "Lieb"]:
                    x_bdry, y_bdry = self.xy_boundaries(lattice, Ny)
                    if self.shape == "Lieb":
                        # Add extra ghost sites for Lieb lattice
                        extra = _lieb_ghost_sites(Nx, Ny)
                elif self.shape == "triangular" and not self.ls:
                    y_bdry, x_bdry = self.xy_boundaries(lattice, Nx)
                else:
                    raise err
                bdry = [x_bdry, y_bdry, extra]  # x, y boundary site indices
                # which axis to mask
                mask_axis = np.nonzero(lattice.size > 2)[0]
                mask_axis = np.append(mask_axis, 2)  # add extra
                if mask_axis.size != 0:
                    masked_idx = np.concatenate([bdry[i] for i in mask_axis])
                    self.mask[masked_idx] = False
                # if extra.size != 0:  # Add traps to be blocked by onsite potential
                #     block[extra] = True
            else:
                raise err
        else:
            raise err
        masked_idx = np.where(~self.mask)[0]
        self.links = squeeze_idx(lattice.links, masked_idx)
        self.Nsite = np.sum(self.mask)
        # self.block = block
        self.is_masked = True
        print("Equalize: ghost sites are set.")

    def xy_boundaries(self, lattice: Lattice, N):
        # Identify x and y boundary site indices
        # For example, for a 4x4 square lattice,
        # x_bdry = [0, 1, 2, 3, 12, 13, 14, 15]
        # and y_bdry = [0, 4, 8, 12, 3, 7, 11, 15]
        x_bdry = np.concatenate((np.arange(N), np.arange(-N, 0)))
        y_bdry = np.concatenate(
            (np.arange(0, lattice.N, N), np.arange(N - 1, lattice.N, N))
        )
        return x_bdry, y_bdry

    def mask_quantity(self, quantity: np.ndarray):
        # Mask out ghost sites
        if quantity.ndim == 1:
            return quantity[self.mask]
        elif quantity.ndim == 2:
            return quantity[self.mask, :][:, self.mask]
        else:
            raise ValueError("Quantity must be 1D or 2D array.")

    def penalty(self, Vdist):
        # Penalty for negative V outside the mask
        # Vdist is modified in place
        if len(Vdist) != self.Nsite:
            # Iif Vdist not match length of mask, skip
            if self.is_masked and self.weight != 0:
                Vdist_unmasked = Vdist[~self.mask] - self.threshold
                # Criteria: make func value 0 at and beyond desired value,
                # not neccesarily threshold
                if self.penfunc == "exp":
                    Vpen = np.exp(-6 * Vdist_unmasked)  # 1e-6 at threshold + 2(kHz)
                elif self.penfunc == "sigmoid":
                    Vpen = 1 / (
                        1 + np.exp(6 * Vdist_unmasked)
                    )  # 1e-6 at threshold + 2(kHz)
                else:
                    Vpen = np.where(
                        Vdist_unmasked < 0, Vdist_unmasked, 0
                    )  # 0 at threshold
                Vdist[~self.mask] = self.weight * Vpen
            else:
                # Trim to be with only masked sites
                Vdist[~self.mask] = 0
