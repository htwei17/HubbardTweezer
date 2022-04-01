from typing import Iterable
from torch import Value
from opt_einsum import contract

from DVR_full import *

ax = 1520E-9
ay = 1620E-9


class Wannier(DVR):

    def update_lattice(self, lattice, lc=(ax, ay)):
        # graph : each line represents coordinate (x, y) of one lattice site
        self.lattice = lattice
        self.graph, self.links = lattice_graph(lattice)
        self.lc = np.array(lc) / a0
        # TODO: let R be set by lattice dimensions

    def __init__(
            self,
            n: np.ndarray,
            R0: np.ndarray,
            lattice: np.ndarray = (2),  # Square lattice size
            lc=(ax, ay),  # Lattice constant
            avg=1,
            model='Gaussian',
            trap=(104.52, 1E-6),
            symmetry: bool = False,
            absorber: bool = False,
            ab_param=(57.04, 1),
            sparse: bool = True) -> None:

        super().__init__(n, R0, avg, model, trap, symmetry, absorber, ab_param,
                         sparse)

        self.update_lattice(lattice, lc)

    def Vfun(self, x, y, z):
        V = 0
        for coord in self.graph:
            coord *= self.lc
            V += super().Vfun(x - coord[0], y - coord[1], z)
        return V


def lattice_graph(size: np.ndarray):
    # Square lattice graph builder
    # graph: each ndarray object in graph is a coord pair (x, y) indicating the posistion of node
    # links: each row in links is a pair of node indices s.t. graph[idx1], graph[idx2] are linked by bounds

    if isinstance(size, Iterable):
        size = np.array(size)

    edge = []
    links = []
    for i in range(size.ndim):
        edge.append(np.arange(-(size[i] - 1) / 2, (size[i] - 1) / 2 + 1))
    if size.ndim == 1:
        graph = [np.arrary([i, 0]) for i in edge]
        links = [np.array([i, i + 1]) for i in range(size[0] - 1)]
    elif size.ndim == 2:
        graph = []
        links = np.array([]).reshape(0, 2)
        node_idx = 0  # Linear index is column prefered
        for i in edge[0]:
            for j in edge[1]:
                graph.append(np.array([i, j]))
                if i > 0 and j > 0:
                    links = np.append(links,
                                      np.array([node_idx - edge[1],
                                                node_idx])[None],
                                      axis=0)  # Row link
                    links = np.append(links,
                                      np.array([node_idx - 1, node_idx])[None],
                                      axis=0)  # Column linke
                node_idx += 1

    return graph, links


def Wannier_eig_basis(dvr: Wannier):
    dvr.p = np.zeros(dim, dtype=int)
    dvr.p[dvr.nd] = 1  # Get all even sector
    dvr.sparse = True

    Nsite = np.prod(dvr.lattice)
    E, Wp = H_solver(dvr, Nsite)
    parity = np.tile(np.array([1, 1]), (Nsite, 1))  # Parity sector marker
    W = [Wp[:, i] for i in range(Nsite)]
    E, W, parity = add_sector(
        [-1, 1], dvr, Nsite, E, W,
        parity)  #  Get odd sector of the x lattice direction

    if dvr.lattice.ndim > 1 and all(Nsite != dvr.lattice):  # 2D lattice
        E, W, parity = add_sector(
            [1, -1], dvr, Nsite, E, W,
            parity)  # Get odd sector of the y lattice direction
        E, W, parity = add_sector(
            [-1, -1], dvr, Nsite, E, W,
            parity)  # Get odd sector of the x, y lattice direction

    idx = np.argsort(
        E)[:Nsite]  # Sort everything by nergy, only keetp lowest Nsite states
    E = E[idx]
    W = W[:, idx]
    parity = parity[idx, :]
    return E, W, parity


def add_sector(sector, dvr: Wannier, Nsite, E, W, parity):
    sec_idx = np.array([True, True, False]) * dvr.nd
    dvr.p[sec_idx] = sector[:np.sum(sec_idx)]

    Em, Wm = H_solver(dvr, Nsite)
    E = np.append(E, Em)
    W += [Wm[:, i] for i in range(Nsite)]
    parity = np.append(parity, np.tile(np.array([1, -1]), (Nsite, 1)))
    return E, W, parity


def two_site_TB(dvr: Wannier):
    lattice = (2)
    Nsite = np.prod(np.array(lattice))
    dvr.update_lattice(lattice)
    E, W, parity = Wannier_eig_basis(dvr)
    total_volumne = np.prod(dvr.R) * 2**np.sum(dvr.nd)

    U = parity_transfm(dvr)
    block_boundary = 0  # For 2-site
    # B = <Delta_i|Delta_j>_B, where _B means integration only over the block of local site
    Bdiag = np.ones(2 * dvr.n[i] + 1)
    x = np.arange(-dvr.n[i], dvr.n[i] + 1) * dvr.dx[i]
    Bdiag[x > block_boundary] = 0
    B = np.diag(Bdiag)

    # Construct localization matrix
    L = np.zeros((Nsite, Nsite))
    # 2-site version
    block_volume = np.prod(dvr.R) * 2**(np.sum(dvr.nd) - 1)

    for i in range(Nsite):
        for j in range(Nsite):
            if i > j:
                # S = <Delta^p_i|Delta^p'_j>_B, where _B means integration only over the block of local site
                pi = int((1 - parity[i, 0]) / 2)
                pj = int((1 - parity[j, 0]) / 2)
                S = U[pi] @ B @ U[pj].conj().T
                ni = dvr.n + 1 - get_init(dvr.n, np.append(parity[i, :], 1))
                nj = dvr.n + 1 - get_init(dvr.n, np.append(parity[j, :], 1))
                Wi = W[i].reshape(*ni)
                Wj = W[j].reshape(*nj)
                L[i, j] = contract('ijk,il,ljk', Wi.conj(), S, Wj)
                L[j, i] = np.conj(L[i, j])
    L += block_volume / total_volumne * np.eye(Nsite)
    # Below are 2-site only
    # Find eigenvector, ie. max localized Wannier function at site. For 2-site case, the other eigenvector is the Wannier func for the other site
    e, a = la.eigh(L)
    idx = np.argsort(e)
    e = e[idx]
    a = a[:, idx]
    A = a.conj().T @ (
        E * a)  # Hermitian matrix, off-diagonals should be equal. Diagonals?
    mu = np.diag(A)  # mu_i for 2 sites
    t = -np.array([A[0, 1], A[1, 0]])  # J_i for 2 sites
    return mu, t


def parity_transfm(n: int):
    # Parity basis transformation matrix: 1D version

    Up = np.zeros((n + 1, 2 * n + 1))
    Up[0, n] = 1  # n=d
    Up[1:, :n] = np.eye(n)  # Negative half-axis
    Up[1:, (n + 1):] = np.eye(n)  # Positive half-axis
    Um = np.zeros((n, 2 * n + 1))
    Um[:, n:] = -1 * np.eye(n)  # Negative half-axis
    Um[:, :n] = np.eye(n)  # Positive half-axis
    return Up, Um
