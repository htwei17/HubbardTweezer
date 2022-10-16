import cmath
from typing import Iterable
import numpy as np


class Lattice:
    def __init__(self, size: np.ndarray,
                 shape: str = 'square',
                 symmetry: bool = True):
        # Abstract lattice graph class
        # No lattice constant is encoded
        # shape: 'ring' 'square' 'Lieb' 'triangle' 'honeycomb' 'kagome'
        # NOTE: might not be very doable since the symmetries of trap are just x,y mirrors
        # nodes: each row is a coordinate (x, y) of one site
        #        indicating the posistion of node (trap center)
        # links: each row in links is a pair of node indices s.t.
        #        graph[idx1], graph[idx2] are linked by bounds

        self.N = np.prod(size)
        self.shape = shape
        self.symmetry = symmetry

        # Convert [n] to [n, 1]
        if self.N == 1:
            self.size = np.ones(1)
            self.dim = 1
        else:
            if size.size == 1:
                self.size = np.resize(
                    np.pad(size, pad_width=(0, 1), constant_values=1), 2
                )
                self.dim = 1
            else:
                self.size = size.copy()
                eff_dim = (size > 1)  # * (np.array(lc) > 0)
                self.dim = size[eff_dim].size
            if shape == 'ring':
                self.dim = 2

        if isinstance(size, Iterable):
            size = np.array(size)

        self.nodes, self.links = build_lattice(size, shape, symmetry)
        self.reflect, self.inv_coords = build_reflection(
            self.nodes, symmetry)
        # Adjust site number after adjust lattice by symmetry
        self.N = self.nodes.shape[0]
        # Independent trap number under reflection symmetry
        self.Nindep = self.reflect.shape[0]


def build_lattice(size, shape, symmetry):
    if np.prod(size) == 1:
        nodes = np.array([[0, 0]])
        links = np.array([]). reshape(0, 2)
    elif shape == 'ring':
        nodes, links = ring_lattice(size[0])
    elif shape == 'square':
        nodes, links, __ = sqr_lattice(size)
    elif shape == 'Lieb':
        nodes, links = Lieb_lattice(size)
    elif shape == 'triangular':
        nodes, links, __ = tri_lattice(size, symmetry)
    elif shape == 'zigzag':
        # Tune lcy != sqrt(3)/2 * lcx to get zigzag from triangular ladder
        # TODO: delete horizontal links
        symmetry = False
        nodes, links, __ = tri_lattice(
            np.array([size[0], 2], dtype=int), symmetry)
        links = links[abs(links[:, 0] - links[:, 1]) != 1]
    elif shape == 'honeycomb':
        nodes, links = hc_lattice(size)
    elif shape == 'defecthoneycomb':
        # Stone-Wales defect
        # inspired by 10.1103/PhysRevE.93.042132
        # and 10.1109/ICONSET.2011.6167932
        nodes, links = defect_hc_lattice(size)
    elif shape == 'kagome':
        nodes, links = kagome_lattice(size)
    elif shape == 'Penrose':
        # Not useful as the diagonal distance is smaller than the bond length
        # Not yet finished
        nodes, links = penrose_tiling(size[0])
    else:
        raise ValueError(f'Lattice: Unknown shape {shape}.')
    return nodes, links


def ring_lattice(size: int) -> np.ndarray:
    # Generate coordinates of 4n points on a ring,
    # with each pair of sites separated by 1

    # Construct (x>0, y>0), then reflect to the other quadrant
    # Indexing is COUNTER-CLOCKWISE
    # Adjust size to 4n
    n = size // 4
    size = 4 * n
    print(f'Ring size adjust to: {size}')
    theta = np.pi / size
    radius = 0.5 / np.sin(theta)
    nodes = np.array([]).reshape(0, 2)
    links = np.array([], dtype=int).reshape(0, 2)

    sectors = [
        np.array([1, 1]),
        np.array([-1, 1]),
        np.array([-1, -1]),
        np.array([1, -1])
    ]
    for idx in range(len(sectors)):
        nodes = quadrant_ring(n, theta, radius, idx, sectors, nodes)
    links = np.array([[i, (i + 1) % size] for i in range(size)])
    return nodes, links


def quadrant_ring(n, theta, radius, idx, sectors, nodes):
    sec = sectors[idx]
    for i in range(n):
        coord = radius * sec * np.array(
            [np.sin(theta * (2 * i + 1)),
             np.cos(theta * (2 * i + 1))])
        nodes = np.append(nodes, coord[None], axis=0)
    return nodes


def penrose_tiling(size: int) -> np.ndarray:
    # Generate coordinates of n layers on a Penrose tiling,
    # with each pair of sites separated by 1
    # Indexing is COUNTER-CLOCKWISE

    n = size
    print(f'Penrose size adjust to: {size}')
    __, nodes = penrose_triangles(n)
    links = np.array([[i, (i + 1) % size] for i in range(size)])
    return nodes, links


def penrose_triangles(layer: int) -> np.ndarray:
    # Generate coordinates of n layers on a Penrose tiling,
    # with each pair of sites separated by 1
    # Indexing is COUNTER-CLOCKWISE

    # Compute the lateral length of the triangle
    r = golden_ratio**(layer - 1)
    # Create wheel of red triangles around the origin
    fold = 10
    triangles = np.empty((fold, 4), dtype=complex)
    nodes = np.zeros((1, 2), dtype=float)
    for i in range(fold):
        B = cmath.rect(r, (2*i - 1) * np.pi / fold)
        C = cmath.rect(r, (2*i + 1) * np.pi / fold)
        nodes = np.append(nodes, np.array([B.real, B.imag])[None], axis=0)
        if i % 2 == 0:
            B, C = C, B  # Make sure to mirror every second triangle
        triangles[i, :] = np.array([0, 0j, B, C])
    print('1st layer done.')
    for i in range(layer - 1):
        triangles, nodes = subdivide(triangles, nodes)
        print(f'{i+2}th layer done.')
    return triangles, nodes


golden_ratio = (1 + np.sqrt(5)) / 2


def subdivide(triangles: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    result = np.empty((0, 4), dtype=complex)
    for color, A, B, C in triangles:
        if color == 0:
            # Subdivide red triangle
            P = A + (B - A) / golden_ratio
            nodes = np.append(nodes, np.array([P.real, P.imag])[None], axis=0)
            result = np.append(
                result, [(0, C, P, B), (1, P, C, A)], axis=0)
        else:
            # Subdivide blue triangle
            Q = B + (A - B) / golden_ratio
            R = B + (C - B) / golden_ratio
            nodes = np.concatenate(
                (nodes, np.array([Q.real, Q.imag], [R.real, R.imag])), axis=0)
            result = np.append(
                result, [(1, R, C, A), (1, Q, R, B), (0, R, Q, A)], axis=0)
    return result, nodes


def sqr_lattice(size: np.ndarray):
    # Square and rectangular lattice graph builder
    # Also including 1D open chain
    # NOTE: In this case, indexing is COLUMN-MAJOR.

    edge = []
    edge_idx = []
    nodes = np.array([]).reshape(0, 2)
    node_idx_pair = np.array([]).reshape(0, 2)
    links = np.array([], dtype=int).reshape(0, 2)

    for i in range(size.size):
        edge.append(np.arange(-(size[i] - 1) / 2, (size[i] - 1) / 2 + 1))
        edge_idx.append(np.arange(1, size[i] + 1, dtype=int))

    node_idx = 0  # Linear index is column (y) prefered
    for i in range(len(edge[0])):
        for j in range(len(edge[1])):
            nodes = np.append(nodes, [[edge[0][i], edge[1][j]]], axis=0)

            node_idx_pair = np.append(node_idx_pair,
                                      [[edge_idx[0][i], edge_idx[1][j]]],
                                      axis=0)
            if i > 0:  # Row link
                links = np.append(links, [[node_idx - size[1], node_idx]],
                                  axis=0)
            if j > 0:  # Column linke
                links = np.append(links, [[node_idx - 1, node_idx]], axis=0)
            node_idx += 1
    return nodes, links, node_idx_pair


def Lieb_lattice(size):
    # Build Lieb lattice graph
    size[size < 3] == 3  # Smallest Lieb lattice plaquette has size 3
    size[size % 2 == 0] += 1  # Make sure size is odd
    print(f'Lieb lattice size adjust to: {size}')
    nodes, links, node_idx_pair = sqr_lattice(size)
    # Remove holes from square lattice to make Lieb lattice
    hole = np.all(node_idx_pair % 2 == 0, axis=1)
    hole_idx = np.nonzero(hole)[0]
    nodes = nodes[~hole, :]
    links = squeeze_idx(links, hole_idx)
    return nodes, links


def tri_lattice(size: np.ndarray,
                major_centered: bool = True,
                symmetry: bool = True):
    # Triangular lattice graph builder
    # NOTE: lc = (ax, ay) is given by hand.
    #       For equilateral triangle,
    #       x direction unit is ax,
    #       y direction unit is ay = sqrt(3)/2 * ax.
    #       Other case is isosceles triangle.
    #       In this function, length unit is (ax, ay).
    # NOTE: In this case, indexing is ROW-MAJOR.
    # NOTE that if given symmetry=True,
    # the triangular lattice is made to he reflection symmetric.
    # E.g. if major_centered=True, the lattice is made to be
    #      * * * * *
    #     * * * * * *
    #      * * * * *
    #      else
    #     * * * * * *
    #      * * * * *
    #     * * * * * *
    # If given symmetry=False, the lattice is in the parallelogram shape.
    # E.g. * * * * *
    #       * * * * *

    if symmetry:
        # Smallest reflection-symmetric triangular lattice plaquette has size 3
        size[size < 3] == 3
        # Make sure y dimension size is odd
        if size[1] % 2 == 0:
            size[1] += 1
    else:
        # Smallest triangular lattice plaquette has size 2 on each direction
        size[size < 2] == 2
        major_centered = True  # No effect
    print(f'Triangular lattice size adjust to: {size}')

    edge = []
    edge_idx = []
    nodes = np.array([]).reshape(0, 2)
    node_idx_pair = np.array([]).reshape(0, 2)
    links = np.array([], dtype=int).reshape(0, 2)

    for i in range(size.size):  # Add major row., size[0] is major row size
        edge.append(np.arange(-(size[i] - 1) / 2, (size[i] - 1) / 2 + 1))
        edge_idx.append(np.arange(1, size[i] + 1, dtype=int))

    if symmetry:  # For reflection symmetry, add minor row w/ L-1 sites
        edge.append(np.arange(-size[0] / 2 + 1, size[0] / 2))
        edge_idx.append(np.arange(1, size[0], dtype=int))
    else:  # For no symmetry, add shifted major row w/ L sites
        edge.append(edge[0] - 0.25)
        edge[0] += 0.25
        edge_idx.append(np.arange(1, size[0] + 1, dtype=int))

    node_idx = 0  # Linear index is row (x) prefered
    for j in range(len(edge[1])):
        # Determine major row & minor row
        if major_centered:
            # Even row is major row
            major_row = j % 2 == 1
        else:
            # Odd row is major row
            major_row = j % 2 == 0
        edge_i, edge_idx_i = (edge[0], edge_idx[0]) if major_row else (
            edge[-1], edge_idx[-1])
        for i in range(len(edge_i)):
            nodes = np.append(nodes, [[edge_i[i], edge[1][j]]], axis=0)
            node_idx_pair = np.append(node_idx_pair,
                                      [[edge_idx_i[i], edge_idx[1][j]]],
                                      axis=0)
            if i > 0:  # Row link
                links = np.append(links, [[node_idx - 1, node_idx]], axis=0)
            if j > 0:  # Column link
                if symmetry:
                    if major_row:  # Major row
                        if i > 0:  # Leftward link
                            links = np.append(links,
                                              [[node_idx - size[0], node_idx]],
                                              axis=0)
                        if i < len(edge[0]) - 1:  # Righttward link
                            links = np.append(links,
                                              [[node_idx - size[0] + 1, node_idx]],
                                              axis=0)
                    else:  # Minor row
                        links = np.append(links, [[node_idx - size[0], node_idx]],
                                          axis=0)  # Leftward link
                        links = np.append(links,
                                          [[node_idx - size[0] + 1, node_idx]],
                                          axis=0)  # Righttward link
                else:
                    links = np.append(links,
                                      [[node_idx - size[0], node_idx]],
                                      axis=0)  # Leftward link
                    if i < len(edge[0]) - 1:  # Righttward link
                        links = np.append(links,
                                          [[node_idx - size[0] + 1, node_idx]],
                                          axis=0)
            node_idx += 1

    return nodes, links, node_idx_pair


def hc_lattice(size, lesshole: bool = True):
    # Smallest reflection-symmetric honeycomb lattice plaquette has size 3
    size[size < 3] == 3
    # Make sure x dimension size is integer multiple of 3
    if size[0] % 3 != 0:
        size[0] += 3 - size[0] % 3
    print(f'Honeycomb lattice size adjust to: {size}')
    nodes, links, node_idx_pair = tri_lattice(
        size, major_centered=lesshole, symmetry=True)
    # Remove holes from square lattice to make honeycomb lattice
    # if lesshole, in a way that less atoms are removed from the lattice

    # Choose which one is major row
    # NOTE: idx starts from 1
    major_row = node_idx_pair[:, 1] % 2 == 0
    if not lesshole:
        major_row = ~major_row
    hole = np.logical_and(major_row,
                          node_idx_pair[:, 0] % 3 == 2)
    hole = np.logical_or(hole,
                         np.logical_and(np.logical_not(major_row),
                                        node_idx_pair[:, 0] % 3 == 0))
    hole_idx = np.nonzero(hole)[0]
    nodes = nodes[~hole, :]
    links = squeeze_idx(links, hole_idx)
    return nodes, links


def defect_hc_lattice(size):
    # Reflection-symmetric Stone-Wales defect honeycomb lattice
    nodes, links = hc_lattice(size, lesshole=False)
    Nsite = len(nodes)
    site1 = Nsite // 2 - 1
    site2 = Nsite // 2
    # Rotate site1 & site2 by pi/2
    # NOTE: since ax != ay, the trap spacing might be changed
    nodes[site1:site1+2, :] = nodes[site1:site1+2, :][:, ::-1]
    # Reconnect links
    site1_obliq_links = np.logical_and(
        np.any(links == site1, axis=1), abs(links[:, 0] - links[:, 1]) != 1)
    site2_obliq_links = np.logical_and(
        np.any(links == site2, axis=1), abs(links[:, 0] - links[:, 1]) != 1)
    site_to_link_site1 = links[site1_obliq_links][0, 0] + 1
    site_to_link_site2 = links[site2_obliq_links][-1, -1] - 1
    links[np.nonzero(site1_obliq_links)[0][-1], -1] = site_to_link_site1
    links[np.nonzero(site2_obliq_links)[0][0], 0] = site_to_link_site2
    return nodes, links


def kagome_lattice(size):
    # Smallest reflection-symmetric kagome lattice plaquette has x size 4,
    # y size 5 (wchich will be automatically adjusted to be odd)
    size[size < 4] == 4
    # Reflection-symmetric kagome lattice always needs y size to be 4n+1
    if size[1] % 4 != 1:
        size[1] = 4 * (size[1] // 4) + 1
        # Make sure x dimension size is even
    if size[0] % 2 != 0:
        size[0] += 1
    print(f'Kagome lattice size adjust to: {size}')
    nodes, links, node_idx_pair = tri_lattice(size, symmetry=True)
    # Remove holes from square lattice to make kagome lattice
    hole = np.logical_and(node_idx_pair[:, 1] % 4 == 1,
                          node_idx_pair[:, 0] % 2 == 1)
    hole = np.logical_or(
        hole,
        np.logical_and(node_idx_pair[:, 1] % 4 == 3,
                       node_idx_pair[:, 0] % 2 == 0))
    hole_idx = np.nonzero(hole)[0]
    nodes = nodes[~hole, :]
    links = squeeze_idx(links, hole_idx)
    return nodes, links


def squeeze_idx(links: np.ndarray, hole_idx: np.ndarray) -> np.ndarray:
    # Shift indices of nodes in links to match the new graph
    links = links[~np.any(np.isin(links, hole_idx),
                          axis=1), :]  # Remove links to holes
    if len(links) == 0:  # No links left
        print('squeeze_idx: No links left')
        return links
    max_idx = np.max(links)
    idx_wo_hole = np.arange(max_idx + 1, dtype=int)
    idx_wo_hole = idx_wo_hole[np.isin(idx_wo_hole, hole_idx) == False]
    shift_no = np.array([sum(i > hole_idx) for i in idx_wo_hole])
    loolup_table = np.concatenate((idx_wo_hole[:, None], shift_no[:, None]),
                                  axis=1)
    for idx in range(loolup_table.shape[0]):
        links[links == loolup_table[idx, 0]] -= loolup_table[idx, 1]
    return links


def build_reflection(graph: np.ndarray, symmetry: bool = True):
    # Build correspondence map of 4-fold reflection sectors in 1D & 2D lattice
    # Entries are site labels, each row is a symmetry equiv class
    # with 4 columns sites from each other
    # Eg. : p=1, m=-1
    #     [pp mp pm mm] equiv pts 1
    #     [pp mp pm mm] equiv pts 2
    #     [pp mp pm mm] equiv pts 3
    #     [pp mp pm mm] equiv pts 4
    #     ...
    # TODO: consider what we can do with multi-fold rotations

    if symmetry:
        nsec = 4  # Number of sectors
        reflection = np.array([], dtype=int).reshape(0, nsec)
        inv_coords = np.array([], dtype=bool).reshape(0, 2)
        for i in range(graph.shape[0]):
            if all(graph[i, :] <= 0):
                pp = i  # [1 1] sector
                mp = np.nonzero(
                    np.prod(np.array([[-1, 1]]) * graph == graph[i, :],
                            axis=1))[0][0]  # [-1 1] sector
                pm = np.nonzero(
                    np.prod(np.array([[1, -1]]) * graph == graph[i, :],
                            axis=1))[0][0]  # [1 -1] sector
                mm = np.nonzero(
                    np.prod(np.array([[-1, -1]]) * graph == graph[i, :],
                            axis=1))[0][0]  # [-1 -1] sector
                reflection = np.append(reflection, [[pp, mp, pm, mm]], axis=0)
                inv_coords = np.append(
                    inv_coords, np.array([graph[i, :] == 0]), axis=0)
    else:
        reflection = np.arange(graph.shape[0])[:, None]
        inv_coords = np.zeros((graph.shape[0], 2), dtype=bool)
    return reflection, inv_coords
