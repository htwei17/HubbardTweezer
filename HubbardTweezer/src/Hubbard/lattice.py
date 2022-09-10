from typing import Iterable
import numpy as np


def lattice_graph(size: np.ndarray,
                  shape: str = 'square') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Square lattice graph builder
    # shape: 'ring' 'square' 'Lieb' 'triangle' 'honeycomb' 'kagome'
    # NOTE: might not be very doable since the symmetries of trap are just x,y mirrors
    # nodes: each row is a coordinate (x, y) of one site
    #        indicating the posistion of node (trap center)
    # links: each row in links is a pair of node indices s.t.
    #        graph[idx1], graph[idx2] are linked by bounds

    if isinstance(size, Iterable):
        size = np.array(size)

    if shape == 'ring':
        nodes, links = ring_coord(size[0])
    elif shape == 'square':
        nodes, links, __ = sqr_lattice(size)
    elif shape == 'Lieb':
        # Build Lieb lattice graph
        size[size < 3] == 3  # Smallest Lieb lattice plaquette has size 3
        size[size % 2 == 0] += 1  # Make sure size is odd
        print(f'Lieb lattice size adjust to: {size}')
        nodes, links, node_idx_pair = sqr_lattice(size)
        # Remove holes from square lattice to make Lieb lattice
        hole = np.all(node_idx_pair % 2 == 0, axis=1)
        hole_idx = np.nonzero(hole)[0]
        nodes = nodes[~hole, :]
        links = shift_links(links, hole_idx)
    elif shape == 'triangular':
        nodes, links, __ = tri_lattice(size)
    elif shape == 'zigzag':
        # TODO: implement zigzag, and other assymmetric lattice
        # FIXME: zigzag has no y-axis symmetry
        #        one way to do is to copy the chain to be mirrored on y-axis
        # NOTE: the code below is not runnable as size-2 will be forced to be 3
        nodes, links, __ = tri_lattice(np.array([size[0], 2]))
    elif shape == 'honeycomb':
        # Smallest reflection-symmetric honeycomb lattice plaquette has size 3
        size[size < 3] == 3
        # Make sure x dimension size is integer multiple of 3
        if size[0] % 3 != 0:
            size[0] += 3 - size[0] % 3
        print(f'Honeycomb lattice size adjust to: {size}')
        nodes, links, node_idx_pair = tri_lattice(size)
        # Remove holes from square lattice to make honeycomb lattice
        hole = np.logical_and(node_idx_pair[:, 1] % 2 == 0,
                              node_idx_pair[:, 0] % 3 == 2)
        hole = np.logical_or(
            hole,
            np.logical_and(node_idx_pair[:, 1] % 2 == 1,
                           node_idx_pair[:, 0] % 3 == 0))
        hole_idx = np.nonzero(hole)[0]
        nodes = nodes[~hole, :]
        links = shift_links(links, hole_idx)
    elif shape == 'kagome':
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
        nodes, links, node_idx_pair = tri_lattice(size)
        # Remove holes from square lattice to make kagome lattice
        hole = np.logical_and(node_idx_pair[:, 1] % 4 == 1,
                              node_idx_pair[:, 0] % 2 == 1)
        hole = np.logical_or(
            hole,
            np.logical_and(node_idx_pair[:, 1] % 4 == 3,
                           node_idx_pair[:, 0] % 2 == 0))
        hole_idx = np.nonzero(hole)[0]
        nodes = nodes[~hole, :]
        links = shift_links(links, hole_idx)

    reflection, inv_coords = build_reflection(nodes, shape)
    # TODO: consider what we can do with multi-fold rotations
    return nodes, links, reflection, inv_coords


def ring_coord(size: int) -> np.ndarray:
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


def tri_lattice(size: np.ndarray):
    # Triangular lattice graph builder
    # NOTE: lc = (ax, ay) is given by hand.
    #       For equilateral triangle,
    #       x direction unit is ax,
    #       y direction unit is ay = sqrt(3)/2 * ax.
    #       Other case is isosceles triangle.
    #       In this function, length unit is (ax, ay).
    # NOTE: In this case, indexing is ROW-MAJOR.
    # NOTE that the triangular lattice is made to he reflection symmetric.
    # E.g. * * * * *
    #     * * * * * *
    #      * * * * *

    # Smallest reflection-symmetric triangular lattice plaquette has size 3
    size[size < 3] == 3
    # Make sure y dimension size is odd
    if size[1] % 2 == 0:
        size[1] += 1
    print(f'Triangular lattice size adjust to: {size}')

    edge = []
    edge_idx = []
    nodes = np.array([]).reshape(0, 2)
    node_idx_pair = np.array([]).reshape(0, 2)
    links = np.array([], dtype=int).reshape(0, 2)

    for i in range(size.size):
        edge.append(np.arange(-(size[i] - 1) / 2, (size[i] - 1) / 2 + 1))
        edge_idx.append(np.arange(1, size[i] + 1, dtype=int))

    # Minor rows that have smaller size than major rows
    edge.append(np.arange(-size[0] / 2 + 1, size[0] / 2))
    edge_idx.append(np.arange(1, size[0], dtype=int))

    node_idx = 0  # Linear index is row (x) prefered
    for j in range(len(edge[1])):
        if j % 2 == 1:
            edge_i = edge[0]
            edge_idx_i = edge_idx[0]
        else:
            edge_i = edge[-1]
            edge_idx_i = edge_idx[-1]
        for i in range(len(edge_i)):
            nodes = np.append(nodes, [[edge_i[i], edge[1][j]]], axis=0)

            node_idx_pair = np.append(node_idx_pair,
                                      [[edge_idx_i[i], edge_idx[1][j]]],
                                      axis=0)
            if i > 0:  # Row link
                links = np.append(links, [[node_idx - 1, node_idx]], axis=0)
            if j > 0:  # Column linke
                if j % 2 == 1:
                    if i > 0:
                        links = np.append(links,
                                          [[node_idx - size[0], node_idx]],
                                          axis=0)
                    if i < len(edge[0]) - 1:
                        links = np.append(links,
                                          [[node_idx - size[0] + 1, node_idx]],
                                          axis=0)
                else:
                    links = np.append(links, [[node_idx - size[0], node_idx]],
                                      axis=0)
                    links = np.append(links,
                                      [[node_idx - size[0] + 1, node_idx]],
                                      axis=0)
            node_idx += 1

    return nodes, links, node_idx_pair


def shift_links(links: np.ndarray, hole_idx: np.ndarray) -> np.ndarray:
    # Shift indices of nodes in links to match the new graph
    links = links[~np.any(np.isin(links, hole_idx),
                          axis=1), :]  # Remove links to holes
    max_idx = np.max(links)
    idx_wo_hole = np.arange(max_idx + 1, dtype=int)
    idx_wo_hole = idx_wo_hole[np.isin(idx_wo_hole, hole_idx) == False]
    shift_no = np.array([sum(i > hole_idx) for i in idx_wo_hole])
    loolup_table = np.concatenate((idx_wo_hole[:, None], shift_no[:, None]),
                                  axis=1)
    for idx in range(loolup_table.shape[0]):
        links[links == loolup_table[idx, 0]] -= loolup_table[idx, 1]
    return links


def build_reflection(graph, shape='c4'):
    # Build correspondence map of 4-fold reflection sectors in 1D & 2D lattice
    # Entries are site labels, each row is a symmetry equiv class
    # with 4 columns sites from each other
    # Eg. : p=1, m=-1
    #     [pp mp pm mm] equiv pts 1
    #     [pp mp pm mm] equiv pts 2
    #     [pp mp pm mm] equiv pts 3
    #     [pp mp pm mm] equiv pts 4
    #     ...

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
    return reflection, inv_coords
