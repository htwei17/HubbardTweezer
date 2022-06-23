from os import link
from typing import Iterable
import numpy as np


def lattice_graph(size: np.ndarray,
                  shape: str = 'square') -> tuple[np.ndarray, np.ndarray]:
    # Square lattice graph builder
    # shape: 'square' or 'Lieb'
    # TODO: add 'triangular', 'hexagonal', 'kagome' lattices, make juse of their more complicated symmetries
    # TODO: add function to equalize Lieb and other lattices
    # NOTE: might not be very doable since the symmetries of trap are just x,y mirrors
    # nodes: each row is a coordinate (x, y) of one site
    #        indicating the posistion of node (trap center)
    # links: each row in links is a pair of node indices s.t.
    #        graph[idx1], graph[idx2] are linked by bounds

    if isinstance(size, Iterable):
        size = np.array(size)

    if shape == 'square':
        # Square and rectangular lattice graph
        nodes, links, __, __ = sqr_lattice(size)
    elif shape == 'Lieb':
        # Build Lieb lattice graph
        size[size == 1] == 3  # Smallest Lieb lattice plaquette has size 3
        size[size % 2 == 0] += 1  # Make sure size is odd
        print(f'Lieb size adjust to: {size}')
        nodes, links, node_idx, node_idx_pair = sqr_lattice(size)
        # Remove holes from square lattice to make Lieb lattice
        Lieb_hole = np.all(node_idx_pair % 2 == 0, axis=1)
        Lieb_hole_idx = np.nonzero(Lieb_hole)[0]
        nodes = nodes[~Lieb_hole, :]
        links = shift_links(links, Lieb_hole_idx)

    reflection = build_reflection(nodes, shape)
    # TODO: consider what we can do with multi-fold rotations
    return nodes, links, reflection


def sqr_lattice(size: np.ndarray):
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
            if i > 0:
                links = np.append(links, [[node_idx - size[1], node_idx]],
                                  axis=0)  # Row link
            if j > 0:
                links = np.append(links, [[node_idx - 1, node_idx]],
                                  axis=0)  # Column linke
            node_idx += 1
    return nodes, links, node_idx, node_idx_pair


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


def build_reflection(graph, shape='square'):
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
    return reflection
