import numpy as np
import matplotlib.pyplot as plt
import graphviz as gv
import matplotlib as mpl

from .equalizer import *


class HubbardGraph(HubbardEqualizer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.graph = gv.Graph('lattice', engine='dot')
        self.edge_fontsize = '4'
        self.edge_fontcolor = '#000066'
        self.graph.attr('node', shape='circle', style='filled', fixedsize='shape',
                        color='#99CCFF',
                        fontname='Meiryo', fontcolor='#FF8000', fontsize='6')
        self.graph.attr('edge', penwidth='2',
                        fontname='Meiryo', fontcolor=self.edge_fontcolor, fontsize=self.edge_fontsize)
        self.edges = self.links.copy()
        self.Nedge = self.edges.shape[0]

    def update_node(self, label='param'):
        if label == 'param':
            # Label onsite chemical potential
            depth = np.real(np.diag(self.A))
            self.node_label = list(
                f'{depth[i]:.3g}\n{self.U[i]:.3g}' for i in range(self.Nsite))
            self.node_label[0] = f'V = {depth[0]:.3g}\n U = {self.U[0]:.3g}'
        elif label == 'adjust':
            # Label trap offset
            self.node_label = list(
                f'{v:.3g}' for v in self.Voff)
            # self.node_label[0] = f'V0 = {self.Voff[0]:.3g}'
        self.node_size = [i * 0.3 for i in self.waists]
        max_depth = np.max(abs(self.Voff))
        self.node_alpha = (self.Voff / max_depth) ** 10

    def update_edge(self, label='param'):
        self.edge_weight = np.zeros(self.Nedge)
        for i in range(self.Nedge):
            edge = self.edges[i]
            if label == 'param':
                # Label bond tunneling
                length = abs(self.A[edge[0], edge[1]])
            elif label == 'adjust':
                # Label bond length
                length = la.norm(np.diff(self.trap_centers[edge, :], axis=0))
            self.edge_weight[i] = length
        self.edge_label = list(
            f'{self.edge_weight[i]:.3g}'
            for i in range(self.Nedge))
        if label == 'param':
            self.edge_label[0] = f't = {self.edge_weight[0]:.3g}'
        max_len = max(self.edge_weight)
        self.edge_alpha = list(
            int(255 * np.sqrt(self.edge_weight[i] / max_len)) for i in range(self.Nedge))

    def add_nnn(self, center=0, limit=3):
        # Add higher neighbor bonds
        # NOTE: explicit square lattice geometry somewhat assumed
        if limit + 2 > self.Nsite:
            limit = self.Nsite - 2
        if center >= self.Nsite:
            center = 0
        self.invis_nodes = np.array([]).reshape(0, 2)
        if self.lattice_dim == 2:
            limit *= 2
        for i in range(limit):
            link = np.array([center, center + i + 2])
            if np.any(np.all(link == self.links, axis=1)):
                continue
            self.edges = np.append(
                self.edges, link[None], axis=0)
            shifted_center = self.shift_node(center, i)
            self.invis_nodes = np.append(
                self.invis_nodes, shifted_center[None], axis=0)
        self.Nedge = self.edges.shape[0]

    def shift_node(self, center, i):
        shifted_center = (self.trap_centers[center,
                                            :] + self.trap_centers[center + i + 2, :])/2
        diff_center = (self.trap_centers[center,
                                         :] - self.trap_centers[center + i + 2, :])/2
        shifted_dim = np.nonzero(abs(diff_center) < 1e-5)
        if len(shifted_dim) == 1:
            shifted_dim = shifted_dim[0]
            shifted_center[shifted_dim] += 0.25
        return shifted_center

    def singleband_params(self, label='param', A=None, U=None):
        if label == 'param' and (A is None or U is None):
            self.singleband_Hubbard(u=True)
        elif label == 'adjust' and A is None:
            self.singleband_Hubbard(u=False)

    def plot_edge(self):
        j = 0
        for i in range(self.Nedge):
            self.graph.attr('edge', penwidth='2', fontname='Meiryo',
                            fontcolor='#000066', fontsize='6')
            edge = self.edges[i]
            color = '#606060' + f'{self.edge_alpha[i]:2x}'
            if i < self.links.shape[0]:
                # Add n.n. bonds
                self.graph.attr('edge', style='solid', splines='false')
                self.graph.edge(f'{edge[0]}', f'{edge[1]}',
                                alpha=f'{self.edge_alpha[i]}',
                                label=self.edge_label[i],
                                color=color)
            else:
                # self.graph.attr('edge', splines='compound')
                # self.graph.edge(f'{edge[0]}', f'{edge[1]}',
                #                 alpha=f'{self.edge_alpha[i]}',
                #                 label=self.edge_label[i],
                #                 color=color)
                # Add invisible node to construct longer bond edges
                if self.edge_weight[i] > 1e-4:
                    self.graph.attr('edge', splines='true')
                    self.graph.node(
                        f'invis_{edge[0]}{edge[1]}', pos=f'{self.invis_nodes[j, 0]},{self.invis_nodes[j, 1]}!', shape="circle", width="0", fixedsize='shape', color='invis', fontcolor=self.edge_fontcolor, footsize=self.edge_fontsize, label=self.edge_label[i])
                    self.graph.edge(f'{edge[0]}', f'invis_{edge[0]}{edge[1]}',
                                    alpha=f'{self.edge_alpha[i]}',
                                    color=color)
                    self.graph.edge(f'invis_{edge[0]}{edge[1]}', f'{edge[1]}',
                                    alpha=f'{self.edge_alpha[i]}',
                                    color=color)
                j += 1

    def plot_node(self):
        for i in range(self.Nsite):
            pos = self.trap_centers[i, :]
            self.graph.node(
                f'{i}', label=self.node_label[i], pos=f'{pos[0]},{pos[1]}!',
                width=f'{self.node_size[i][0]}', height=f'{self.node_size[i][1]}', alpha=f'{self.node_alpha[i]}')

    def draw_graph(self, label='param', nnn=False, A=None, U=None):
        self.singleband_params(label, A, U)
        self.update_node(label)
        if label == 'param' and nnn:
            self.add_nnn()
        self.update_edge(label)

        if self.verbosity:
            print('\nStart to plot graph...')

        self.plot_node()
        self.plot_edge()

        self.graph.render(
            f'{self.lattice} graphviz {self.dim}d {self.lattice_shape} {label} {self.waist_dir} {self.eq_label}.pdf')
