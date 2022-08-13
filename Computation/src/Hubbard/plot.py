import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

from .equalizer import *

# params = {
#     'figure.dpi': 300,
#     # 'figure.figsize': (15, 5),
#     'legend.fontsize': 'x-large',
#     'axes.labelsize': 'xx-large',
#     'axes.titlesize': 'xx-large',
#     'xtick.labelsize': 'xx-large',
#     'ytick.labelsize': 'xx-large'
# }
# mpl.rcParams.update(params)


class HubbardGraph(HubbardParamEqualizer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # # Resize [n] to [n, 1]
        # self.lattice = np.resize(
        #     np.pad(self.lattice, pad_width=(0, 1), constant_values=1), 2)
        self.edges = [tuple(row) for row in self.links]
        self.graph = nx.DiGraph(self.edges, name='Lattice')
        self.pos = dict(
            # (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
            (n, self.trap_centers[n]) for n in self.graph.nodes())

    def update_edge_weight(self, label='param'):
        for link in self.graph.edges:
            if label == 'param':
                # Label bond tunneling
                length = abs(self.A[link[0], link[1]])
            elif label == 'adjust':
                # Label bond length
                length = la.norm(np.diff(self.trap_centers[link, :], axis=0))
            self.graph[link[0]][link[1]]['weight'] = length
        self.edge_label = dict(
            (edge, f'{self.graph[edge[0]][edge[1]]["weight"]:.3g}')
            for edge in self.graph.edges)
        max_len = max(dict(self.graph.edges).items(),
                      key=lambda x: x[1]["weight"])[-1]["weight"]
        self.edge_alpha = [
            self.graph[edge[0]][edge[1]]["weight"] / max_len
            for edge in self.graph.edges
        ]

    def update_node_weight(self, label='param'):
        if label == 'param':
            # Label onsite chemical potential
            depth = np.real(np.diag(self.A))
            self.node_label = dict(
                (n, f'{depth[n]:.3g}') for n in self.graph.nodes)
        elif label == 'adjust':
            # Label trap offset
            self.node_label = dict(
                (n, f'{self.Voff[n]:.3g}') for n in self.graph.nodes)
        self.node_size = [i**10 * 600 for i in self.Voff]

    def add_nnn(self, center=0, limit=3):
        # Add higher neighbor bonds
        # NOTE: explicit square lattice geometry somewhat assumed
        # FIXME: 3x2 lattice error as this gives an index 6
        if limit + 2 > self.Nsite:
            limit = self.Nsite - 2
        if center >= self.Nsite:
            center = 0
        if self.lattice_dim == 1:
            for i in range(limit):
                self.graph.add_edge(center, i + 2)
        elif self.lattice_dim == 2:
            for i in range(2 * limit):
                self.graph.add_edge(center, i + 2)

    def singleband_params(self, label='param', A=None, U=None):
        if label == 'param' and (A == None or U == None):
            self.singleband_Hubbard(u=True)
        elif label == 'adjust' and A == None:
            self.singleband_Hubbard(u=False)

    def draw_graph(self, label='param', nnn=False, A=None, U=None):
        self.singleband_params(label, A, U)
        if label == 'param' and nnn:
            self.add_nnn()
        self.update_edge_weight(label)
        self.update_node_weight(label)

        if self.verbosity:
            print('\nStart to plot graph...')
        if self.lattice_dim == 1:
            fs = (self.lattice[0] * 2, self.lattice[1] * 6)
        elif self.lattice_dim == 2:
            fs = tuple(2 * i for i in self.lattice)
        f = plt.figure(figsize=fs)
        nx.draw_networkx_nodes(self.graph,
                               pos=self.pos,
                               node_color='#99CCFF',
                               node_size=self.node_size)
        nx.draw_networkx_labels(self.graph,
                                pos=self.pos,
                                font_color='#000066',
                                font_size=8,
                                labels=self.node_label)
        link_list = list(self.graph.edges)
        for i in range(len(link_list)):
            el = link_list[i]
            cs = "arc3"
            if not any((el == self.links).all(axis=1)):
                cs = "arc3,rad=0.2"
            nx.draw_networkx_edges(self.graph,
                                   self.pos,
                                   arrows=True,
                                   arrowstyle='-',
                                   edgelist=[el],
                                   edge_color='#606060',
                                   connectionstyle=cs,
                                   label=self.edge_label[link_list[i]],
                                   alpha=np.sqrt(self.edge_alpha[i]),
                                   width=3)
        nx.draw_networkx_edge_labels(self.graph,
                                     self.pos,
                                     font_size=10,
                                     edge_labels=self.edge_label,
                                     font_color=[0.256, 0.439, 0.588])
        if label == 'param':
            self.draw_node_overhead_labels(font_size=10, font_color='#FF8000')
        plt.axis('off')
        plt.savefig(
            f'{self.lattice} graph {self.dim}d {label} {self.eq_label}.pdf')

    def draw_node_overhead_labels(
            self,
            font_size=12,
            font_color="k",
            font_family="sans-serif",
            font_weight="normal",
            alpha=None,
            bbox=None,
            #  bbox=dict(facecolor='red', alpha=0.5)
            horizontalalignment="center",
            verticalalignment="center",
            ax=None):
        if ax is None:
            ax = plt.gca()
        if self.lattice_dim == 1:
            offset = (0, 0)
        elif self.lattice_dim == 2:
            offset = (-0.1, 0.1)

        for i in range(self.Nsite):
            x, y = self.pos[i]
            ax.text(x + offset[0],
                    y + offset[1],
                    s=f'{self.U[i]:.3g}',
                    size=font_size,
                    color=font_color,
                    family=font_family,
                    weight=font_weight,
                    alpha=alpha,
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    bbox=bbox,
                    transform=ax.transData)
