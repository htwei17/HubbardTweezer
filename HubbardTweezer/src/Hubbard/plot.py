import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
from scipy.stats.mstats import gmean

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


class HubbardGraph(HubbardEqualizer):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # # Resize [n] to [n, 1]
        # self.lattice = np.resize(
        #     np.pad(self.lattice, pad_width=(0, 1), constant_values=1), 2)
        self.edges = [tuple(row) for row in self.lattice.links]
        self.graph = nx.DiGraph(self.edges, name='Lattice')
        self.pos = dict(
            # (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
            (n, self.wf_centers[n]) for n in self.graph.nodes())

    def set_edges(self, label='param'):
        for link in self.graph.edges:
            if label == 'param':
                # Label bond tunneling
                length = abs(self.A[link[0], link[1]]) * 1e3  # Convert to kHz
            elif label == 'adjust':
                # Label bond length
                length = la.norm(np.diff(self.trap_centers[link, :], axis=0))
            self.graph[link[0]][link[1]]['weight'] = length
        self.edge_label = dict(
            (edge, f'{self.graph[edge[0]][edge[1]]["weight"]:.0f}')
            for edge in self.graph.edges)
        max_len = max(dict(self.graph.edges).items(),
                      key=lambda x: x[1]["weight"])[-1]["weight"]
        self.edge_alpha = np.array([
            self.graph[edge[0]][edge[1]]["weight"] / max_len
            for edge in self.graph.edges
        ])

    def set_nodes(self, label='param'):
        if label == 'param':
            # Label onsite chemical potential
            self.pos = dict(
                (n, self.wf_centers[n]) for n in self.graph.nodes())
            depth = np.real(np.diag(self.A)) * 1e3  # Convert to kHz
            self.node_label = dict(
                (n, f'{depth[n]:.0f}') for n in self.graph.nodes)
        elif label == 'adjust':
            # Label trap offset
            self.pos = dict(
                # (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
                (n, self.trap_centers[n]) for n in self.graph.nodes())
            self.node_label = dict(
                (n, f'{self.Voff[n]:.3g}') for n in self.graph.nodes)
        self.node_size = [i**2 * 600 for i in gmean(self.waists, axis=1)]
        max_depth = np.max(abs(self.Voff))
        self.node_alpha = (self.Voff / max_depth) ** 10

    def add_nnn(self, center=0, limit=3):
        # Add higher neighbor bonds
        # NOTE: explicit square lattice geometry assumed
        # FIXME: 3x2 lattice error as this gives an index 6
        if self.lattice.shape == 'zigzag':
            for i in range(min(limit, self.Nsite // 2)):
                self.graph.add_edge(i, i + 1)
        if self.lattice.shape in ['square', 'Lieb', 'triangular', 'zigzag']:
            if limit + 2 > self.Nsite:
                limit = self.Nsite - 2
            if center >= self.Nsite:
                center = 0
            if self.lattice.dim == 1:
                for i in range(limit):
                    self.graph.add_edge(center, i + 2)
            elif self.lattice.dim == 2:
                for i in range(2 * limit):
                    self.graph.add_edge(center, i + 2)
        else:
            print(
                f'WARNING: nnn not supported for {self.lattice.shape} lattice. \
                    Nothing doen.')
            return

    def singleband_params(self, label='param', A=None, U=None):
        if label == 'param' and (A is None or U is None):
            self.singleband_Hubbard(u=True)
        elif label == 'adjust' and A is None:
            self.singleband_Hubbard(u=False)
        elif label not in ['param', 'adjust']:
            raise ValueError('Invalid label.')

    def draw_graph(self, label='param', nnn=False, A=None, U=None):
        self.singleband_params(label, A, U)
        if label == 'param' and nnn:
            self.add_nnn()
        if all(abs(self.wf_centers[:, 1]) < 1e-6):
            self.lattice.dim = 1
            self.size = np.array([self.Nsite, 1])
            self.wf_centers[:, 1] = 0

        self.set_edges(label)
        self.set_nodes(label)

        if self.verbosity:
            print('\nStart to plot graph...')

        if self.lattice.dim == 1:
            fs = (3 * (self.size[0] - 1), 3)
            margins = (2e-2, 1)if nnn else (2e-2, 0.5)
        elif self.lattice.dim == 2:
            margins = (0.1, 0.15)
            fs = (3 * (self.size[0] - 1), 3 * (self.size[1] - 1))
            if self.lattice.shape == 'ring':
                fs[1] = fs[0]
        plt.figure(figsize=fs)

        self.draw_nodes(label, nnn, margins)
        self.draw_edges()

        plt.axis('off')
        plt.savefig(
            f'{self.size} nx {self.dim}d {self.lattice.shape} {label} {self.waist_dir} {self.eq_label}.pdf')

    def draw_edges(self):
        link_list = list(self.graph.edges)
        isnn = np.array([])
        self.nn_edge_label = dict()
        self.nnn_edge_label = dict()
        for i in link_list:
            isnn = np.append(isnn, any((i == self.lattice.links).all(axis=1)))
            if isnn[-1]:
                self.nn_edge_label[i] = self.edge_label[i]
            else:
                self.nnn_edge_label[i] = self.edge_label[i]
        edge_font_color = [0.256, 0.439, 0.588]
        for i in range(len(link_list)):
            el = link_list[i]
            cs = "arc3"  # rad=0, meaning straight line
            # For all further neighbor edges, use curved lines
            if not isnn[i]:
                cs = "arc3,rad=0.2"  # rad=0.2, meaning C1 to C0-C2 is 0.2 * C0-C2 distance
            nx.draw_networkx_edges(self.graph,
                                   self.pos,
                                   arrows=True,
                                   arrowstyle='-',
                                   edgelist=[el],
                                   edge_color='#606060',
                                   connectionstyle=cs,
                                   alpha=np.sqrt(self.edge_alpha[i]),
                                   width=3)
        self.draw_edge_labels(self.pos,
                              self.nn_edge_label,
                              nnn=False,
                              font_size=14,
                              font_color=edge_font_color)
        self.draw_edge_labels(self.pos,
                              self.nnn_edge_label,
                              nnn=True,
                              font_size=14,
                              font_color=edge_font_color)

    def draw_nodes(self, label, nnn, margins):
        nx.draw_networkx_nodes(self.graph,
                               pos=self.pos,
                               node_color='#99CCFF',
                               alpha=self.node_alpha,
                               node_size=self.node_size,
                               margins=margins)
        nx.draw_networkx_labels(self.graph,
                                pos=self.pos,
                                font_color='#000066',
                                font_size=12,
                                labels=self.node_label)
        if label == 'param':
            self.draw_node_overhead_labels(
                nnn, font_size=14, font_color='#FF8000')

    def draw_node_overhead_labels(self,
                                  nnn,
                                  font_size=14,
                                  font_color="k",
                                  font_family="sans-serif",
                                  font_weight="normal",
                                  alpha=None,
                                  ax: plt.Axes = None):
        if ax is None:
            ax = plt.gca()
        if self.lattice.dim == 1:
            shift = (0, 0.02) if nnn else (0, 0.05)
        elif self.lattice.dim == 2:
            shift = (-0.2, 0.2)
        self.overhead_pos = dict(
            (n, (self.pos[n][0] + shift[0], self.pos[n][1] + shift[1]))
            for n in self.graph.nodes())
        self.overhead_label = dict(
            (n, f'{self.U[n]*1e3:.0f}') for n in self.graph.nodes)

        nx.draw_networkx_labels(self.graph,
                                pos=self.overhead_pos,
                                font_color=font_color,
                                font_size=font_size,
                                font_family=font_family,
                                font_weight=font_weight,
                                alpha=alpha,
                                labels=self.overhead_label)

    def draw_edge_labels(self,
                         pos,
                         edge_labels: dict,
                         nnn=False,
                         font_size=10,
                         font_color="k",
                         font_family="sans-serif",
                         font_weight="normal",
                         alpha=None,
                         bbox=None,
                         horizontalalignment="center",
                         verticalalignment="center",
                         ax: plt.Axes = None,
                         rotate=True,
                         clip_on=True,
                         ):
        if ax is None:
            ax = plt.gca()

        labels = edge_labels
        text_items = {}
        for (n1, n2), label in labels.items():
            (x1, y1) = pos[n1]
            (x2, y2) = pos[n2]
            (x, y) = ((x1 + x2) / 2, (y1 + y2) / 2)
            if nnn:
                rad = 0.013 if self.lattice.dim == 1 else 0.1
                (dx, dy) = (x2 - x1, y2 - y1)
                (x, y) = (x + rad * dy, y - rad * dx)

            if rotate:
                # in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(
                    np.array((angle,)), xy.reshape((1, 2))
                )[0]
            else:
                trans_angle = 0.0
            # use default box of white with white border
            if bbox is None:
                bbox = dict(boxstyle="round", ec=(
                    1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
            if not isinstance(label, str):
                label = str(label)  # this makes "1" and 1 labeled the same

            t = ax.text(
                x,
                y,
                label,
                size=font_size,
                color=font_color,
                family=font_family,
                weight=font_weight,
                alpha=alpha,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                rotation=trans_angle,
                transform=ax.transData,
                bbox=bbox,
                zorder=1,
                clip_on=clip_on,
            )
            text_items[(n1, n2)] = t

        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
