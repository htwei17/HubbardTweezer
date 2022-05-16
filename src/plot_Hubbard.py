import matplotlib.pyplot as plt
from wannier import *
import networkx as nx
import matplotlib as mpl

params = {
    'figure.dpi': 300,
    # 'figure.figsize': (15, 5),
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large'
}
mpl.rcParams.update(params)


class Graph(Wannier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # # Resize [n] to [n, 1]
        # self.lattice = np.resize(
        #     np.pad(self.lattice, pad_width=(0, 1), constant_values=1), 2)
        self.edges = [tuple(row) for row in self.links]
        self.graph = nx.Graph(self.edges, name='Lattice')
        self.pos = dict(
            (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
            for n in self.graph.nodes())

    def update_edge_weight(self):
        for link in self.links:
            # # Label bond length
            # length = la.norm(np.diff(self.trap_centers[link], axis=0))
            # Label bond tunneling
            length = abs(self.A[link[0], link[1]])
            self.graph[link[0]][link[1]]['weight'] = length
        self.edge_label = dict(
            (edge, f'{self.graph[edge[0]][edge[1]]["weight"]:.3g}')
            for edge in self.edges)

    def update_node_weight(self):
        # # Label trap offset
        # self.node_label = dict(
        #     (n, f'{self.Voff[n]:.3g}') for n in self.graph.nodes())
        # Label onsite chemical potential
        depth = np.real(np.diag(self.A))
        self.node_label = dict(
            (n, f'{depth[n]:.3g}') for n in self.graph.nodes())
        self.node_size = [i**10 * 600 for i in self.Voff]

    def draw_graph(self):
        nx.draw(self.graph,
                pos=self.pos,
                edge_color='#606060',
                with_labels=False,
                width=3,
                node_color='#99CCFF',
                node_size=self.node_size)
        nx.draw_networkx_labels(self.graph,
                                pos=self.pos,
                                font_color='#00994C',
                                font_size=8,
                                labels=self.node_label)
        nx.draw_networkx_edge_labels(self.graph,
                                     self.pos,
                                     edge_labels=self.edge_label,
                                     font_color=[0.256, 0.439, 0.588])
        plt.axis('off')
        plt.savefig('Graph.pdf')