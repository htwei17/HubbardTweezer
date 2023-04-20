import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .equalizer import *

LINE_WIDTH = 6
# FONT_FAMILY = 'serif'
FONT_FAMILY = "cursive"
# FONT_FAMILY = 'fantasy'
# BOND_COLOR = '#4AC26D'
# BOND_TEXT_COLOR = np.array([0.122972391,	0.63525259,	0.529459411])
BOND_TEXT_COLOR = "darkcyan"
BOND_TEXT_SIZE = 32
NODE_SIZE = 2400
MIN_GAP = 18
# NODE_COLOR = '#BFDF25'
NODE_EDGE_WIDTH = LINE_WIDTH
# NODE_TEXT_COLOR = np.array([0.282250485,	0.146422331, 0.461908376])
NODE_TEXT_SIZE = 28
# OVERHEAD_COLOR = np.array([0.62352941, 0.85490196, 0.22745098])
OVERHEAD_COLOR = "firebrick"
OVERHEAD_SUZE = 30
FONT_WEIGHT = 200
WAIST_SCALE = 3
SCALEBAR_TEXT_SIZE = 32

color_scheme1 = {
    "bond": "teal",
    "bond_text": "darkcyan",
    "node": "paleturquoise",
    "node_text": "darkslateblue",
    "overhead": "firebrick",
}

color_scheme2 = {
    "bond": "goldenrod",
    "bond_text": "olive",
    "node": "wheat",
    "node_text": "darkolivegreen",
    "overhead": "saddlebrown",
}

color_scheme3 = {
    "bond": "olivedrab",
    "bond_text": "darkcyan",
    "node": "greenyellow",
    "node_text": "darkslateblue",
    "overhead": "darkred",
}

params = {
    # 'figure.dpi': 300,
    # 'figure.figsize': (15, 5),
    # 'legend.fontsize': 'x-large',
    # 'axes.labelsize': 'xx-large',
    # 'axes.titlesize': 'xx-large',
    # 'xtick.labelsize': 'xx-large',
    # 'ytick.labelsize': 'xx-large'
    "mathtext.fontset": "cm",
    "font.family": FONT_FAMILY,
}
plt.rcParams.update(params)


class HubbardGraph(HubbardEqualizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # # Resize [n] to [n, 1]
        # self.lattice = np.resize(
        #     np.pad(self.lattice, pad_width=(0, 1), constant_values=1), 2)
        self.edges = [tuple(row) for row in self.lattice.links]
        self.graph = nx.DiGraph(self.edges, name="Lattice")
        self.pos = dict(
            # (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
            (n, self.wf_centers[n])
            for n in self.graph.nodes()
        )

    def set_edges(self, label="param"):
        for link in self.graph.edges:
            if label == "param":
                # Label bond tunneling
                length = abs(self.A[link[0], link[1]]) * 1e3  # Convert to kHz
            elif label == "adjust":
                # Label bond length
                length = la.norm(np.diff(self.trap_centers[link, :], axis=0))
            else:
                length = 1.0
            self.graph[link[0]][link[1]]["weight"] = length
        self.edge_label = dict(
            (edge, f'{self.graph[edge[0]][edge[1]]["weight"]:.0f}')
            for edge in self.graph.edges
        )
        self.edge_alpha = np.array(
            [self.graph[edge[0]][edge[1]]["weight"] for edge in self.graph.edges]
        )
        is_masked_links = np.array(
            [
                np.logical_or(self.ghost.mask[edge[0]], self.ghost.mask[edge[1]])
                for edge in self.graph.edges
            ]
        )
        max_len = max(self.edge_alpha[is_masked_links])
        self.edge_alpha /= max_len
        self.edge_alpha = np.clip(self.edge_alpha, 0.0, 1)

    def set_nodes(self, label="param"):
        if label == "param":
            # Label onsite chemical potential
            self.pos = dict((n, self.wf_centers[n]) for n in self.graph.nodes())
            depth = np.real(np.diag(self.A)) * 1e3  # Convert to kHz
            self.node_label = dict((n, f"{int(depth[n])}") for n in self.graph.nodes)
        elif label == "adjust":
            # Label trap offset
            self.pos = dict(
                # (n, np.sign(self.trap_centers[n]) * abs(self.trap_centers[n])**1.1)
                (n, self.trap_centers[n])
                for n in self.graph.nodes()
            )
            self.node_label = dict((n, f"{self.Voff[n]:.3g}") for n in self.graph.nodes)
        else:
            self.pos = dict((n, self.wf_centers[n]) for n in self.graph.nodes())
            self.node_label = dict((n, "") for n in self.graph.nodes)
        self.node_size = self.waists[:, 0] ** WAIST_SCALE * NODE_SIZE
        max_depth = np.max(abs(self.Voff[self.ghost.mask]))
        self.node_alpha = (self.Voff / max_depth) ** 10
        self.node_alpha = np.clip(self.node_alpha, 0.0, 1)

    def add_nnn(self, center=0, limit=3):
        # Add higher neighbor bonds
        # NOTE: explicit square lattice geometry assumed
        # FIXME: 3x2 lattice error as this gives an index 6
        if self.lattice.shape == "zigzag":
            for i in range(min(limit, self.lattice.N // 2)):
                self.graph.add_edge(i, i + 1)
        if self.lattice.shape in ["square", "Lieb", "triangular", "zigzag"]:
            if limit + 2 > self.lattice.N:
                limit = self.lattice.N - 2
            if center >= self.lattice.N:
                center = 0
            if self.lattice.dim == 1:
                for i in range(limit):
                    self.graph.add_edge(center, i + 2)
            elif self.lattice.dim == 2:
                for i in range(2 * limit):
                    self.graph.add_edge(center, i + 2)
        else:
            print(
                f"WARNING: nnn not supported for {self.lattice.shape} lattice. \
                    Nothing doen."
            )
            return

    def singleband_params(self, label="param", band=1, A=None, U=None):
        if label == "param" and (A is None or U is None):
            self.singleband_Hubbard(u=True, band=band)
        elif label == "adjust" and A is None:
            self.singleband_Hubbard(u=False, band=band)
        elif label not in ["param", "adjust"]:
            raise ValueError("Invalid label.")

    def draw_graph(
        self, label="param", band=1, nnn=False, A=None, U=None, scalebar=True
    ):
        if isinstance(band, int):
            self.singleband_params(label, band, A, U)
            if band == 1:
                self.color = color_scheme1
            elif band == 2:
                self.color = color_scheme2
        elif isinstance(band, Iterable):
            label = "interband"
            if U.ndim == 1:
                self.U = U
            else:
                self.U = U[band[0] - 1, band[1] - 1]
            self.color = color_scheme3
        else:
            raise ValueError(f"Invalid band argument {band}.")
        if label == "param" and nnn:
            self.add_nnn()
        if all(abs(self.wf_centers[:, 1]) < 1e-6):
            self.lattice.dim = 1
            self.lattice.size = np.array([self.lattice.N, 1])
            self.wf_centers[:, 1] = 0

        self.set_edges(label)
        self.set_nodes(label)

        if self.verbosity:
            print("\nStart to plot graph...")

        if self.lattice.dim == 1:
            fs = [3.5 * (self.lattice.size[0] - 1), 3.5]
            margins = (5e-2, 0.8) if nnn else (5e-2, 0.8)
        elif self.lattice.dim == 2:
            fs = [3.5 * (self.lattice.size[0] - 1), 3.5 * (self.lattice.size[1] - 1)]
            if self.lattice.shape == "ring":
                fs[1] = fs[0]
            margins = (
                0.2 / np.sqrt(self.lattice.size[0] - 2),
                0.25 / np.sqrt(self.lattice.size[1] - 2),
            )
        plt.figure(figsize=fs)

        self.draw_nodes(label, nnn, margins)
        self.draw_edges(label)
        if scalebar:
            self.add_scalebar(color=self.color["bond"])

        plt.axis("off")
        plt.savefig(
            f"{self.lattice.size} nx {self.dim}d {self.lattice.shape} {label} {self.waist_dir} {self.eq_label} band{band}.pdf",
            transparent=True,
            bbox_inches="tight",
        )

    def draw_edges(self, label="param"):
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
        for i in range(len(link_list)):
            el = link_list[i]
            cs = "arc3"  # rad=0, meaning straight line
            # For all further neighbor edges, use curved lines
            if not isnn[i]:
                cs = "arc3,rad=0.3"  # rad=0.2, meaning C1 to C0-C2 is 0.2 * C0-C2 distance
            nx.draw_networkx_edges(
                self.graph,
                self.pos,
                arrows=True,
                arrowstyle="-",
                style="dashed" if label == "interband" else "solid",
                edgelist=[el],
                edge_color=self.color["bond"],
                connectionstyle=cs,
                alpha=np.sqrt(self.edge_alpha[i]),
                width=LINE_WIDTH,
                min_source_margin=MIN_GAP + LINE_WIDTH,
                min_target_margin=MIN_GAP + LINE_WIDTH,
            )
        if label in ["param", "adjust"]:
            self.draw_edge_labels(
                self.pos,
                self.nn_edge_label,
                nnn=False,
                font_size=NODE_TEXT_SIZE,
                font_color=self.color["bond_text"],
            )
            self.draw_edge_labels(
                self.pos,
                self.nnn_edge_label,
                nnn=True,
                font_size=NODE_TEXT_SIZE,
                font_color=self.color["bond_text"],
            )

    def draw_nodes(self, label, nnn, margins):
        fillstyle = "none" if label == "interband" else "full"
        node_list = list(self.graph.nodes)
        for i in range(len(node_list)):
            em = eliptic_marker((self.waists[i, 1] / self.waists[i, 0]) ** WAIST_SCALE)
            nx.draw_networkx_nodes(
                self.graph,
                pos=self.pos,
                nodelist=[node_list[i]],
                node_color=self.color["node"],
                node_shape=MarkerStyle(marker=em, fillstyle=fillstyle),
                linewidths=NODE_EDGE_WIDTH,
                alpha=self.node_alpha[i],
                node_size=self.node_size[i],
                margins=margins,
            )
        if label in ["param", "adjust"]:
            nx.draw_networkx_labels(
                self.graph,
                pos=self.pos,
                font_family=FONT_FAMILY,
                font_color=self.color["node_text"],
                font_size=NODE_TEXT_SIZE,
                font_weight=FONT_WEIGHT,
                labels=self.node_label,
            )
        if label in ["param", "interband"]:
            self.draw_node_overhead_labels(
                nnn, font_size=OVERHEAD_SUZE, font_color=self.color["overhead"]
            )

    def draw_node_overhead_labels(
        self,
        nnn,
        font_size=OVERHEAD_SUZE,
        font_color=OVERHEAD_COLOR,
        font_family=FONT_FAMILY,
        font_weight=FONT_WEIGHT,
        alpha=None,
        ax: plt.Axes = None,
    ):
        if ax is None:
            ax = plt.gca()
        # Shift in unit of w, since wy is not defined in 1D
        if self.lattice.dim == 1:
            shift = (0, 0.06) if nnn else (0, 0.05)
        elif self.lattice.dim == 2:
            shift = (-0.35, 0.35)
        self.overhead_pos = dict(
            (n, (self.pos[n][0] + shift[0], self.pos[n][1] + shift[1]))
            for n in self.graph.nodes()
        )
        self.overhead_label = dict(
            (n, f"{self.U[n]*1e3:.0f}") for n in self.graph.nodes
        )

        nx.draw_networkx_labels(
            self.graph,
            pos=self.overhead_pos,
            font_color=font_color,
            font_size=font_size,
            font_family=font_family,
            font_weight=font_weight,
            alpha=alpha,
            labels=self.overhead_label,
        )

    def draw_edge_labels(
        self,
        pos,
        edge_labels: dict,
        nnn=False,
        font_size=BOND_TEXT_SIZE,
        font_color=BOND_TEXT_COLOR,
        font_family=FONT_FAMILY,
        font_weight=FONT_WEIGHT,
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax: plt.Axes = None,
        rotate=False,
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
                rad = 0.025 if self.lattice.dim == 1 else 0.1
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
                bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
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

    def add_scalebar(
        self, ax: plt.Axes = None, color="teal", scale=1.0, unit="$\mu m$"
    ):
        if ax is None:
            ax = plt.gca()
        fontprops = fm.FontProperties(size=SCALEBAR_TEXT_SIZE)
        sb = AnchoredSizeBar(
            ax.transData,
            scale,
            f"{scale}" + unit,
            loc="lower center",
            pad=0.01,
            color=color,
            frameon=False,
            fontproperties=fontprops,
        )
        ax.add_artist(sb)


def eliptic_marker(epsilon):
    circle = Path.unit_circle()
    verts = np.copy(circle.vertices)
    verts[:, 0] *= epsilon
    em = Path(verts, circle.codes)
    return em
