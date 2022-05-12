import matplotlib.pyplot as plt
from wannier import *
import networkx as nx


class Graph(Wannier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def gen_graph(self):
        G = nx.grid_graph(tuple(self.lattice))
