from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class Visualizer:
    """
    Class for visualizing graphs using NetworkX and Matplotlib.
    """

    graph: nx.Graph = None
    communities: Dict[str, int] = None
    layout: str = "spring"
    max_nodes: int = 1000
    node_size: int = 10
    alpha: float = 0.6

    def visualize(self, save_path: str = None):
        n = self.graph.number_of_nodes()
        if self.layout == "spring":
            k = max(0.1, 1.0 / (n**0.5))
            pos = nx.spring_layout(self.graph, k=k, seed=42)
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

        plt.figure(figsize=(10, 10))

        node_colors = [self.communities.get(node, 0) for node in self.graph.nodes()]

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=self.node_size,
            node_color=node_colors,
            cmap=plt.cm.tab20,
            alpha=self.alpha,
        )
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=0.2)
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Saved to", save_path)
        else:
            plt.show()
