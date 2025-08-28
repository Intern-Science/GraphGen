from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from graphgen.models.storage.networkx_storage import NetworkXStorage


@dataclass
class CommunityDetector:
    """Class for community detection algorithms."""

    graph_storage: NetworkXStorage = None
    method: str = "leiden"
    method_params: Dict[str, Any] = None

    async def detect_communities(self) -> Dict[str, int]:
        if self.method == "leiden":
            return await self._leiden_communities(**self.method_params or {})
        raise ValueError(f"Unknown community detection method: {self.method}")

    async def get_graph(self):
        return await self.graph_storage.get_graph()

    async def _leiden_communities(
        self, max_size: int = None, **kwargs
    ) -> Dict[str, int]:
        """
        Detect communities using the Leiden algorithm.
        If max_size is given, any community larger than max_size will be split
        into smaller sub-communities each having at most max_size nodes.
        """
        import igraph as ig
        import networkx as nx
        from leidenalg import ModularityVertexPartition, find_partition

        graph = await self.get_graph()
        graph.remove_nodes_from(list(nx.isolates(graph)))

        ig_graph = ig.Graph.TupleList(graph.edges(), directed=False)

        random_seed = kwargs.get("random_seed", 42)
        use_lcc = kwargs.get("use_lcc", False)

        communities: Dict[str, int] = {}
        if use_lcc:
            lcc = ig_graph.components().giant()
            partition = find_partition(lcc, ModularityVertexPartition, seed=random_seed)
            for part, cluster in enumerate(partition):
                for v in cluster:
                    communities[lcc.vs[v]["name"]] = part
        else:
            offset = 0
            for component in ig_graph.components():
                subgraph = ig_graph.induced_subgraph(component)
                partition = find_partition(
                    subgraph, ModularityVertexPartition, seed=random_seed
                )
                for part, cluster in enumerate(partition):
                    for v in cluster:
                        original_node = subgraph.vs[v]["name"]
                        communities[original_node] = part + offset
                offset += len(partition)

        # split large communities if max_size is specified
        if max_size is None or max_size <= 0:
            return communities

        return await self._split_communities(communities, max_size)

    @staticmethod
    async def _split_communities(
        communities: Dict[str, int], max_size: int
    ) -> Dict[str, int]:
        """
        Split communities larger than max_size into smaller sub-communities.
        """
        cid2nodes: Dict[int, List[str]] = defaultdict(list)
        for node, cid in communities.items():
            cid2nodes[cid].append(node)

        new_communities: Dict[str, int] = {}
        new_cid = 0
        for cid, nodes in cid2nodes.items():
            if len(nodes) <= max_size:
                for n in nodes:
                    new_communities[n] = new_cid
                new_cid += 1
            else:
                for start in range(0, len(nodes), max_size):
                    sub = nodes[start : start + max_size]
                    for n in sub:
                        new_communities[n] = new_cid
                    new_cid += 1

        return new_communities
