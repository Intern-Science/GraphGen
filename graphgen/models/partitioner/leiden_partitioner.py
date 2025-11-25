from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import igraph as ig
from leidenalg import ModularityVertexPartition, find_partition

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community


class LeidenPartitioner(BasePartitioner):
    """
    Leiden partitioner that partitions the graph into communities using the Leiden algorithm.
    """

    async def partition(
        self,
        g: BaseGraphStorage,
        max_size: int = 20,
        use_lcc: bool = False,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> List[Community]:
        """
        Leiden Partition follows these steps:
        1. export the graph from graph storage
        2. use the leiden algorithm to detect communities, get {node: community_id}
        3. split large communities if max_size is given
        4. convert {node: community_id} to List[Community]
        :param g
        :param max_size: maximum size of each community, if None or <=0, no limit
        :param use_lcc: whether to use the largest connected component only
        :param random_seed
        :param kwargs: other parameters for the leiden algorithm
        :return:
        """
        nodes = g.get_all_nodes()  # List[Tuple[str, dict]]
        edges = g.get_all_edges()  # List[Tuple[str, str, dict]]

        node2cid: Dict[str, int] = await self._run_leiden(
            nodes, edges, use_lcc, random_seed
        )

        if max_size is not None and max_size > 0:
            node2cid = await self._split_communities(node2cid, max_size)

        cid2nodes: Dict[int, List[str]] = defaultdict(list)
        for n, cid in node2cid.items():
            cid2nodes[cid].append(n)

        communities: List[Community] = []
        for cid, nodes in cid2nodes.items():
            node_set: Set[str] = set(nodes)
            comm_edges: List[Tuple[str, str]] = [
                (u, v) for u, v, _ in edges if u in node_set and v in node_set
            ]
            communities.append(Community(id=cid, nodes=nodes, edges=comm_edges))
        return communities

    @staticmethod
    async def _run_leiden(
        nodes: List[Tuple[str, dict]],
        edges: List[Tuple[str, str, dict]],
        use_lcc: bool = False,
        random_seed: int = 42,
    ) -> Dict[str, int]:
        # build igraph
        ig_graph = ig.Graph.TupleList(((u, v) for u, v, _ in edges), directed=False)

        # remove isolated nodes
        ig_graph.delete_vertices(ig_graph.vs.select(_degree_eq=0))

        node2cid: Dict[str, int] = {}
        if use_lcc:
            lcc = ig_graph.components().giant()
            partition = find_partition(lcc, ModularityVertexPartition, seed=random_seed)
            for part_id, cluster in enumerate(partition):
                for v in cluster:
                    node2cid[lcc.vs[v]["name"]] = part_id
        else:
            offset = 0
            for component in ig_graph.components():
                subgraph = ig_graph.induced_subgraph(component)
                partition = find_partition(
                    subgraph, ModularityVertexPartition, seed=random_seed
                )
                for part_id, cluster in enumerate(partition):
                    for v in cluster:
                        original_node = subgraph.vs[v]["name"]
                        node2cid[original_node] = part_id + offset
                offset += len(partition)
        return node2cid

    @staticmethod
    async def _split_communities(
        node2cid: Dict[str, int], max_size: int
    ) -> Dict[str, int]:
        """
        Split communities larger than max_size into smaller sub-communities.
        """
        cid2nodes: Dict[int, List[str]] = defaultdict(list)
        for n, cid in node2cid.items():
            cid2nodes[cid].append(n)

        new_mapping: Dict[str, int] = {}
        new_cid = 0
        for nodes in cid2nodes.values():
            if len(nodes) <= max_size:
                for n in nodes:
                    new_mapping[n] = new_cid
                new_cid += 1
            else:
                for start in range(0, len(nodes), max_size):
                    chunk = nodes[start : start + max_size]
                    for n in chunk:
                        new_mapping[n] = new_cid
                    new_cid += 1
        return new_mapping
