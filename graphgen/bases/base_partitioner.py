from abc import ABC, abstractmethod
from typing import Any, List

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Community


class BasePartitioner(ABC):
    @abstractmethod
    async def partition(
        self,
        g: BaseGraphStorage,
        **kwargs: Any,
    ) -> List[Community]:
        """
        Graph -> Communities
        :param g: Graph storage instance
        :param kwargs: Additional parameters for partitioning
        :return: List of communities
        """

    @staticmethod
    async def community2batch(
        communities: List[Community], g: BaseGraphStorage
    ) -> list[
        tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ]
    ]:
        """
        Convert communities to batches of nodes and edges.
        :param communities
        :param g: Graph storage instance
        :return: List of batches, each batch is a tuple of (nodes, edges)
        """
        batches = []
        for comm in communities:
            nodes = comm.nodes
            edges = comm.edges
            nodes_data = []
            for node in nodes:
                node_data = await g.get_node(node)
                if node_data:
                    nodes_data.append((node, node_data))
            edges_data = []
            for u, v in edges:
                edge_data = await g.get_edge(u, v)
                if edge_data:
                    edges_data.append((u, v, edge_data))
                else:
                    edge_data = await g.get_edge(v, u)
                    if edge_data:
                        edges_data.append((v, u, edge_data))
            batches.append((nodes_data, edges_data))
        return batches

    @staticmethod
    def _build_adjacency_list(
        nodes: List[tuple[str, dict]], edges: List[tuple[str, str, dict]]
    ) -> tuple[dict[str, List[str]], set[tuple[str, str]]]:
        """
        Build adjacency list and edge set from nodes and edges.
        :param nodes
        :param edges
        :return: adjacency list, edge set
        """
        adj: dict[str, List[str]] = {n[0]: [] for n in nodes}
        edge_set: set[tuple[str, str]] = set()
        for e in edges:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
        return adj, edge_set
