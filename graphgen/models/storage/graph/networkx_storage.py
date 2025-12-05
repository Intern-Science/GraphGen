import html
import os
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import networkx as nx

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.utils import logger


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> Optional[nx.Graph]:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            "Writing graph with %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        通过对节点和边进行排序来实现
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    source, target = target, source
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
        Initialize the NetworkX graph storage by loading an existing graph from a GraphML file,
        if it exists, or creating a new empty graph otherwise.
        """
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                "Loaded graph from %s with %d nodes, %d edges",
                self._graphml_xml_file,
                preloaded_graph.number_of_nodes(),
                preloaded_graph.number_of_edges(),
            )
        self._graph = preloaded_graph or nx.Graph()

    def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    def get_all_nodes(self) -> Union[list[tuple[str, dict]], None]:
        return list(self._graph.nodes(data=True))

    def node_degree(self, node_id: str) -> int:
        return int(self._graph.degree[node_id])

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return int(self._graph.degree[src_id] + self._graph.degree[tgt_id])

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    def get_all_edges(self) -> Union[list[tuple[str, str, dict]], None]:
        return list(self._graph.edges(data=True))

    def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id, data=True))
        return None

    def get_graph(self) -> nx.Graph:
        return self._graph

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    def update_node(self, node_id: str, node_data: dict[str, str]):
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id].update(node_data)
        else:
            logger.warning("Node %s not found in the graph for update.", node_id)

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        if self._graph.has_edge(source_node_id, target_node_id):
            self._graph.edges[(source_node_id, target_node_id)].update(edge_data)
        else:
            logger.warning(
                "Edge %s -> %s not found in the graph for update.",
                source_node_id,
                target_node_id,
            )

    def delete_node(self, node_id: str):
        """
        Delete a node from the graph based on the specified node_id.

        :param node_id: The node_id to delete
        """
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)
            logger.info("Node %s deleted from the graph.", node_id)
        else:
            logger.warning("Node %s not found in the graph for deletion.", node_id)

    def clear(self):
        """
        Clear the graph by removing all nodes and edges.
        """
        self._graph.clear()
        logger.info("Graph %s cleared.", self.namespace)

    def reload(self):
        """
        Reload the graph from the GraphML file.
        """
        self.__post_init__()
