import random
from collections import deque
from typing import Any, List, Literal, Set, Tuple

from graphgen.bases import BaseGraphStorage
from graphgen.bases.datatypes import Community

from .bfs_partitioner import BFSPartitioner

NODE_UNIT: str = "n"
EDGE_UNIT: str = "e"


class AnchorBFSPartitioner(BFSPartitioner):
    """
    Anchor BFS partitioner that partitions the graph into communities of a fixed size.
    1. Randomly choose a node of a specified type as the anchor.
    2. Expand the community using BFS until the max unit size is reached.(A unit is a node or an edge.)
    3. Non-anchor units can only be "pulled" into a community and never become seeds themselves.
    For example, for VQA tasks, we may want to use image nodes as anchors and expand to nearby text nodes and edges.
    """

    def __init__(
        self,
        *,
        anchor_type: Literal["image"] = "image",
        anchor_ids: Set[str] | None = None,
    ) -> None:
        super().__init__()
        self.anchor_type = anchor_type
        self.anchor_ids = anchor_ids

    async def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 1,
        **kwargs: Any,
    ) -> List[Community]:
        nodes = g.get_all_nodes()  # List[tuple[id, meta]]
        edges = g.get_all_edges()  # List[tuple[u, v, meta]]

        adj, _ = self._build_adjacency_list(nodes, edges)

        anchors: Set[str] = await self._pick_anchor_ids(nodes)
        if not anchors:
            return []  # if no anchors, return empty list

        used_n: set[str] = set()
        used_e: set[frozenset[str]] = set()
        communities: List[Community] = []

        seeds = list(anchors)
        random.shuffle(seeds)

        for seed_node in seeds:
            if seed_node in used_n:
                continue
            comm_n, comm_e = await self._grow_community(
                seed_node, adj, max_units_per_community, used_n, used_e
            )
            if comm_n or comm_e:
                communities.append(
                    Community(id=len(communities), nodes=comm_n, edges=comm_e)
                )

        return communities

    async def _pick_anchor_ids(
        self,
        nodes: List[tuple[str, dict]],
    ) -> Set[str]:
        if self.anchor_ids is not None:
            return self.anchor_ids

        anchor_ids: Set[str] = set()
        for node_id, meta in nodes:
            node_type = str(meta.get("entity_type", "")).lower()
            if self.anchor_type.lower() in node_type:
                anchor_ids.add(node_id)
        return anchor_ids

    @staticmethod
    async def _grow_community(
        seed: str,
        adj: dict[str, List[str]],
        max_units: int,
        used_n: set[str],
        used_e: set[frozenset[str]],
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Grow a community from the seed node using BFS.
        :param seed: seed node id
        :param adj: adjacency list
        :param max_units: maximum number of units (nodes + edges) in the community
        :param used_n: set of used node ids
        :param used_e: set of used edge keys
        :return: (list of node ids, list of edge tuples)
        """
        comm_n: List[str] = []
        comm_e: List[Tuple[str, str]] = []
        queue: deque[tuple[str, Any]] = deque([(NODE_UNIT, seed)])
        cnt = 0

        while queue and cnt < max_units:
            k, it = queue.popleft()

            if k == NODE_UNIT:
                if it in used_n:
                    continue
                used_n.add(it)
                comm_n.append(it)
                cnt += 1
                for nei in adj[it]:
                    e_key = frozenset((it, nei))
                    if e_key not in used_e:
                        queue.append((EDGE_UNIT, e_key))
            else:  # EDGE_UNIT
                if it in used_e:
                    continue
                used_e.add(it)
                u, v = it
                comm_e.append((u, v))
                cnt += 1
                for n in it:
                    if n not in used_n:
                        queue.append((NODE_UNIT, n))

        return comm_n, comm_e
