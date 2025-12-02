import random
from collections import deque
from typing import Any, List

from graphgen.bases import BaseGraphStorage, BasePartitioner
from graphgen.bases.datatypes import Community

NODE_UNIT: str = "n"
EDGE_UNIT: str = "e"


class BFSPartitioner(BasePartitioner):
    """
    BFS partitioner that partitions the graph into communities of a fixed size.
    1. Randomly choose a unit.
    2. Expand the community using BFS until the max unit size is reached.
    (A unit is a node or an edge.)
    """

    async def partition(
        self,
        g: BaseGraphStorage,
        max_units_per_community: int = 1,
        **kwargs: Any,
    ) -> List[Community]:
        nodes = g.get_all_nodes()
        edges = g.get_all_edges()

        adj, _ = self._build_adjacency_list(nodes, edges)

        used_n: set[str] = set()
        used_e: set[frozenset[str]] = set()
        communities: List[Community] = []

        units = [(NODE_UNIT, n[0]) for n in nodes] + [
            (EDGE_UNIT, frozenset((u, v))) for u, v, _ in edges
        ]
        random.shuffle(units)

        for kind, seed in units:
            if (kind == NODE_UNIT and seed in used_n) or (
                kind == EDGE_UNIT and seed in used_e
            ):
                continue

            comm_n: List[str] = []
            comm_e: List[tuple[str, str]] = []
            queue: deque[tuple[str, Any]] = deque([(kind, seed)])
            cnt = 0

            while queue and cnt < max_units_per_community:
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
                else:
                    if it in used_e:
                        continue
                    used_e.add(it)

                    u, v = it
                    comm_e.append((u, v))
                    cnt += 1
                    # push nodes that are not visited
                    for n in it:
                        if n not in used_n:
                            queue.append((NODE_UNIT, n))

            if comm_n or comm_e:
                communities.append(
                    Community(id=len(communities), nodes=comm_n, edges=comm_e)
                )

        return communities
