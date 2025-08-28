import random
from collections import defaultdict

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import NetworkXStorage, TraverseStrategy
from graphgen.utils import logger


async def _get_node_info(
    node_id: str,
    graph_storage: NetworkXStorage,
) -> dict:
    """
    Get node info

    :param node_id: node id
    :param graph_storage: graph storage instance
    :return: node info
    """
    node_data = await graph_storage.get_node(node_id)
    return {"node_id": node_id, **node_data}


def _get_level_n_edges_by_max_width(
    edge_adj_list: dict,
    node_dict: dict,
    edges: list,
    nodes,
    src_edge: tuple,
    max_depth: int,
    bidirectional: bool,
    max_extra_edges: int,
    edge_sampling: str,
    loss_strategy: str = "only_edge",
) -> list:
    """
    Get level n edges for an edge.
    n is decided by max_depth in traverse_strategy

    :param edge_adj_list
    :param node_dict
    :param edges
    :param nodes
    :param src_edge
    :param max_depth
    :param bidirectional
    :param max_extra_edges
    :param edge_sampling
    :return: level n edges
    """
    src_id, tgt_id, _ = src_edge

    level_n_edges = []

    start_nodes = {tgt_id} if not bidirectional else {src_id, tgt_id}

    while max_depth > 0 and max_extra_edges > 0:
        max_depth -= 1

        candidate_edges = [
            edges[edge_id]
            for node in start_nodes
            for edge_id in edge_adj_list[node]
            if not edges[edge_id][2].get("visited", False)
        ]

        if not candidate_edges:
            break

        if len(candidate_edges) >= max_extra_edges:
            if loss_strategy == "both":
                er_tuples = [
                    ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
                    for edge in candidate_edges
                ]
                candidate_edges = _sort_tuples(er_tuples, edge_sampling)[
                    :max_extra_edges
                ]
            elif loss_strategy == "only_edge":
                candidate_edges = _sort_edges(candidate_edges, edge_sampling)[
                    :max_extra_edges
                ]
            else:
                raise ValueError(f"Invalid loss strategy: {loss_strategy}")

            for edge in candidate_edges:
                level_n_edges.append(edge)
                edge[2]["visited"] = True
            break

        max_extra_edges -= len(candidate_edges)
        new_start_nodes = set()

        for edge in candidate_edges:
            level_n_edges.append(edge)
            edge[2]["visited"] = True

            if not edge[0] in start_nodes:
                new_start_nodes.add(edge[0])
            if not edge[1] in start_nodes:
                new_start_nodes.add(edge[1])

        start_nodes = new_start_nodes

    return level_n_edges


def _get_level_n_edges_by_max_tokens(
    edge_adj_list: dict,
    node_dict: dict,
    edges: list,
    nodes: list,
    src_edge: tuple,
    max_depth: int,
    bidirectional: bool,
    max_tokens: int,
    edge_sampling: str,
    loss_strategy: str = "only_edge",
) -> list:
    """
    Get level n edges for an edge.
    n is decided by max_depth in traverse_strategy.

    :param edge_adj_list
    :param node_dict
    :param edges
    :param nodes
    :param src_edge
    :param max_depth
    :param bidirectional
    :param max_tokens
    :param edge_sampling
    :return: level n edges
    """
    src_id, tgt_id, src_edge_data = src_edge

    max_tokens -= (
        src_edge_data["length"]
        + nodes[node_dict[src_id]][1]["length"]
        + nodes[node_dict[tgt_id]][1]["length"]
    )

    level_n_edges = []

    start_nodes = {tgt_id} if not bidirectional else {src_id, tgt_id}
    temp_nodes = {src_id, tgt_id}

    while max_depth > 0 and max_tokens > 0:
        max_depth -= 1

        candidate_edges = [
            edges[edge_id]
            for node in start_nodes
            for edge_id in edge_adj_list[node]
            if not edges[edge_id][2].get("visited", False)
        ]

        if not candidate_edges:
            break

        if loss_strategy == "both":
            er_tuples = [
                ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
                for edge in candidate_edges
            ]
            candidate_edges = _sort_tuples(er_tuples, edge_sampling)
        elif loss_strategy == "only_edge":
            candidate_edges = _sort_edges(candidate_edges, edge_sampling)
        else:
            raise ValueError(f"Invalid loss strategy: {loss_strategy}")

        for edge in candidate_edges:
            max_tokens -= edge[2]["length"]
            if not edge[0] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[0]]][1]["length"]
            if not edge[1] in temp_nodes:
                max_tokens -= nodes[node_dict[edge[1]]][1]["length"]

            if max_tokens < 0:
                return level_n_edges

            level_n_edges.append(edge)
            edge[2]["visited"] = True
            temp_nodes.add(edge[0])
            temp_nodes.add(edge[1])

        new_start_nodes = set()
        for edge in candidate_edges:
            if not edge[0] in start_nodes:
                new_start_nodes.add(edge[0])
            if not edge[1] in start_nodes:
                new_start_nodes.add(edge[1])

        start_nodes = new_start_nodes

    return level_n_edges


def _sort_tuples(er_tuples: list, edge_sampling: str) -> list:
    """
    Sort edges with edge sampling strategy

    :param er_tuples: [(nodes:list, edge:tuple)]
    :param edge_sampling: edge sampling strategy (random, min_loss, max_loss)
    :return: sorted edges
    """
    if edge_sampling == "random":
        er_tuples = random.sample(er_tuples, len(er_tuples))
    elif edge_sampling == "min_loss":
        er_tuples = sorted(
            er_tuples,
            key=lambda x: sum(node[1]["loss"] for node in x[0]) + x[1][2]["loss"],
        )
    elif edge_sampling == "max_loss":
        er_tuples = sorted(
            er_tuples,
            key=lambda x: sum(node[1]["loss"] for node in x[0]) + x[1][2]["loss"],
            reverse=True,
        )
    else:
        raise ValueError(f"Invalid edge sampling: {edge_sampling}")
    edges = [edge for _, edge in er_tuples]
    return edges


def _sort_edges(edges: list, edge_sampling: str) -> list:
    """
    Sort edges with edge sampling strategy

    :param edges: total edges
    :param edge_sampling: edge sampling strategy (random, min_loss, max_loss)
    :return: sorted edges
    """
    if edge_sampling == "random":
        random.shuffle(edges)
    elif edge_sampling == "min_loss":
        edges = sorted(edges, key=lambda x: x[2]["loss"])
    elif edge_sampling == "max_loss":
        edges = sorted(edges, key=lambda x: x[2]["loss"], reverse=True)
    else:
        raise ValueError(f"Invalid edge sampling: {edge_sampling}")
    return edges


async def get_batches_with_strategy(  # pylint: disable=too-many-branches
    nodes: list,
    edges: list,
    graph_storage: NetworkXStorage,
    traverse_strategy: TraverseStrategy,
):
    expand_method = traverse_strategy.expand_method
    if expand_method == "max_width":
        logger.info("Using max width strategy")
    elif expand_method == "max_tokens":
        logger.info("Using max tokens strategy")
    else:
        raise ValueError(f"Invalid expand method: {expand_method}")

    max_depth = traverse_strategy.max_depth
    edge_sampling = traverse_strategy.edge_sampling

    # 构建临接矩阵
    edge_adj_list = defaultdict(list)
    node_dict = {}
    processing_batches = []

    node_cache = {}

    async def get_cached_node_info(node_id: str) -> dict:
        if node_id not in node_cache:
            node_cache[node_id] = await _get_node_info(node_id, graph_storage)
        return node_cache[node_id]

    for i, (node_name, _) in enumerate(nodes):
        node_dict[node_name] = i

    if traverse_strategy.loss_strategy == "both":
        er_tuples = [
            ([nodes[node_dict[edge[0]]], nodes[node_dict[edge[1]]]], edge)
            for edge in edges
        ]
        edges = _sort_tuples(er_tuples, edge_sampling)
    elif traverse_strategy.loss_strategy == "only_edge":
        edges = _sort_edges(edges, edge_sampling)
    else:
        raise ValueError(f"Invalid loss strategy: {traverse_strategy.loss_strategy}")

    for i, (src, tgt, _) in enumerate(edges):
        edge_adj_list[src].append(i)
        edge_adj_list[tgt].append(i)

    for edge in tqdm_async(edges, desc="Preparing batches"):
        if "visited" in edge[2] and edge[2]["visited"]:
            continue

        edge[2]["visited"] = True

        _process_nodes = []
        _process_edges = []

        src_id = edge[0]
        tgt_id = edge[1]

        _process_nodes.extend(
            [await get_cached_node_info(src_id), await get_cached_node_info(tgt_id)]
        )
        _process_edges.append(edge)

        if expand_method == "max_width":
            level_n_edges = _get_level_n_edges_by_max_width(
                edge_adj_list,
                node_dict,
                edges,
                nodes,
                edge,
                max_depth,
                traverse_strategy.bidirectional,
                traverse_strategy.max_extra_edges,
                edge_sampling,
                traverse_strategy.loss_strategy,
            )
        else:
            level_n_edges = _get_level_n_edges_by_max_tokens(
                edge_adj_list,
                node_dict,
                edges,
                nodes,
                edge,
                max_depth,
                traverse_strategy.bidirectional,
                traverse_strategy.max_tokens,
                edge_sampling,
                traverse_strategy.loss_strategy,
            )

        for _edge in level_n_edges:
            _process_nodes.append(await get_cached_node_info(_edge[0]))
            _process_nodes.append(await get_cached_node_info(_edge[1]))
            _process_edges.append(_edge)

        # 去重
        _process_nodes = list(
            {node["node_id"]: node for node in _process_nodes}.values()
        )
        _process_edges = list(
            {(edge[0], edge[1]): edge for edge in _process_edges}.values()
        )

        processing_batches.append((_process_nodes, _process_edges))

    logger.info("Processing batches: %d", len(processing_batches))

    # isolate nodes
    isolated_node_strategy = traverse_strategy.isolated_node_strategy
    if isolated_node_strategy == "add":
        processing_batches = await _add_isolated_nodes(
            nodes, processing_batches, graph_storage
        )
        logger.info(
            "Processing batches after adding isolated nodes: %d",
            len(processing_batches),
        )

    return processing_batches


async def _add_isolated_nodes(
    nodes: list,
    processing_batches: list,
    graph_storage: NetworkXStorage,
) -> list:
    visited_nodes = set()
    for _process_nodes, _process_edges in processing_batches:
        for node in _process_nodes:
            visited_nodes.add(node["node_id"])
    for node in nodes:
        if node[0] not in visited_nodes:
            _process_nodes = [await _get_node_info(node[0], graph_storage)]
            processing_batches.append((_process_nodes, []))
    return processing_batches
