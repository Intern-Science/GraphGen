from typing import Any

from graphgen.bases import BaseGraphStorage, BaseKVStorage, BaseTokenizer
from graphgen.models import (
    AnchorBFSPartitioner,
    BFSPartitioner,
    DFSPartitioner,
    ECEPartitioner,
    LeidenPartitioner,
)
from graphgen.utils import logger

from .pre_tokenize import pre_tokenize


async def partition_kg(
    kg_instance: BaseGraphStorage,
    chunk_storage: BaseKVStorage,
    tokenizer: Any = BaseTokenizer,
    partition_config: dict = None,
) -> list[
    tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]]
]:
    method = partition_config["method"]
    method_params = partition_config["method_params"]
    if method == "bfs":
        logger.info("Partitioning knowledge graph using BFS method.")
        partitioner = BFSPartitioner()
    elif method == "dfs":
        logger.info("Partitioning knowledge graph using DFS method.")
        partitioner = DFSPartitioner()
    elif method == "ece":
        logger.info("Partitioning knowledge graph using ECE method.")
        # TODO： before ECE partitioning, we need to:
        # 1. 'quiz and judge' to get the comprehension loss if unit_sampling is not random
        # 2. pre-tokenize nodes and edges to get the token length
        edges = await kg_instance.get_all_edges()
        nodes = await kg_instance.get_all_nodes()
        await pre_tokenize(kg_instance, tokenizer, edges, nodes)
        partitioner = ECEPartitioner()
    elif method == "leiden":
        logger.info("Partitioning knowledge graph using Leiden method.")
        partitioner = LeidenPartitioner()
    elif method == "anchor_bfs":
        logger.info("Partitioning knowledge graph using Anchor BFS method.")
        partitioner = AnchorBFSPartitioner(
            anchor_type=method_params.get("anchor_type"),
            anchor_ids=set(method_params.get("anchor_ids", []))
            if method_params.get("anchor_ids")
            else None,
        )
    else:
        raise ValueError(f"Unsupported partition method: {method}")

    communities = await partitioner.partition(g=kg_instance, **method_params)
    logger.info("Partitioned the graph into %d communities.", len(communities))
    batches = await partitioner.community2batch(communities, g=kg_instance)

    batches = await attach_additional_data_to_node(batches, chunk_storage)
    return batches


async def attach_additional_data_to_node(
    batches: list[
        tuple[
            list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]
        ]
    ],
    chunk_storage: BaseKVStorage,
) -> list[
    tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict] | tuple[Any, Any, Any]]]
]:
    """入口函数：为所有节点追加额外数据。"""
    for batch in batches:
        for node_id, node_data in batch[0]:
            await _attach_by_type(node_id, node_data, chunk_storage)
    return batches


async def _attach_by_type(
    node_id: str,
    node_data: dict,
    chunk_storage: BaseKVStorage,
) -> None:
    """根据 entity_type 给单个节点追加额外数据。"""
    entity_type = (node_data.get("entity_type") or "").lower()
    if not entity_type:
        return

    # 统一取 source_id
    source_ids = [
        sid.strip()
        for sid in node_data.get("source_id", "").split("<SEP>")
        if sid.strip()
    ]

    # 处理图片
    if "image" in entity_type:
        captions = [
            data["image_caption"]
            for sid in source_ids
            if "image" in sid.lower() and (data := await chunk_storage.get_by_id(sid))
        ]
        if captions:
            node_data["images"] = captions
            logger.debug("Attached image data to node %s", node_id)

    # 处理蛋白
    if "protein" in entity_type:
        captions = [
            data["protein_caption"]
            for sid in source_ids
            if "protein" in sid.lower() and (data := await chunk_storage.get_by_id(sid))
        ]
        if captions:
            node_data["protein"] = captions
            logger.debug("Attached protein data to node %s", node_id)
