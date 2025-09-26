import asyncio
from collections import Counter

from tqdm.asyncio import tqdm as tqdm_async

from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.models import Tokenizer, TopkTokenModel
from graphgen.templates import KG_EXTRACTION_PROMPT, KG_SUMMARIZATION_PROMPT
from graphgen.utils import detect_main_language, logger
from graphgen.utils.format import split_string_by_multi_markers


async def _handle_kg_summary(
    entity_or_relation_name: str,
    description: str,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
    max_summary_tokens: int = 200,
) -> str:
    """
    处理实体或关系的描述信息

    :param entity_or_relation_name
    :param description
    :param llm_client
    :param tokenizer_instance
    :param max_summary_tokens
    :return: new description
    """
    language = detect_main_language(description)
    if language == "en":
        language = "English"
    else:
        language = "Chinese"
    KG_EXTRACTION_PROMPT["FORMAT"]["language"] = language

    tokens = tokenizer_instance.encode_string(description)
    if len(tokens) < max_summary_tokens:
        return description

    use_description = tokenizer_instance.decode_tokens(tokens[:max_summary_tokens])
    prompt = KG_SUMMARIZATION_PROMPT[language]["TEMPLATE"].format(
        entity_name=entity_or_relation_name,
        description_list=use_description.split("<SEP>"),
        **KG_SUMMARIZATION_PROMPT["FORMAT"],
    )
    new_description = await llm_client.generate_answer(prompt)
    logger.info(
        "Entity or relation %s summary: %s", entity_or_relation_name, new_description
    )
    return new_description


async def merge_nodes(
    nodes_data: dict,
    kg_instance: BaseGraphStorage,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
    max_concurrent: int = 1000,
):
    """
    Merge nodes

    :param nodes_data
    :param kg_instance
    :param llm_client
    :param tokenizer_instance
    :param max_concurrent
    :return
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_node(entity_name: str, node_data: list[dict]):
        async with semaphore:
            entity_types = []
            source_ids = []
            descriptions = []

            node = await kg_instance.get_node(entity_name)
            if node is not None:
                entity_types.append(node["entity_type"])
                source_ids.extend(
                    split_string_by_multi_markers(node["source_id"], ["<SEP>"])
                )
                descriptions.append(node["description"])

            # 统计当前节点数据和已有节点数据的entity_type出现次数，取出现次数最多的entity_type
            entity_type = sorted(
                Counter([dp["entity_type"] for dp in node_data] + entity_types).items(),
                key=lambda x: x[1],
                reverse=True,
            )[0][0]

            description = "<SEP>".join(
                sorted(set([dp["description"] for dp in node_data] + descriptions))
            )
            description = await _handle_kg_summary(
                entity_name, description, llm_client, tokenizer_instance
            )

            source_id = "<SEP>".join(
                set([dp["source_id"] for dp in node_data] + source_ids)
            )

            node_data = {
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id,
            }
            await kg_instance.upsert_node(entity_name, node_data=node_data)
            node_data["entity_name"] = entity_name
            return node_data

    logger.info("Inserting entities into storage...")
    entities_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [process_single_node(k, v) for k, v in nodes_data.items()]
        ),
        total=len(nodes_data),
        desc="Inserting entities into storage",
        unit="entity",
    ):
        try:
            entities_data.append(await result)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error occurred while inserting entities into storage: %s", e)


async def merge_edges(
    edges_data: dict,
    kg_instance: BaseGraphStorage,
    llm_client: TopkTokenModel,
    tokenizer_instance: Tokenizer,
    max_concurrent: int = 1000,
):
    """
    Merge edges

    :param edges_data
    :param kg_instance
    :param llm_client
    :param tokenizer_instance
    :param max_concurrent
    :return
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_edge(src_id: str, tgt_id: str, edge_data: list[dict]):
        async with semaphore:
            source_ids = []
            descriptions = []

            edge = await kg_instance.get_edge(src_id, tgt_id)
            if edge is not None:
                source_ids.extend(
                    split_string_by_multi_markers(edge["source_id"], ["<SEP>"])
                )
                descriptions.append(edge["description"])

            description = "<SEP>".join(
                sorted(set([dp["description"] for dp in edge_data] + descriptions))
            )
            source_id = "<SEP>".join(
                set([dp["source_id"] for dp in edge_data] + source_ids)
            )

            for insert_id in [src_id, tgt_id]:
                if not await kg_instance.has_node(insert_id):
                    await kg_instance.upsert_node(
                        insert_id,
                        node_data={
                            "source_id": source_id,
                            "description": description,
                            "entity_type": "UNKNOWN",
                        },
                    )

            description = await _handle_kg_summary(
                f"({src_id}, {tgt_id})", description, llm_client, tokenizer_instance
            )

            await kg_instance.upsert_edge(
                src_id,
                tgt_id,
                edge_data={"source_id": source_id, "description": description},
            )

            edge_data = {"src_id": src_id, "tgt_id": tgt_id, "description": description}
            return edge_data

    logger.info("Inserting relationships into storage...")
    relationships_data = []
    for result in tqdm_async(
        asyncio.as_completed(
            [
                process_single_edge(src_id, tgt_id, v)
                for (src_id, tgt_id), v in edges_data.items()
            ]
        ),
        total=len(edges_data),
        desc="Inserting relationships into storage",
        unit="relationship",
    ):
        try:
            relationships_data.append(await result)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Error occurred while inserting relationships into storage: %s", e
            )
