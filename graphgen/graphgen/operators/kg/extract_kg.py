import asyncio
import re
from collections import defaultdict
from typing import List

import gradio as gr
from tqdm.asyncio import tqdm as tqdm_async

from graphgen.models import Chunk, OpenAIModel, Tokenizer
from graphgen.models.storage.base_storage import BaseGraphStorage
from graphgen.operators.kg.merge_kg import merge_edges, merge_nodes
from graphgen.templates import KG_EXTRACTION_PROMPT
from graphgen.utils import (
    detect_if_chinese,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    logger,
    pack_history_conversations,
    split_string_by_multi_markers,
)


# pylint: disable=too-many-statements
async def extract_kg(
    llm_client: OpenAIModel,
    kg_instance: BaseGraphStorage,
    tokenizer_instance: Tokenizer,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
    max_concurrent: int = 1000,
):
    """
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param tokenizer_instance
    :param chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :param max_concurrent
    :return:
    """

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_single_content(chunk: Chunk, max_loop: int = 3):
        async with semaphore:
            chunk_id = chunk.id
            content = chunk.content
            if detect_if_chinese(content):
                language = "Chinese"
            else:
                language = "English"
            KG_EXTRACTION_PROMPT["FORMAT"]["language"] = language

            hint_prompt = KG_EXTRACTION_PROMPT[language]["TEMPLATE"].format(
                **KG_EXTRACTION_PROMPT["FORMAT"], input_text=content
            )

            final_result = await llm_client.generate_answer(hint_prompt)
            logger.info("First result: %s", final_result)

            history = pack_history_conversations(hint_prompt, final_result)
            for loop_index in range(max_loop):
                if_loop_result = await llm_client.generate_answer(
                    text=KG_EXTRACTION_PROMPT[language]["IF_LOOP"], history=history
                )
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

                glean_result = await llm_client.generate_answer(
                    text=KG_EXTRACTION_PROMPT[language]["CONTINUE"], history=history
                )
                logger.info("Loop %s glean: %s", loop_index, glean_result)

                history += pack_history_conversations(
                    KG_EXTRACTION_PROMPT[language]["CONTINUE"], glean_result
                )
                final_result += glean_result
                if loop_index == max_loop - 1:
                    break

            records = split_string_by_multi_markers(
                final_result,
                [
                    KG_EXTRACTION_PROMPT["FORMAT"]["record_delimiter"],
                    KG_EXTRACTION_PROMPT["FORMAT"]["completion_delimiter"],
                ],
            )

            nodes = defaultdict(list)
            edges = defaultdict(list)

            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)  # 提取括号内的内容
                record_attributes = split_string_by_multi_markers(
                    record, [KG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
                )

                entity = await handle_single_entity_extraction(
                    record_attributes, chunk_id
                )
                if entity is not None:
                    nodes[entity["entity_name"]].append(entity)
                    continue
                relation = await handle_single_relationship_extraction(
                    record_attributes, chunk_id
                )
                if relation is not None:
                    edges[(relation["src_id"], relation["tgt_id"])].append(relation)
            return dict(nodes), dict(edges)

    results = []
    chunk_number = len(chunks)
    async for result in tqdm_async(
        asyncio.as_completed([_process_single_content(c) for c in chunks]),
        total=len(chunks),
        desc="[2/4]Extracting entities and relationships from chunks",
        unit="chunk",
    ):
        try:
            if progress_bar is not None:
                progress_bar(
                    len(results) / chunk_number,
                    desc="[3/4]Extracting entities and relationships from chunks",
                )
            results.append(await result)
            if progress_bar is not None and len(results) == chunk_number:
                progress_bar(
                    1, desc="[3/4]Extracting entities and relationships from chunks"
                )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Error occurred while extracting entities and relationships from chunks: %s",
                e,
            )

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    await merge_nodes(nodes, kg_instance, llm_client, tokenizer_instance)
    await merge_edges(edges, kg_instance, llm_client, tokenizer_instance)

    return kg_instance
