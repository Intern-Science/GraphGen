from typing import List

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.utils import logger

from .build_mm_kg import build_mm_kg
from .build_text_kg import build_text_kg


async def build_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
):
    """
    Build knowledge graph (KG) and merge into kg_instance
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param anchor_type: get this type of information from chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :return:
    """

    text_chunks = [chunk for chunk in chunks if chunk.type == "text"]
    mm_chunks = [
        chunk
        for chunk in chunks
        if chunk.type in ("image", "video", "table", "formula")
    ]

    if len(text_chunks) == 0:
        logger.info("All text chunks are already in the storage")
    else:
        logger.info("[Text Entity and Relation Extraction] processing ...")
        await build_text_kg(
            llm_client=llm_client,
            kg_instance=kg_instance,
            chunks=text_chunks,
            progress_bar=progress_bar,
        )

    if len(mm_chunks) == 0:
        logger.info("All multi-modal chunks are already in the storage")
    else:
        logger.info("[Multi-modal Entity and Relation Extraction] processing ...")
        await build_mm_kg(
            llm_client=llm_client,
            kg_instance=kg_instance,
            chunks=mm_chunks,
            progress_bar=progress_bar,
        )

    return kg_instance
