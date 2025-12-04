from typing import List

import pandas as pd

from graphgen.bases import BaseGraphStorage, BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.common import init_llm, init_storage
from graphgen.utils import logger

from .build_mm_kg import build_mm_kg
from .build_text_kg import build_text_kg


class BuildKGService:
    def __init__(self, working_dir: str = "cache"):
        self.llm_client: BaseLLMWrapper = init_llm("synthesizer")
        self.graph_storage: BaseGraphStorage = init_storage(
            backend="networkx", working_dir=working_dir, namespace="graph"
        )

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        docs = batch.to_dict(orient="records")
        docs = [Chunk.from_dict(doc["_chunk_id"], doc) for doc in docs]

        # consume the chunks and build kg
        self.build_kg(docs)
        return pd.DataFrame()

    def build_kg(self, chunks: List[Chunk]) -> None:
        """
        Build knowledge graph (KG) and merge into kg_instance
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
            build_text_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=text_chunks,
            )
        if len(mm_chunks) == 0:
            logger.info("All multi-modal chunks are already in the storage")
        else:
            logger.info("[Multi-modal Entity and Relation Extraction] processing ...")
            build_mm_kg(
                llm_client=self.llm_client,
                kg_instance=self.graph_storage,
                chunks=mm_chunks,
            )

        self.graph_storage.index_done_callback()
