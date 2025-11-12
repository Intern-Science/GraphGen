import os
import time
from typing import Dict

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Chunk
from graphgen.engine import op
from graphgen.models import (
    JsonKVStorage,
    JsonListStorage,
    MetaJsonKVStorage,
    NetworkXStorage,
    OpenAIClient,
    Tokenizer,
)
from graphgen.operators import (
    build_kg,
    chunk_documents,
    extract_info,
    generate_qas,
    init_llm,
    judge_statement,
    partition_kg,
    quiz,
    read_files,
    search_all,
)
from graphgen.utils import async_to_sync_method, compute_mm_hash, logger

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class GraphGen:
    def __init__(
        self,
        unique_id: int = int(time.time()),
        working_dir: str = os.path.join(sys_path, "cache"),
        tokenizer_instance: Tokenizer = None,
        synthesizer_llm_client: OpenAIClient = None,
        trainee_llm_client: OpenAIClient = None,
        progress_bar: gr.Progress = None,
    ):
        self.unique_id: int = unique_id
        self.working_dir: str = working_dir

        # llm
        self.tokenizer_instance: Tokenizer = tokenizer_instance or Tokenizer(
            model_name=os.getenv("TOKENIZER_MODEL")
        )

        self.synthesizer_llm_client: BaseLLMWrapper = (
            synthesizer_llm_client or init_llm("synthesizer")
        )
        self.trainee_llm_client: BaseLLMWrapper = trainee_llm_client

        self.meta_storage: MetaJsonKVStorage = MetaJsonKVStorage(
            self.working_dir, namespace="_meta"
        )
        self.full_docs_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="full_docs"
        )
        self.chunks_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="chunks"
        )
        self.graph_storage: NetworkXStorage = NetworkXStorage(
            self.working_dir, namespace="graph"
        )
        self.search_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="search"
        )
        self.rephrase_storage: JsonKVStorage = JsonKVStorage(
            self.working_dir, namespace="rephrase"
        )
        self.partition_storage: JsonListStorage = JsonListStorage(
            self.working_dir, namespace="partition"
        )
        self.qa_storage: JsonListStorage = JsonListStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="qa",
        )
        self.extract_storage: JsonKVStorage = JsonKVStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="extraction",
        )

        # webui
        self.progress_bar: gr.Progress = progress_bar

    @op("read", deps=[])
    @async_to_sync_method
    async def read(self, read_config: Dict):
        """
        read files from input sources
        """
        data = read_files(**read_config, cache_dir=self.working_dir)
        if len(data) == 0:
            logger.warning("No data to process")
            return

        assert isinstance(data, list) and isinstance(data[0], dict)

        # TODO: configurable whether to use coreference resolution

        new_docs = {compute_mm_hash(doc, prefix="doc-"): doc for doc in data}
        _add_doc_keys = await self.full_docs_storage.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if len(new_docs) == 0:
            logger.warning("All documents are already in the storage")
            return

        await self.full_docs_storage.upsert(new_docs)
        await self.full_docs_storage.index_done_callback()

    @op("chunk", deps=["read"])
    @async_to_sync_method
    async def chunk(self, chunk_config: Dict):
        """
        chunk documents into smaller pieces from full_docs_storage if not already present
        """

        new_docs = await self.meta_storage.get_new_data(self.full_docs_storage)
        if len(new_docs) == 0:
            logger.warning("All documents are already in the storage")
            return

        inserting_chunks = await chunk_documents(
            new_docs,
            self.tokenizer_instance,
            self.progress_bar,
            **chunk_config,
        )

        _add_chunk_keys = await self.chunks_storage.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }

        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return

        await self.chunks_storage.upsert(inserting_chunks)
        await self.chunks_storage.index_done_callback()
        await self.meta_storage.mark_done(self.full_docs_storage)
        await self.meta_storage.index_done_callback()

    @op("build_kg", deps=["chunk"])
    @async_to_sync_method
    async def build_kg(self):
        """
        build knowledge graph from text chunks
        """
        # Step 1: get new chunks according to meta and chunks storage
        inserting_chunks = await self.meta_storage.get_new_data(self.chunks_storage)
        if len(inserting_chunks) == 0:
            logger.warning("All chunks are already in the storage")
            return

        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))
        # Step 2: build knowledge graph from new chunks
        _add_entities_and_relations = await build_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            chunks=[Chunk.from_dict(k, v) for k, v in inserting_chunks.items()],
            progress_bar=self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted from text chunks")
            return

        # Step 3: mark meta
        await self.graph_storage.index_done_callback()
        await self.meta_storage.mark_done(self.chunks_storage)
        await self.meta_storage.index_done_callback()

        return _add_entities_and_relations

    @op("search", deps=["read"])
    @async_to_sync_method
    async def search(self, search_config: Dict):
        logger.info("[Search] %s ...", ", ".join(search_config["data_sources"]))

        seeds = await self.meta_storage.get_new_data(self.full_docs_storage)
        if len(seeds) == 0:
            logger.warning("All documents are already been searched")
            return
        search_results = await search_all(
            seed_data=seeds,
            search_config=search_config,
        )

        _add_search_keys = await self.search_storage.filter_keys(
            list(search_results.keys())
        )
        search_results = {
            k: v for k, v in search_results.items() if k in _add_search_keys
        }
        if len(search_results) == 0:
            logger.warning("All search results are already in the storage")
            return
        await self.search_storage.upsert(search_results)
        await self.search_storage.index_done_callback()
        await self.meta_storage.mark_done(self.full_docs_storage)
        await self.meta_storage.index_done_callback()

    @op("quiz_and_judge", deps=["build_kg"])
    @async_to_sync_method
    async def quiz_and_judge(self, quiz_and_judge_config: Dict):
        logger.warning(
            "Quiz and Judge operation needs trainee LLM client."
            " Make sure to provide one."
        )
        max_samples = quiz_and_judge_config["quiz_samples"]
        await quiz(
            self.synthesizer_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            max_samples,
        )

        # TODO： assert trainee_llm_client is valid before judge
        if not self.trainee_llm_client:
            # TODO: shutdown existing synthesizer_llm_client properly
            logger.info("No trainee LLM client provided, initializing a new one.")
            self.synthesizer_llm_client.shutdown()
            self.trainee_llm_client = init_llm("trainee")

        re_judge = quiz_and_judge_config["re_judge"]
        _update_relations = await judge_statement(
            self.trainee_llm_client,
            self.graph_storage,
            self.rephrase_storage,
            re_judge,
        )

        await self.rephrase_storage.index_done_callback()
        await _update_relations.index_done_callback()

        logger.info("Shutting down trainee LLM client.")
        self.trainee_llm_client.shutdown()
        self.trainee_llm_client = None
        logger.info("Restarting synthesizer LLM client.")
        self.synthesizer_llm_client.restart()

    @op("partition", deps=["build_kg"])
    @async_to_sync_method
    async def partition(self, partition_config: Dict):
        batches = await partition_kg(
            self.graph_storage,
            self.chunks_storage,
            self.tokenizer_instance,
            partition_config,
        )
        await self.partition_storage.upsert(batches)
        return batches

    @op("extract", deps=["chunk"])
    @async_to_sync_method
    async def extract(self, extract_config: Dict):
        logger.info("Extracting information from given chunks...")

        results = await extract_info(
            self.synthesizer_llm_client,
            self.chunks_storage,
            extract_config,
            progress_bar=self.progress_bar,
        )
        if not results:
            logger.warning("No information extracted")
            return

        await self.extract_storage.upsert(results)
        await self.extract_storage.index_done_callback()
        await self.meta_storage.mark_done(self.chunks_storage)
        await self.meta_storage.index_done_callback()

    @op("generate", deps=["partition"])
    @async_to_sync_method
    async def generate(self, generate_config: Dict):

        batches = self.partition_storage.data
        if not batches:
            logger.warning("No partitions found for QA generation")
            return

        # Step 2： generate QA pairs
        results = await generate_qas(
            self.synthesizer_llm_client,
            batches,
            generate_config,
            progress_bar=self.progress_bar,
        )

        if not results:
            logger.warning("No QA pairs generated")
            return

        # Step 3: store the generated QA pairs
        await self.qa_storage.upsert(results)
        await self.qa_storage.index_done_callback()

    @async_to_sync_method
    async def clear(self):
        await self.full_docs_storage.drop()
        await self.chunks_storage.drop()
        await self.search_storage.drop()
        await self.graph_storage.clear()
        await self.rephrase_storage.drop()
        await self.qa_storage.drop()

        logger.info("All caches are cleared")

    # TODO: add data filtering step here in the future
    # graph_gen.filter(filter_config=config["filter"])
