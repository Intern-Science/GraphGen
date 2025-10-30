import asyncio
import os
import time
from typing import Dict, cast

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import StorageNameSpace
from graphgen.bases.datatypes import Chunk
from graphgen.models import (
    JsonKVStorage,
    JsonListStorage,
    NetworkXStorage,
    OpenAIClient,
    Tokenizer,
)
from graphgen.operators import (
    build_kg,
    chunk_documents,
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
        self.qa_storage: JsonListStorage = JsonListStorage(
            os.path.join(self.working_dir, "data", "graphgen", f"{self.unique_id}"),
            namespace="qa",
        )

        # webui
        self.progress_bar: gr.Progress = progress_bar

    @async_to_sync_method
    async def insert(self, read_config: Dict, split_config: Dict):
        """
        insert chunks into the graph
        """
        # Step 1: Read files
        data = read_files(read_config["input_file"], self.working_dir)
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

        inserting_chunks = await chunk_documents(
            new_docs,
            split_config["chunk_size"],
            split_config["chunk_overlap"],
            self.tokenizer_instance,
            self.progress_bar,
        )

        # _add_chunk_keys = await self.chunks_storage.filter_keys(
        #     list(inserting_chunks.keys())
        # )
        # inserting_chunks = {
        #     k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        # }
        #
        # if len(inserting_chunks) == 0:
        #     logger.warning("All chunks are already in the storage")
        #     return

        logger.info("[New Chunks] inserting %d chunks", len(inserting_chunks))
        await self.chunks_storage.upsert(inserting_chunks)

        _add_entities_and_relations = await build_kg(
            llm_client=self.synthesizer_llm_client,
            kg_instance=self.graph_storage,
            chunks=[Chunk.from_dict(k, v) for k, v in inserting_chunks.items()],
            anchor_type=read_config.get("anchor_type", None),
            progress_bar=self.progress_bar,
        )
        if not _add_entities_and_relations:
            logger.warning("No entities or relations extracted from text chunks")
            return

        await self._insert_done()
        return _add_entities_and_relations

    async def _insert_done(self):
        tasks = []
        for storage_instance in [
            self.full_docs_storage,
            self.chunks_storage,
            self.graph_storage,
            self.search_storage,
        ]:
            if storage_instance is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_instance).index_done_callback())
        await asyncio.gather(*tasks)

    @async_to_sync_method
    async def search(self, search_config: Dict):
        logger.info(
            "Search is %s", "enabled" if search_config["enabled"] else "disabled"
        )
        if search_config["enabled"]:
            logger.info("[Search] %s ...", ", ".join(search_config["search_types"]))
            all_nodes = await self.graph_storage.get_all_nodes()
            all_nodes_names = [node[0] for node in all_nodes]
            new_search_entities = await self.full_docs_storage.filter_keys(
                all_nodes_names
            )
            logger.info(
                "[Search] Found %d entities to search", len(new_search_entities)
            )
            _add_search_data = await search_all(
                search_types=search_config["search_types"],
                search_entities=new_search_entities,
            )
            if _add_search_data:
                await self.search_storage.upsert(_add_search_data)
                logger.info("[Search] %d entities searched", len(_add_search_data))

                # Format search results for inserting
                search_results = []
                for _, search_data in _add_search_data.items():
                    search_results.extend(
                        [
                            {"content": search_data[key]}
                            for key in list(search_data.keys())
                        ]
                    )
                # TODO: fix insert after search
                await self.insert()

    @async_to_sync_method
    async def quiz_and_judge(self, quiz_and_judge_config: Dict):
        if quiz_and_judge_config is None or not quiz_and_judge_config.get(
            "enabled", False
        ):
            logger.warning("Quiz and Judge is not used in this pipeline.")
            return
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

    @async_to_sync_method
    async def generate(self, partition_config: Dict, generate_config: Dict):
        # Step 1: partition the graph
        batches = await partition_kg(
            self.graph_storage,
            self.chunks_storage,
            self.tokenizer_instance,
            partition_config,
        )

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
