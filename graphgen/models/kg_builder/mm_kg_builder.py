import re
from collections import defaultdict
from typing import Dict, List, Tuple

from graphgen.bases import Chunk
from graphgen.templates import MMKG_EXTRACTION_PROMPT
from graphgen.utils import (
    detect_main_language,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    logger,
    split_string_by_multi_markers,
)

from .light_rag_kg_builder import LightRAGKGBuilder


class MMKGBuilder(LightRAGKGBuilder):
    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """
        Extract entities and relationships from a single multi-modal chunk using the LLM client.
        Expect to get a mini graph which contains a central multi-modal entity
        and its related text entities and relationships.
        Like:
        (image: "image_of_eiffel_tower") --[located_in]--> (text: "Paris")
        (image: "image_of_eiffel_tower") --[built_in]--> (text: "1889")
        (text: "Eiffel Tower") --[height]--> (text: "324 meters")
        :param chunk
        """
        chunk_id = chunk.id
        chunk_type = chunk.type  # image | table | formula | ...
        metadata = chunk.metadata

        # choose different extraction strategies based on chunk type
        if chunk_type == "image":
            image_caption = "\n".join(metadata.get("image_caption", ""))
            language = detect_main_language(image_caption)
            prompt_template = MMKG_EXTRACTION_PROMPT[language].format(
                **MMKG_EXTRACTION_PROMPT["FORMAT"],
                chunk_type=chunk_type,
                chunk_id=chunk_id,
                chunk_text=image_caption,
            )
            result = await self.llm_client.generate_answer(prompt_template)
            logger.debug("Image chunk extraction result: %s", result)

            # parse the result
            records = split_string_by_multi_markers(
                result,
                [
                    MMKG_EXTRACTION_PROMPT["FORMAT"]["record_delimiter"],
                    MMKG_EXTRACTION_PROMPT["FORMAT"]["completion_delimiter"],
                ],
            )

            nodes = defaultdict(list)
            edges = defaultdict(list)

            for record in records:
                match = re.search(r"\((.*)\)", record)
                if not match:
                    continue
                inner = match.group(1)

                attributes = split_string_by_multi_markers(
                    inner, [MMKG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
                )

                entity = await handle_single_entity_extraction(attributes, chunk_id)
                if entity is not None:
                    nodes[entity["entity_name"]].append(entity)
                    continue

                relation = await handle_single_relationship_extraction(
                    attributes, chunk_id
                )
                if relation is not None:
                    key = (relation["src_id"], relation["tgt_id"])
                    edges[key].append(relation)

            return dict(nodes), dict(edges)

        if chunk_type == "table":
            pass  # TODO: implement table-based entity and relationship extraction
        if chunk_type == "formula":
            pass  # TODO: implement formula-based entity and relationship extraction

        logger.error("Unsupported chunk type for MMKGBuilder: %s", chunk_type)
        return defaultdict(list), defaultdict(list)
