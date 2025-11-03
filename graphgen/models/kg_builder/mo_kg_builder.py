import re
from collections import defaultdict
from typing import Dict, List, Tuple

from graphgen.bases import Chunk
from graphgen.templates import PROTEIN_KG_EXTRACTION_PROMPT
from graphgen.utils import (
    detect_main_language,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    logger,
    split_string_by_multi_markers,
)

from .light_rag_kg_builder import LightRAGKGBuilder


class MOKGBuilder(LightRAGKGBuilder):
    @staticmethod
    async def scan_document_for_schema(
        chunk: Chunk, schema: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """
        Scan the document chunk to extract entities and relationships based on the provided schema.
        :param chunk: The document chunk to be scanned.
        :param schema: A dictionary defining the entities and relationships to be extracted.
        :return: A tuple containing two dictionaries - one for entities and one for relationships.
        """
        # TODO: use hard-coded PROTEIN_KG_EXTRACTION_PROMPT for protein chunks,
        #  support schema for other chunk types later
        print(chunk.id, schema)
        return {}, {}

    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """
        Multi-Omics Knowledge Graph Builder
        Step1: Extract and output a JSON object containing protein information from the given chunk.
        Step2: Get more details about the protein by querying external databases if necessary.
        Step3: Construct entities and relationships for the protein knowledge graph.
        Step4: Return the entities and relationships.
        :param chunk
        :return: Tuple containing entities and relationships.
        """
        # TODO: Implement the multi-omics KG extraction logic here
        chunk_id = chunk.id
        chunk_type = chunk.type  # genome | protein | ...
        metadata = chunk.metadata

        # choose different extraction strategies based on chunk type
        if chunk_type == "protein":
            protein_caption = ""
            for key, value in metadata["protein_caption"].items():
                protein_caption += f"{key}: {value}\n"
            logger.debug("Protein chunk caption: %s", protein_caption)

            language = detect_main_language(protein_caption)
            prompt_template = PROTEIN_KG_EXTRACTION_PROMPT[language].format(
                **PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"],
                input_text=protein_caption,
            )
            result = await self.llm_client.generate_answer(prompt_template)
            logger.debug("Protein chunk extraction result: %s", result)

            # parse the result
            records = split_string_by_multi_markers(
                result,
                [
                    PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["record_delimiter"],
                    PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["completion_delimiter"],
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
                    inner, [PROTEIN_KG_EXTRACTION_PROMPT["FORMAT"]["tuple_delimiter"]]
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
