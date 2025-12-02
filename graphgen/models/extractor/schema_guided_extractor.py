import json
from typing import Dict, List

from graphgen.bases import BaseExtractor, BaseLLMWrapper
from graphgen.templates import SCHEMA_GUIDED_EXTRACTION_PROMPT
from graphgen.utils import compute_dict_hash, detect_main_language, logger


class SchemaGuidedExtractor(BaseExtractor):
    """
    Use JSON/YAML Schema or Pydantic Model to guide the LLM to extract structured information from text.

    Usage example:
        schema = {
                "type": "legal contract",
                "description": "A legal contract for leasing property.",
                "properties": {
                    "end_date": {"type": "string", "description": "The end date of the lease."},
                    "leased_space": {"type": "string", "description": "Description of the space that is being leased."},
                    "lessee": {"type": "string", "description": "The lessee's name (and possibly address)."},
                    "lessor": {"type": "string", "description": "The lessor's name (and possibly address)."},
                    "signing_date": {"type": "string", "description": "The date the contract was signed."},
                    "start_date": {"type": "string", "description": "The start date of the lease."},
                    "term_of_payment": {"type": "string", "description": "Description of the payment terms."},
                    "designated_use": {"type": "string",
                    "description": "Description of the designated use of the property being leased."},
                    "extension_period": {"type": "string",
                    "description": "Description of the extension options for the lease."},
                    "expiration_date_of_lease": {"type": "string", "description": "The expiration data of the lease."}
                },
                "required": ["lessee", "lessor", "start_date", "end_date"]
            }
        extractor = SchemaGuidedExtractor(llm_client, schema)
        result = extractor.extract(text)

    """

    def __init__(self, llm_client: BaseLLMWrapper, schema: dict):
        super().__init__(llm_client)
        self.schema = schema
        self.required_keys = self.schema.get("required")
        if not self.required_keys:
            # If no required keys are specified, use all keys from the schema as default
            self.required_keys = list(self.schema.get("properties", {}).keys())

    def build_prompt(self, text: str) -> str:
        schema_explanation = ""
        for field, details in self.schema.get("properties", {}).items():
            description = details.get("description", "No description provided.")
            schema_explanation += f'- "{field}": {description}\n'

        lang = detect_main_language(text)

        prompt = SCHEMA_GUIDED_EXTRACTION_PROMPT[lang].format(
            field=self.schema.get("name", "the document"),
            schema_explanation=schema_explanation,
            examples="",
            text=text,
        )
        return prompt

    async def extract(self, chunk: dict) -> dict:
        _chunk_id = list(chunk.keys())[0]
        text = chunk[_chunk_id].get("content", "")

        prompt = self.build_prompt(text)
        response = await self.llm_client.generate_answer(prompt)
        try:
            extracted_info = json.loads(response)
            # Ensure all required keys are present
            for key in self.required_keys:
                if key not in extracted_info:
                    extracted_info[key] = ""
            if any(extracted_info[key] == "" for key in self.required_keys):
                logger.debug("Missing required keys in extraction: %s", extracted_info)
                return {}
            main_keys_info = {key: extracted_info[key] for key in self.required_keys}
            logger.debug("Extracted info: %s", extracted_info)

            # add chunk metadata
            extracted_info["_chunk_id"] = _chunk_id

            return {
                compute_dict_hash(main_keys_info, prefix="extract-"): extracted_info
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse extraction response: %s", response)
            return {}

    @staticmethod
    async def merge_extractions(
        extraction_list: List[Dict[str, dict]]
    ) -> Dict[str, dict]:
        """
        Merge multiple extraction results based on their hashes.
        :param extraction_list: List of extraction results, each is a dict with hash as key and record as value.
        :return: Merged extraction results.
        """
        merged: Dict[str, dict] = {}
        for ext in extraction_list:
            for h, rec in ext.items():
                if h not in merged:
                    merged[h] = rec.copy()
                else:
                    for k, v in rec.items():
                        if k not in merged[h] or merged[h][k] == v:
                            merged[h][k] = v
                        else:
                            merged[h][k] = f"{merged[h][k]}<SEP>{v}"
        return merged
