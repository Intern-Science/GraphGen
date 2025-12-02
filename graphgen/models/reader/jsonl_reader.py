import json
from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger


class JSONLReader(BaseReader):
    """
    Reader for JSONL files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    assert "type" in doc, f"Missing 'type' in document: {doc}"
                    if doc.get("type") == "text" and self.text_column not in doc:
                        raise ValueError(
                            f"Missing '{self.text_column}' in document: {doc}"
                        )
                    docs.append(doc)
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON line: %s. Error: %s", line, e)
        return self.filter(docs)
