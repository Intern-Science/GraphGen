import json
from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader


class JSONReader(BaseReader):
    """
    Reader for JSON files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for doc in data:
                    assert "type" in doc, f"Missing 'type' in document: {doc}"
                    if doc.get("type") == "text" and self.text_column not in doc:
                        raise ValueError(
                            f"Missing '{self.text_column}' in document: {doc}"
                        )
                return self.filter(data)
            raise ValueError("JSON file must contain a list of documents.")
