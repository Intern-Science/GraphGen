import json
from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader


class JsonReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for doc in data:
                    if self.text_column not in doc:
                        raise ValueError(
                            f"Missing '{self.text_column}' in document: {doc}"
                        )
                return data
            raise ValueError("JSON file must contain a list of documents.")
