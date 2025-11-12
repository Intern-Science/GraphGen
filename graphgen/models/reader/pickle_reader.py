import pickle
from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader


class PickleReader(BaseReader):
    """
    Read pickle files, requiring the top-level object to be List[Dict[str, Any]].

    Columns:
    - type: The type of the document (e.g., "text", "image", etc.)
    - if type is "text", "content" column must be present.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, list):
            raise ValueError("Pickle file must contain a list of documents.")

        for doc in data:
            if not isinstance(doc, dict):
                raise ValueError("Every item in the list must be a dict.")
            assert "type" in doc, f"Missing 'type' in document: {doc}"
            if doc.get("type") == "text" and self.text_column not in doc:
                raise ValueError(f"Missing '{self.text_column}' in document: {doc}")

        return self.filter(data)
