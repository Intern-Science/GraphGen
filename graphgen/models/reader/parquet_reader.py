from typing import Any, Dict, List

import pandas as pd

from graphgen.bases.base_reader import BaseReader


class ParquetReader(BaseReader):
    """
    Read parquet files, requiring the schema to be restored to List[Dict[str, Any]].
    Columns:
    - type: The type of the document (e.g., "text", "image", etc.)
    - if type is "text", "content" column must be present.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:
        df = pd.read_parquet(file_path)
        data: List[Dict[str, Any]] = df.to_dict(orient="records")

        for doc in data:
            assert "type" in doc, f"Missing 'type' in document: {doc}"
            if doc.get("type") == "text" and self.text_column not in doc:
                raise ValueError(f"Missing '{self.text_column}' in document: {doc}")
        return self.filter(data)
