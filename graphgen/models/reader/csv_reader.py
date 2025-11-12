from typing import Any, Dict, List

import pandas as pd

from graphgen.bases.base_reader import BaseReader


class CSVReader(BaseReader):
    """
    Reader for CSV files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(self, file_path: str) -> List[Dict[str, Any]]:

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            assert "type" in row, f"Missing 'type' column in document: {row.to_dict()}"
            if row["type"] == "text" and self.text_column not in row:
                raise ValueError(
                    f"Missing '{self.text_column}' in document: {row.to_dict()}"
                )
        return self.filter(df.to_dict(orient="records"))
