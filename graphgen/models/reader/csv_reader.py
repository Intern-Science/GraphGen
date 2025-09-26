from typing import Any, Dict, List

import pandas as pd

from graphgen.bases.base_reader import BaseReader


class CsvReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:

        df = pd.read_csv(file_path)
        if self.text_column not in df.columns:
            raise ValueError(f"Missing '{self.text_column}' column in CSV file.")
        return df.to_dict(orient="records")
