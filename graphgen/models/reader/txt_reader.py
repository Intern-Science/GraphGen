from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader


class TxtReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append({self.text_column: line})
        return docs
