from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader


class TXTReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            docs = [{"type": "text", self.text_column: f.read()}]
        return self.filter(docs)
