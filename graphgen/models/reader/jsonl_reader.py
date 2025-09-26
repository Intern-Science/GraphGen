import json
from typing import Any, Dict, List

from graphgen.bases.base_reader import BaseReader
from graphgen.utils import logger


class JsonlReader(BaseReader):
    def read(self, file_path: str) -> List[Dict[str, Any]]:
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if self.text_column in doc:
                        docs.append(doc)
                    else:
                        raise ValueError(
                            f"Missing '{self.text_column}' in document: {doc}"
                        )
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON line: %s. Error: %s", line, e)
        return docs
