from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class BaseTokenizer(ABC):
    model_name: str = "cl100k_base"

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text -> token ids."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids -> text."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def chunk_by_token_size(
        self,
        content: str,
        *,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
    ) -> List[dict]:
        tokens = self.encode(content)
        results = []
        step = max_token_size - overlap_token_size
        for index, start in enumerate(range(0, len(tokens), step)):
            chunk_ids = tokens[start : start + max_token_size]
            results.append(
                {
                    "tokens": len(chunk_ids),
                    "content": self.decode(chunk_ids).strip(),
                    "chunk_order_index": index,
                }
            )
        return results
