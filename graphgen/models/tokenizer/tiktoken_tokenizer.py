from typing import List

import tiktoken

from graphgen.bases import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = "cl100k_base"):
        super().__init__(model_name)
        self.enc = tiktoken.get_encoding(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids)
