from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

from graphgen.bases import BaseTokenizer


@dataclass
class HFTokenizer(BaseTokenizer):
    def __post_init__(self):
        self.enc = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids, skip_special_tokens=True)
