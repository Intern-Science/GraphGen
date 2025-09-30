from dataclasses import dataclass, field
from typing import List

from graphgen.bases import BaseTokenizer

from .hf_tokenizer import HFTokenizer
from .tiktoken_tokenizer import TiktokenTokenizer

try:
    from transformers import AutoTokenizer

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def get_tokenizer_impl(tokenizer_name: str = "cl100k_base") -> BaseTokenizer:
    import tiktoken

    if tokenizer_name in tiktoken.list_encoding_names():
        return TiktokenTokenizer(model_name=tokenizer_name)

    # 2. HuggingFace
    if _HF_AVAILABLE:
        return HFTokenizer(model_name=tokenizer_name)

    raise ValueError(
        f"Unknown tokenizer {tokenizer_name} and HuggingFace not available."
    )


@dataclass
class Tokenizer(BaseTokenizer):
    """
    Encapsulates different tokenization implementations based on the specified model name.
    """

    model_name: str = "cl100k_base"
    _impl: BaseTokenizer = field(init=False, repr=False)

    def __post_init__(self):
        self._impl = get_tokenizer_impl(self.model_name)

    def encode(self, text: str) -> List[int]:
        return self._impl.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._impl.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        return self._impl.count_tokens(text)
