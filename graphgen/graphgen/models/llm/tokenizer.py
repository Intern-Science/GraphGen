from dataclasses import dataclass
from typing import List
import tiktoken

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


def get_tokenizer(tokenizer_name: str = "cl100k_base"):
    """
    Get a tokenizer instance by name.

    :param tokenizer_name: tokenizer name, tiktoken encoding name or Hugging Face model name
    :return: tokenizer instance
    """
    if tokenizer_name in tiktoken.list_encoding_names():
        return tiktoken.get_encoding(tokenizer_name)
    if TRANSFORMERS_AVAILABLE:
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from Hugging Face: {e}") from e
    else:
        raise ValueError("Hugging Face Transformers is not available, please install it first.")

@dataclass
class Tokenizer:
    model_name: str = "cl100k_base"

    def __post_init__(self):
        self.tokenizer = get_tokenizer(self.model_name)

    def encode_string(self, text: str) -> List[int]:
        """
        Encode text to tokens

        :param text
        :return: tokens
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: List[int]) -> str:
        """
        Decode tokens to text

        :param tokens
        :return: text
        """
        return self.tokenizer.decode(tokens)

    def chunk_by_token_size(
        self, content: str, overlap_token_size=128, max_token_size=1024
    ):
        tokens = self.encode_string(content)
        results = []
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = self.decode_tokens(
                tokens[start : start + max_token_size]
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        return results
