# TODO: implement ollama client
from typing import Any, List, Optional

from graphgen.bases import BaseLLMClient, Token


class OllamaClient(BaseLLMClient):
    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> str:
        pass

    async def generate_topk_per_token(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        pass

    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None, **extra: Any
    ) -> List[Token]:
        pass
