from typing import Any, List, Optional

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


# TODO: implement TensorRTWrapper methods
class TensorRTWrapper(BaseLLMWrapper):
    """
    Async inference backend based on TensorRT-LLM
    """

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
