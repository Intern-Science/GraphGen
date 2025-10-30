from typing import Any, List, Optional

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.datatypes import Token


# TODO: implement TGIWrapper methods
class TGIWrapper(BaseLLMWrapper):
    """
    Async inference backend based on TGI (Text-Generation-Inference)
    """

    def __init__(
        self,
        model_url: str,  # e.g. "http://localhost:8080"
        temperature: float = 0.0,
        top_p: float = 1.0,
        topk: int = 5,
        **kwargs: Any
    ):
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)

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
