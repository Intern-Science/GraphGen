from abc import ABC, abstractmethod
from typing import List, Optional

from graphgen.bases import Token


class TopkTokenModel(ABC):
    def __init__(
        self,
        do_sample: bool = False,
        temperature: float = 0,
        max_tokens: int = 4096,
        repetition_penalty: float = 1.05,
        num_beams: int = 1,
        topk: int = 50,
        topp: float = 0.95,
        topk_per_token: int = 5,
    ):
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.topk = topk
        self.topp = topp
        self.topk_per_token = topk_per_token

    @abstractmethod
    async def generate_topk_per_token(self, text: str) -> List[Token]:
        """
        Generate prob, text and candidates for each token of the model's output.
        This function is used to visualize the inference process.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_inputs_prob(
        self, text: str, history: Optional[List[str]] = None
    ) -> List[Token]:
        """
        Generate prob and text for each token of the input text.
        This function is used to visualize the ppl.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_answer(
        self, text: str, history: Optional[List[str]] = None
    ) -> str:
        """
        Generate answer from the model.
        """
        raise NotImplementedError
