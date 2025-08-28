import math
from dataclasses import dataclass, field
from typing import List, Union, Optional


@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: Union[float, None] = field(default=None)

    @property
    def logprob(self) -> float:
        return math.log(self.prob)


@dataclass
class TopkTokenModel:
    do_sample: bool = False
    temperature: float = 0
    max_tokens: int = 4096
    repetition_penalty: float = 1.05
    num_beams: int = 1
    topk: int = 50
    topp: float = 0.95

    topk_per_token: int = 5  # number of topk tokens to generate for each token

    async def generate_topk_per_token(self, text: str) -> List[Token]:
        """
        Generate prob, text and candidates for each token of the model's output.
        This function is used to visualize the inference process.
        """
        raise NotImplementedError

    async def generate_inputs_prob(self, text: str, history: Optional[List[str]] = None) -> List[Token]:
        """
        Generate prob and text for each token of the input text.
        This function is used to visualize the ppl.
        """
        raise NotImplementedError

    async def generate_answer(self, text: str, history: Optional[List[str]] = None) -> str:
        """
        Generate answer from the model.
        """
        raise NotImplementedError
