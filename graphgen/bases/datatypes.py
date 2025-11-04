import math
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Chunk:
    id: str
    content: str
    type: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(key: str, data: dict) -> "Chunk":
        return Chunk(
            id=key,
            content=data.get("content", ""),
            type=data.get("type", "text"),
            metadata={k: v for k, v in data.items() if k != "content"},
        )


@dataclass
class QAPair:
    """
    A pair of question and answer.
    """

    question: str
    answer: str


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
class Community:
    id: Union[int, str]
    nodes: List[str] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
