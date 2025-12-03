import math
from dataclasses import dataclass, field
from typing import List, Union

from pydantic import BaseModel, Field, field_validator


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


class Node(BaseModel):
    id: str = Field(..., description="unique node id")
    op_name: str = Field(..., description="operator name")
    type: str = Field(
        ..., description="task type, e.g., map, filter, flatmap, aggregate, map_batch"
    )
    params: dict = Field(default_factory=dict, description="operator parameters")
    dependencies: List[str] = Field(
        default_factory=list, description="list of dependent node ids"
    )

    @classmethod
    @field_validator("type")
    def validate_type(cls, v: str) -> str:
        valid_types = {"map", "filter", "flatmap", "aggregate", "map_batch"}
        if v not in valid_types:
            raise ValueError(f"Invalid node type: {v}. Must be one of {valid_types}.")
        return v


class Config(BaseModel):
    nodes: List[Node] = Field(
        ..., min_length=1, description="list of nodes in the computation graph"
    )

    @classmethod
    @field_validator("nodes")
    def validate_unique_ids(cls, v: List[Node]) -> List[Node]:
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            duplicates = {id_ for id_ in ids if ids.count(id_) > 1}
            raise ValueError(f"Duplicate node ids found: {duplicates}")
        return v
