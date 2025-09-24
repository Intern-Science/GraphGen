from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class QAPair:
    """
    A pair of question and answer.
    """

    question: str
    answer: str
