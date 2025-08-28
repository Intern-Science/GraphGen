from dataclasses import dataclass

@dataclass
class TextPair:
    """
    A pair of input data.
    """
    question: str
    answer: str
