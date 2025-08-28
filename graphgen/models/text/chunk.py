from dataclasses import dataclass


@dataclass
class Chunk:
    id : str
    content: str
