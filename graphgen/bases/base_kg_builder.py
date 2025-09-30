from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from graphgen.bases.base_llm_client import BaseLLMClient
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk


@dataclass
class BaseKGBuilder(ABC):
    kg_instance: BaseGraphStorage
    llm_client: BaseLLMClient

    _nodes: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    _edges: Dict[Tuple[str, str], List[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def build(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    async def extract_all(self, chunks: List[Chunk]) -> None:
        """Extract nodes and edges from all chunks."""
        raise NotImplementedError

    @abstractmethod
    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """Extract nodes and edges from a single chunk."""
        raise NotImplementedError

    @abstractmethod
    async def merge_nodes(
        self, nodes_data: Dict[str, List[dict]], kg_instance: BaseGraphStorage, llm
    ) -> None:
        """Merge extracted nodes into the knowledge graph."""
        raise NotImplementedError
