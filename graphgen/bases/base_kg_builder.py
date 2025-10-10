from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from graphgen.bases.base_llm_client import BaseLLMClient
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk


@dataclass
class BaseKGBuilder(ABC):
    llm_client: BaseLLMClient

    _nodes: Dict[str, List[dict]] = field(default_factory=lambda: defaultdict(list))
    _edges: Dict[Tuple[str, str], List[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @abstractmethod
    async def extract(
        self, chunk: Chunk
    ) -> Tuple[Dict[str, List[dict]], Dict[Tuple[str, str], List[dict]]]:
        """Extract nodes and edges from a single chunk."""
        raise NotImplementedError

    @abstractmethod
    async def merge_nodes(
        self,
        node_data: tuple[str, List[dict]],
        kg_instance: BaseGraphStorage,
    ) -> None:
        """Merge extracted nodes into the knowledge graph."""
        raise NotImplementedError

    @abstractmethod
    async def merge_edges(
        self,
        edges_data: tuple[Tuple[str, str], List[dict]],
        kg_instance: BaseGraphStorage,
    ) -> None:
        """Merge extracted edges into the knowledge graph."""
        raise NotImplementedError
