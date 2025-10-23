from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from graphgen.bases.base_llm_client import BaseLLMClient
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk


class BaseKGBuilder(ABC):
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self._nodes: Dict[str, List[dict]] = defaultdict(list)
        self._edges: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

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
