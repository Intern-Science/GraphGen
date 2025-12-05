from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar("T")


@dataclass
class StorageNameSpace:
    working_dir: str = None
    namespace: str = None

    def index_done_callback(self):
        """commit the storage operations after indexing"""

    def query_done_callback(self):
        """commit the storage operations after querying"""


class BaseKVStorage(Generic[T], StorageNameSpace):
    def all_keys(self) -> list[str]:
        raise NotImplementedError

    def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    def get_all(self) -> dict[str, T]:
        raise NotImplementedError

    def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    def drop(self):
        raise NotImplementedError


class BaseGraphStorage(StorageNameSpace):
    def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    def update_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    def get_all_nodes(self) -> Union[list[tuple[str, dict]], None]:
        raise NotImplementedError

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    def update_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    def get_all_edges(self) -> Union[list[tuple[str, str, dict]], None]:
        raise NotImplementedError

    def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    def delete_node(self, node_id: str):
        raise NotImplementedError
