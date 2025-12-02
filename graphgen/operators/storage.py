import os
from typing import Any

import ray

from graphgen.models import JsonKVStorage, JsonListStorage, NetworkXStorage


@ray.remote
class StorageManager:
    """
    Centralized storage for all operators

    Example Usage:
    ----------
    # init
    storage_manager = StorageManager.remote(working_dir="/path/to/dir", unique_id=123)

    # visit storage in tasks
    @ray.remote
    def some_task(storage_manager):
        full_docs_storage = ray.get(storage_manager.get_storage.remote("full_docs"))

    # visit storage in other actors
    @ray.remote
    class SomeOperator:
        def __init__(self, storage_manager):
            self.storage_manager = storage_manager
        def some_method(self):
            full_docs_storage = ray.get(self.storage_manager.get_storage.remote("full_docs"))
    """

    def __init__(self, working_dir: str, unique_id: int):
        self.working_dir = working_dir
        self.unique_id = unique_id

        # Initialize all storage backends
        self.storages = {
            "full_docs": JsonKVStorage(working_dir, namespace="full_docs"),
            "chunks": JsonKVStorage(working_dir, namespace="chunks"),
            "graph": NetworkXStorage(working_dir, namespace="graph"),
            "rephrase": JsonKVStorage(working_dir, namespace="rephrase"),
            "partition": JsonListStorage(working_dir, namespace="partition"),
            "search": JsonKVStorage(
                os.path.join(working_dir, "data", "graphgen", f"{unique_id}"),
                namespace="search",
            ),
            "extraction": JsonKVStorage(
                os.path.join(working_dir, "data", "graphgen", f"{unique_id}"),
                namespace="extraction",
            ),
            "qa": JsonListStorage(
                os.path.join(working_dir, "data", "graphgen", f"{unique_id}"),
                namespace="qa",
            ),
        }

    def get_storage(self, name: str) -> Any:
        return self.storages.get(name)
