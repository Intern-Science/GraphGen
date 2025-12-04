from graphgen.models import JsonKVStorage, NetworkXStorage


class StorageFactory:
    """
    Factory class to create storage instances based on backend.
    Supported backends:
        kv_storage(key-value storage):
            - json_kv: JsonKVStorage
        graph_storage:
            - networkx: NetworkXStorage (graph storage)
    """

    @staticmethod
    def create_storage(backend: str, working_dir: str, namespace: str):
        if backend == "json_kv":
            return JsonKVStorage(working_dir, namespace=namespace)

        if backend == "networkx":
            return NetworkXStorage(working_dir, namespace=namespace)

        raise NotImplementedError(
            f"Storage backend '{backend}' is not implemented yet."
        )


def init_storage(backend: str, working_dir: str, namespace: str):
    return StorageFactory.create_storage(backend, working_dir, namespace)
