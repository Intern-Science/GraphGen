from pathlib import Path
from typing import Any, Iterator, Optional

# rocksdict is a lightweight C wrapper around RocksDB for Python, pylint may not recognize it
# pylint: disable=no-name-in-module
from rocksdict import Rdict


class RocksDBCache:
    def __init__(self, cache_dir: str):
        self.db_path = Path(cache_dir)
        self.db = Rdict(str(self.db_path))

    def get(self, key: str) -> Optional[Any]:
        return self.db.get(key)

    def set(self, key: str, value: Any):
        self.db[key] = value

    def delete(self, key: str):
        try:
            del self.db[key]
        except KeyError:
            # If the key does not exist, do nothing (deletion is idempotent for caches)
            pass

    def close(self):
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
            self.db = None

    def __del__(self):
        # Ensure the database is closed when the object is destroyed
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self) -> Iterator[str]:
        return iter(self.db.keys())
