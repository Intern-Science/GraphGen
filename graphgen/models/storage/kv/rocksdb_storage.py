import os
from dataclasses import dataclass
from typing import Any, Dict, List, Set

# rocksdict is a lightweight C wrapper around RocksDB for Python, pylint may not recognize it
# pylint: disable=no-name-in-module
from rocksdict import Rdict

from graphgen.bases.base_storage import BaseKVStorage
from graphgen.utils import logger


@dataclass
class RocksDBKVStorage(BaseKVStorage):
    _db: Rdict = None
    _db_path: str = None

    def __post_init__(self):
        self._db_path = os.path.join(self.working_dir, f"{self.namespace}.db")
        self._db = Rdict(self._db_path)
        logger.info("Load KV (RocksDB) %s at %s", self.namespace, self._db_path)

    @property
    def data(self):
        return self._db

    def all_keys(self) -> List[str]:
        return list(self._db.keys())

    def index_done_callback(self):
        self._db.flush()
        logger.info("RocksDB flushed for %s", self.namespace)

    def get_by_id(self, id: str) -> Any:
        return self._db.get(id, None)

    def get_by_ids(self, ids: List[str], fields: List[str] = None) -> List[Any]:
        result = []
        for index in ids:
            item = self._db.get(index, None)
            if item is None:
                result.append(None)
                continue

            if fields is None:
                result.append(item)
            else:
                result.append({k: v for k, v in item.items() if k in fields})
        return result

    def get_all(self) -> Dict[str, Dict]:
        return dict(self._db)

    def filter_keys(self, data: List[str]) -> Set[str]:
        return {s for s in data if s not in self._db}

    def upsert(self, data: Dict[str, Any]):
        left_data = {}
        for k, v in data.items():
            if k not in self._db:
                left_data[k] = v

        if left_data:
            for k, v in left_data.items():
                self._db[k] = v

            # if left_data is very large, it is recommended to use self._db.write_batch() for optimization

        return left_data

    def drop(self):
        self._db.close()
        Rdict.destroy(self._db_path)
        self._db = Rdict(self._db_path)
        logger.info("Dropped RocksDB %s", self.namespace)

    def close(self):
        if self._db:
            self._db.close()
