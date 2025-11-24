from typing import List, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader


class ParquetReader(BaseReader):
    """
    Read parquet files, requiring the schema to be restored to List[Dict[str, Any]].
    Columns:
    - type: The type of the document (e.g., "text", "image", etc.)
    - if type is "text", "content" column must be present.
    """

    def read(
        self,
        input_path: Union[str, List[str]],
        parallelism: int = None,
    ) -> Dataset:
        """
        Read Parquet files using Ray Data.

        :param input_path: Path to Parquet file or list of Parquet files.
        :param parallelism: Number of blocks for Ray Dataset reading.
        :return: Ray Dataset containing validated documents.
        """
        if not ray.is_initialized():
            ray.init()

        ds = ray.data.read_parquet(input_path, override_num_blocks=parallelism)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds
