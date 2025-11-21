from typing import List, Union

import ray
from ray.data import Dataset

from graphgen.bases.base_reader import BaseReader


class JSONReader(BaseReader):
    """
    Reader for JSON files.
    Columns:
        - type: The type of the document (e.g., "text", "image", etc.)
        - if type is "text", "content" column must be present.
    """

    def read(
        self,
        input_path: Union[str, List[str]],
        parallelism: int = 4,
    ) -> Dataset:
        """
        Read JSON file and return Ray Dataset.
        :param input_path: Path to JSON file or list of JSON files.
        :param parallelism: Number of parallel workers for reading files.
        :return: Ray Dataset containing validated and filtered data.
        """

        ds = ray.data.read_json(input_path, parallelism=parallelism)
        ds = ds.map_batches(self._validate_batch, batch_format="pandas")
        ds = ds.filter(self._should_keep_item)
        return ds
